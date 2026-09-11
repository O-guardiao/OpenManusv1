import asyncio
import hashlib
import json
import time
from typing import Any, List, Optional, Union

from pydantic import Field, PrivateAttr, model_validator

from app.agent.react import ReActAgent
from app.config import config
from app.exceptions import TokenLimitExceeded
from app.harness.context import ContextWindow
from app.logger import logger
from app.oak.formal_runtime import (
    BudgetExceeded,
    BudgetLimits,
    EffectConflict,
    approval_reference,
)
from app.oak.runtime import OakRuntime
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection
from app.tool.base import tool_result_failed
from app.tool.policy import ToolPolicy


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_policy: ToolPolicy = Field(
        default_factory=lambda: ToolPolicy.guarded(
            audit_log_path=config.workspace_root / "audit" / "toolcall-events.jsonl"
        )
    )
    oak_runtime: OakRuntime = Field(default_factory=OakRuntime)
    _current_base64_image: Optional[str] = None
    _tool_provenance: dict[str, Any] = PrivateAttr(default_factory=dict)
    _owns_prestarted_oak_task: bool = PrivateAttr(default=False)
    _prestarted_request_sha256: str = PrivateAttr(default="")
    _pending_oak_status: str = PrivateAttr(default="")
    _declared_failure: bool = PrivateAttr(default=False)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None
    defer_oak_finish: bool = False
    context_window_tokens: int = Field(default=32768, gt=0)
    stream_responses: bool = False
    run_observer: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def align_oak_session(self) -> "ToolCallAgent":
        """Keep policy receipts and the in-memory evidence graph in one session."""

        self.oak_runtime.session_id = self.tool_policy.session_id
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.run_observer:
            await self.run_observer.checkpoint()
        tools = self.available_tool_params()
        system_msgs = [Message.system_message(self.system_prompt)] if self.system_prompt else []
        # Offline lifecycle fixtures have no initialized tokenizer. UTF-8 bytes
        # provide a conservative fallback, never a fabricated billing count.
        count_tokens = (self.llm.count_tokens if getattr(self.llm, 'tokenizer', None)
                        else lambda text: len(text.encode('utf-8')))
        guidance = [Message.user_message(self.next_step_prompt)] if self.next_step_prompt else []
        reserve = (ContextWindow.token_cost(system_msgs + guidance, count_tokens=count_tokens)
                   + count_tokens(json.dumps(tools, ensure_ascii=False))
                   + int(getattr(self.llm, 'max_tokens', 1024)) + 256)
        messages = ContextWindow.select(
            self.messages, count_tokens=count_tokens, max_tokens=self.context_window_tokens,
            reserve_tokens=reserve, max_messages=self.memory.max_messages,
        )
        if self.run_observer:
            self.run_observer.emit('context_projected', transcript_messages=len(self.messages),
                projected_messages=len(messages), local_token_limit=self.context_window_tokens)
        messages = [*messages, *guidance]
        if self.oak_runtime.task_active:
            try:
                cycle_id = f"agent:{self.name}"
                self.oak_runtime.consume_budget("steps", cycle_id=cycle_id)
                self.oak_runtime.consume_budget("model_calls", cycle_id=cycle_id)
                context_bytes = len(json.dumps(
                    [message.to_dict() for message in [*system_msgs, *messages]],
                    ensure_ascii=False,
                ).encode('utf-8')) + len(json.dumps(tools).encode('utf-8'))
                self.oak_runtime.observe_context_bytes(
                    context_bytes, cycle_id=cycle_id
                )
            except BudgetExceeded as error:
                logger.warning(f"Global OaK budget halted {self.name}: {error}")
                self.state = AgentState.FINISHED
                return False
        try:
            # Get response with tool options
            stream_options = {}
            if self.stream_responses:
                stream_options['stream'] = True
                if self.run_observer:
                    stream_options['on_event'] = self.run_observer.on_model_event
            response = await self.llm.ask_tool(
                messages=messages,
                system_msgs=system_msgs or None,
                tools=tools,
                tool_choice=self.tool_choices,
                **stream_options,
            )
        except ValueError:
            raise
        except Exception as e:
            # Check if this is a RetryError containing TokenLimitExceeded
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        if self.run_observer:
            self.run_observer.emit('model_response', tool_calls=len(self.tool_calls),
                usage=getattr(self.llm, 'last_usage', None))
        if self.oak_runtime.task_active:
            token_total = int(getattr(self.llm, "total_input_tokens", 0)) + int(
                getattr(self.llm, "total_completion_tokens", 0)
            )
            try:
                self.oak_runtime.observe_token_total(
                    token_total, cycle_id=f"agent:{self.name}"
                )
            except BudgetExceeded as error:
                logger.warning(f"Global OaK token budget halted {self.name}: {error}")
                self.state = AgentState.FINISHED
                return False
        content = response.content if response and response.content else ""

        # Log response info
        logger.info(f"✨ {self.name} returned {len(content)} characters of reasoning")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            argument_digest = hashlib.sha256(
                (tool_calls[0].function.arguments or "{}").encode("utf-8")
            ).hexdigest()
            logger.info(f"🔧 First tool argument digest: {argument_digest}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    self.state = AgentState.FINISHED
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # A complete text-only reply is terminal. The acceptance/receipt gate
            # still determines success; more steps cannot improve an absent action.
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                if not content:
                    raise ValueError('Provider returned neither content nor tools')
                if self.run_observer:
                    before = len(self.messages)
                    await self.run_observer.checkpoint()
                    if len(self.messages) != before:
                        # Steering received during inference gets another turn;
                        # the just-returned answer cannot settle the new request.
                        return False
                self.state = AgentState.FINISHED
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            raise

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        image_messages = []
        for command in self.tool_calls:
            if self.run_observer:
                await self.run_observer.checkpoint(apply_steering=False)
            # Reset base64_image for each tool call
            self._current_base64_image = None

            result = await self.execute_tool(command)

            logger.info(
                f"🎯 Tool '{command.function.name}' returned {len(result)} characters"
            )

            # Add tool response to memory
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                provenance=self._tool_provenance.pop(command.id, None),
            )
            self.memory.add_message(tool_msg)
            if self._current_base64_image:
                image_messages.append(
                    Message.user_message(
                        content=f"Image returned by {command.function.name}:",
                        base64_image=self._current_base64_image,
                    )
                )
            results.append(result)

        self.memory.add_messages(image_messages)
        if self.run_observer:
            before = len(self.messages)
            await self.run_observer.checkpoint()
            if len(self.messages) != before and self.state == AgentState.FINISHED:
                self.state = AgentState.RUNNING
                self._declared_failure = False
        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute a single tool call with robust error handling"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        raw_arguments = command.function.arguments or "{}"
        arguments_sha256 = hashlib.sha256(raw_arguments.encode("utf-8")).hexdigest()
        if self.oak_runtime.task_active:
            try:
                self.oak_runtime.consume_budget(
                    "tool_calls", cycle_id=f"agent:{self.name}"
                )
            except BudgetExceeded as error:
                return f"Error: Tool '{name}' was not executed: {error}."
        if name not in self.available_tools.tool_map:
            first_observation = True
            if self.oak_runtime.task_active:
                first_observation = self.oak_runtime.record_unavailable_alternative(
                    name,
                    reason="not_in_session_tool_map",
                )
            try:
                self._record_tool_event(
                    name=name,
                    decision="denied",
                    status="not_executed",
                    reason=(
                        "unknown_tool"
                        if first_observation
                        else "permanently_unavailable_retry_rejected"
                    ),
                    arguments_sha256=arguments_sha256,
                )
            except OSError:
                logger.error("Unknown tool denied; audit trail unavailable")
            return f"Error: Unknown tool '{name}'"

        try:
            args = json.loads(raw_arguments)
            if not isinstance(args, dict):
                raise json.JSONDecodeError("tool arguments must be an object", raw_arguments, 0)
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            try:
                self._record_tool_event(
                    name=name,
                    decision="denied",
                    status="invalid_arguments",
                    reason="invalid_json",
                    arguments_sha256=arguments_sha256,
                )
            except OSError:
                logger.error("Invalid tool arguments could not be audited")
            self.oak_runtime.record_tool_outcome(name, "invalid_arguments", "invalid_json")
            logger.error(f"📝 Invalid JSON arguments for '{name}' ({arguments_sha256})")
            return f"Error: {error_msg}"

        if not self.tool_policy.allows(name):
            try:
                self._record_tool_event(
                    name=name,
                    decision="denied",
                    status="not_executed",
                    reason="not_granted_for_session",
                    arguments_sha256=arguments_sha256,
                )
            except OSError:
                logger.error("Tool denied; audit trail unavailable")
            return (
                f"Error: Tool '{name}' is not authorized for this session. "
                f"Restart with --allow-tool {name}."
            )

        exact_approved = self.tool_policy.allows_tainted(name, args)
        try:
            decision = self.oak_runtime.authorize_tool(
                name,
                args,
                granted=True,
                tainted_sink_approved=exact_approved,
                principal=self.tool_policy.principal,
                policy_version=self.tool_policy.policy_version,
            )
        except Exception as error:
            self.oak_runtime.record_gate_failure(
                name, args, error_type=type(error).__name__
            )
            try:
                self._record_tool_event(
                    name=name,
                    decision="denied",
                    status="not_executed",
                    reason="guard_error",
                    arguments_sha256=arguments_sha256,
                )
            except OSError:
                logger.error("Gate failure could not be audited")
            return f"Error: Tool '{name}' was not executed: guard evaluation failed."
        try:
            self.tool_policy.record_oak_guard(
                agent_name=self.name,
                tool_name=name,
                allowed=decision.allowed,
                reason=decision.reason,
                source=decision.profile.source,
                sink=decision.profile.sink,
                risk_score=decision.profile.risk_score,
                taint_count=len(decision.taint_ids),
                outcome=decision.outcome,
                arguments_sha256=decision.arguments_sha256,
                approval_key_value=decision.approval_key,
                state_version=decision.state_version,
            )
        except OSError:
            logger.error("Tool blocked because the OaK decision could not be audited")
            self.oak_runtime.record_gate_failure(
                name,
                args,
                error_type="GuardAuditUnavailable",
            )
            return f"Error: Tool '{name}' was not executed: audit trail unavailable."
        if not decision.allowed:
            reference = approval_reference(name, args)
            return (
                f"Error: Tool '{name}' requires approval for these exact arguments "
                f"because untrusted source data is active. Approval reference: {reference}."
            )

        try:
            self._record_tool_event(
                name=name,
                decision="allowed",
                status="authorized",
                reason="granted_for_session",
                arguments_sha256=arguments_sha256,
            )
        except OSError:
            logger.error("Authorized tool was blocked because the audit trail is unavailable")
            self.oak_runtime.record_gate_failure(
                name,
                args,
                error_type="AuthorizationAuditUnavailable",
            )
            return f"Error: Tool '{name}' was not executed: audit trail unavailable."

        effect = None
        execution_args = dict(args)
        if decision.profile.sink:
            try:
                effect = self.oak_runtime.begin_effect(
                    operation_id=command.id,
                    principal=self.tool_policy.principal,
                    tool_name=name,
                    arguments=args,
                    gate_event_id=decision.event_id,
                )
            except (BudgetExceeded, EffectConflict) as error:
                return f"Error: Tool '{name}' was not executed: {error}."
            if effect.status != "intent_logged":
                return (
                    f"Observed prior outcome for `{name}`; the effect was not dispatched again."
                )
            try:
                self.tool_policy.record_effect_intent(
                    agent_name=self.name,
                    operation_id=effect.operation_id,
                    tool_name=name,
                    arguments_sha256=effect.arguments_sha256,
                    idempotency_key=effect.idempotency_key,
                    gate_event_id=decision.event_id,
                )
                if self.run_observer:
                    self.run_observer.effect_intent(
                        effect.operation_id, name, effect.arguments_sha256
                    )
                execution_args = self._inject_idempotency_key(
                    name, execution_args, effect.idempotency_key
                )
                self.oak_runtime.dispatch_effect(effect.operation_id)
            except (OSError, BudgetExceeded, EffectConflict) as error:
                self.oak_runtime.observe_effect(
                    effect.operation_id, "failed_before_effect"
                )
                return f"Error: Tool '{name}' was not executed: {type(error).__name__}."

        effect_observed = False
        started_at = time.perf_counter()
        try:
            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(
                name=name, tool_input=execution_args
            )
            status = "failed" if tool_result_failed(result) else "succeeded"
            if name == 'terminate' and args.get('status') == 'failure':
                self._declared_failure = True
            self.oak_runtime.record_tool_outcome(
                name,
                status,
                "tool_result",
                operation_id=effect.operation_id if effect else None,
            )
            observe_limit = (
                int(self.max_observe)
                if isinstance(self.max_observe, int) and not isinstance(self.max_observe, bool)
                else None
            )
            integrated = self.oak_runtime.integrate_tool_result(
                name,
                str(result),
                max_content_length=observe_limit,
            )
            try:
                self._record_tool_event(
                    name=name,
                    decision="allowed",
                    status=status,
                    reason="tool_result",
                    arguments_sha256=arguments_sha256,
                    duration_ms=round((time.perf_counter() - started_at) * 1000),
                )
                self.tool_policy.record_provenance(
                    agent_name=self.name,
                    tool_name=name,
                    content_sha256=integrated.provenance.content_sha256,
                    normalized_sha256=integrated.provenance.normalized_sha256,
                    carrier_type=integrated.provenance.carrier_type,
                    trust_tier=integrated.provenance.trust_tier,
                    taint_count=len(integrated.provenance.taint_ids),
                    transformations=integrated.provenance.transformations,
                )
                if effect is not None:
                    effect_status = "unknown" if status == "failed" else "succeeded"
                    self.tool_policy.record_effect_outcome(
                        agent_name=self.name,
                        operation_id=effect.operation_id,
                        tool_name=name,
                        status=effect_status,
                        idempotency_key=effect.idempotency_key,
                    )
                    self.oak_runtime.observe_effect(
                        effect.operation_id, effect_status
                    )
                    if self.run_observer:
                        self.run_observer.effect_outcome(
                            effect.operation_id, effect_status, defer_until_observation=True
                        )
                    effect_observed = True
            except OSError:
                logger.error(
                    "Tool executed but its completion audit event could not be written"
                )
                if effect is not None and not effect_observed:
                    try:
                        self.oak_runtime.observe_effect(
                            effect.operation_id, "unknown"
                        )
                        effect_observed = True
                    except (EffectConflict, ValueError) as observation_error:
                        self.oak_runtime.record_gate_failure(
                            name,
                            args,
                            error_type=type(observation_error).__name__,
                        )
                return (
                    f"Error: Tool '{name}' may have executed, but its outcome is "
                    "UNKNOWN because the audit trail failed. Do not retry automatically."
                )

            self._tool_provenance[command.id] = integrated.provenance

            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{integrated.model_content}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except Exception as e:
            error_type = type(e).__name__
            self.oak_runtime.record_tool_outcome(
                name,
                "failed",
                type(e).__name__,
                operation_id=effect.operation_id if effect else None,
            )
            if effect is not None and not effect_observed:
                try:
                    self.oak_runtime.observe_effect(
                        effect.operation_id, "unknown"
                    )
                    effect_observed = True
                except (EffectConflict, ValueError) as observation_error:
                    self.oak_runtime.record_gate_failure(
                        name,
                        args,
                        error_type=type(observation_error).__name__,
                    )
                try:
                    self.tool_policy.record_effect_outcome(
                        agent_name=self.name,
                        operation_id=effect.operation_id,
                        tool_name=name,
                        status="unknown",
                        idempotency_key=effect.idempotency_key,
                    )
                except OSError:
                    logger.error("Unknown effect outcome could not be persisted")
            integrated_error = self.oak_runtime.integrate_tool_result(
                name,
                f"{error_type}: {str(e)}",
                max_content_length=(
                    int(self.max_observe)
                    if isinstance(self.max_observe, int)
                    and not isinstance(self.max_observe, bool)
                    else None
                ),
            )
            self._tool_provenance[command.id] = integrated_error.provenance
            try:
                self._record_tool_event(
                    name=name,
                    decision="allowed",
                    status="failed",
                    reason=error_type,
                    arguments_sha256=arguments_sha256,
                    duration_ms=round((time.perf_counter() - started_at) * 1000),
                )
                self.tool_policy.record_provenance(
                    agent_name=self.name,
                    tool_name=name,
                    content_sha256=integrated_error.provenance.content_sha256,
                    normalized_sha256=integrated_error.provenance.normalized_sha256,
                    carrier_type=integrated_error.provenance.carrier_type,
                    trust_tier=integrated_error.provenance.trust_tier,
                    taint_count=len(integrated_error.provenance.taint_ids),
                    transformations=integrated_error.provenance.transformations,
                )
            except OSError:
                logger.error("Tool failed and the completion audit event could not be written")
            logger.error(f"Tool '{name}' failed with {error_type}")
            return f"Error: {integrated_error.model_content}"

    def available_tool_params(self) -> List[dict]:
        """Expose granted capabilities; the execution gate still owns authority.

        Exact argument approval cannot be decided from a JSON schema. Keeping a
        granted sink visible lets the first denied call return its reviewable
        reference without ever dispatching the effect.
        """

        return [
            tool.to_param()
            for tool in self.available_tools
            if self.tool_policy.allows(tool.name)
        ]

    def _inject_idempotency_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """Pass the operation key only when the declared tool contract accepts it.

        The runtime still owns and audits the key when a legacy adapter has no
        idempotency field.  Silently adding an undeclared argument would break
        existing tool contracts and turn the safety layer into a failure source.
        """

        tool = self.available_tools.get_tool(tool_name)
        schema = getattr(tool, "parameters", None) or {}
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return arguments
        for field_name in ("idempotency_key", "idempotencyKey", "operation_id"):
            if field_name in properties and field_name not in arguments:
                return {**arguments, field_name: key}
        return arguments

    def _record_tool_event(
        self,
        *,
        name: str,
        decision: str,
        status: str,
        reason: str,
        arguments_sha256: str,
        duration_ms: int | None = None,
    ) -> None:
        self.tool_policy.record(
            agent_name=self.name,
            tool_name=name,
            decision=decision,
            status=status,
            reason=reason,
            arguments_sha256=arguments_sha256,
            duration_ms=duration_ms,
        )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """Clean up resources used by the agent's tools."""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        failures: list[tuple[str, str]] = []
        for tool_name, tool_instance in reversed(
            tuple(self.available_tools.tool_map.items())
        ):
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    failures.append((tool_name, type(e).__name__))
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        if failures:
            summary = ", ".join(
                f"{tool_name}:{error_type}"
                for tool_name, error_type in failures
            )
            raise RuntimeError(f"tool cleanup failed: {summary}")
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """Run with one OaK task lifecycle; the caller owns cleanup."""

        prestarted = (
            bool(request)
            and self._owns_prestarted_oak_task
            and self.oak_runtime.task_active
        )
        if prestarted:
            request_sha256 = hashlib.sha256((request or "").encode("utf-8")).hexdigest()
            if request_sha256 != self._prestarted_request_sha256:
                raise RuntimeError("prestarted OaK task belongs to a different request")
        owns_task = prestarted or (bool(request) and not self.oak_runtime.task_active)
        if owns_task and not prestarted:
            if self.oak_runtime.budget_limits == BudgetLimits():
                self.oak_runtime.configure_budget(BudgetLimits.for_steps(self.max_steps))
            self.oak_runtime.begin_task(request or "")
        if owns_task:
            self._restore_memory_provenance()
        try:
            result = await super().run(request)
            requested_status = (
                "max_steps_reached"
                if self._step_limit_reached
                else "failed" if self._declared_failure else "completed"
            )
            if owns_task:
                self.oak_runtime.record_final_response(result)
                if self.defer_oak_finish:
                    self._pending_oak_status = requested_status
                else:
                    self.oak_runtime.finish_task(requested_status)
                    self._clear_prestarted_task()
            return result
        except BaseException:
            if owns_task and self.oak_runtime.task_active:
                if self.defer_oak_finish:
                    self._pending_oak_status = "failed"
                else:
                    self.oak_runtime.finish_task("failed")
                    self._clear_prestarted_task()
            raise

    def start_owned_oak_task(self, request: str) -> None:
        """Start a root task before provider initialization or other effects."""

        if self.oak_runtime.task_active:
            raise RuntimeError("an OaK root task is already active")
        if self.oak_runtime.budget_limits == BudgetLimits():
            self.oak_runtime.configure_budget(BudgetLimits.for_steps(self.max_steps))
        self.oak_runtime.begin_task(request)
        self._restore_memory_provenance()
        self._owns_prestarted_oak_task = True
        self._prestarted_request_sha256 = hashlib.sha256(
            request.encode("utf-8")
        ).hexdigest()
        self._pending_oak_status = ""

    def _restore_memory_provenance(self) -> None:
        """Reattach trust metadata for model-visible messages to this task."""

        if not self.oak_runtime.task_active:
            return
        for message in self.messages:
            if message.provenance is not None:
                self.oak_runtime.restore_provenance(message.provenance)

    def finalize_oak_task(self, status: str | None = None) -> str:
        """Seal a deferred root task after cleanup has become observable."""

        if not self.oak_runtime.task_active:
            return self.oak_runtime.last_task_status or status or "failed"
        requested_status = status or self._pending_oak_status or "failed"
        actual_status = self.oak_runtime.finish_task(requested_status)
        self._clear_prestarted_task()
        return actual_status

    def _clear_prestarted_task(self) -> None:
        self._owns_prestarted_oak_task = False
        self._prestarted_request_sha256 = ""
        self._pending_oak_status = ""
        self._declared_failure = False
