"""Provider response boundary: retries, stream assembly and honest usage.

Tool arguments are validated as strict JSON objects here. Tool-specific argument
schemas and authorization belong to the dispatcher, before any execution.
"""

import inspect
import json
import sys
from typing import Awaitable, Callable, Optional

from openai import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletionMessage

from app.exceptions import ProviderResponseError, StreamInterruptedError


EventCallback = Callable[[dict], Awaitable[None]]


def is_transient_provider_error(error: BaseException) -> bool:
    """Retry transport failures, 408, 429 and 5xx; never arbitrary exceptions."""
    if isinstance(error, APIConnectionError):
        return True  # APITimeoutError derives from APIConnectionError.
    return isinstance(error, APIStatusError) and (
        error.status_code in (408, 429) or 500 <= error.status_code <= 599
    )


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def validate_message(message, finish_reason, tools=None) -> ChatCompletionMessage:
    """Require a complete envelope and strict JSON, without claiming schema validation."""
    if finish_reason in {"length", "content_filter", "max_tokens", "guardrail_intervened"}:
        raise ProviderResponseError(
            "Provider did not finish the response.", "provider_truncated"
        )
    if finish_reason not in {"stop", "tool_calls", "end_turn", "tool_use"}:
        raise ProviderResponseError("Missing or unsupported response finish reason.")
    calls = _field(message, "tool_calls") or []
    data = {
        "role": _field(message, "role"),
        "content": _field(message, "content"),
        "tool_calls": [
            {
                "id": _field(call, "id"),
                "type": _field(call, "type"),
                "function": {
                    "name": _field(_field(call, "function"), "name"),
                    "arguments": _field(_field(call, "function"), "arguments"),
                },
            }
            for call in calls
        ] or None,
    }
    try:
        result = ChatCompletionMessage.model_validate(data, strict=True)
    except ValueError as error:
        raise ProviderResponseError("Invalid assistant message envelope.") from error
    if not result.tool_calls and not result.content:
        raise ProviderResponseError("Empty assistant response.")
    if finish_reason in {"tool_calls", "tool_use"} and not result.tool_calls:
        raise ProviderResponseError("Tool finish reason without a tool call.")
    available = {_field(_field(tool, "function"), "name") for tool in tools or []}
    seen_ids = set()
    for call in result.tool_calls or []:
        if (
            not call.id.strip()
            or call.id in seen_ids
            or call.function.name not in available
        ):
            raise ProviderResponseError("Missing/duplicate tool ID or unoffered tool name.")
        seen_ids.add(call.id)
        try:
            arguments = json.loads(
                call.function.arguments,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, TypeError, RecursionError) as error:
            raise ProviderResponseError(
                "Tool arguments must be complete, strict JSON."
            ) from error
        if not isinstance(arguments, dict):
            raise ProviderResponseError("Tool arguments must be a JSON object.")
    return result


def record_usage(llm, usage, estimated_input_tokens: int) -> None:
    """Keep provider totals separate from estimates reserved for the input budget."""
    prompt = _field(usage, "prompt_tokens")
    completion = _field(usage, "completion_tokens")
    prompt = prompt if type(prompt) is int and prompt >= 0 else None
    completion = completion if type(completion) is int and completion >= 0 else None
    complete_usage = prompt is not None and completion is not None
    absent_usage = prompt is None and completion is None
    llm.last_usage = {
        "status": "available" if complete_usage else "unavailable" if absent_usage else "partial",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if complete_usage else None,
        "budget_status": "provider_reported" if prompt is not None else "estimated",
        "estimated_input_tokens": estimated_input_tokens if prompt is None else None,
    }
    if prompt is not None:
        llm.total_input_tokens += prompt
    else:
        llm.unreported_input_tokens = (
            getattr(llm, "unreported_input_tokens", 0) + estimated_input_tokens
        )
    if completion is not None:
        llm.total_completion_tokens += completion


class _StreamAssembly:
    def __init__(self):
        self.text = []
        self.calls = {}
        self.finish_reason = None
        self.usage = None
        self.emitted = False

    def add_call(self, delta):
        index = _field(delta, "index")
        if type(index) is not int or index < 0:
            raise ProviderResponseError("Tool delta is missing a valid index.")
        call = self.calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        kind = _field(delta, "type")
        if kind is not None and kind != "function":
            raise ProviderResponseError("Unsupported streamed tool type.")
        if _field(delta, "id"):
            call["id"] += _field(delta, "id")
        function = _field(delta, "function")
        for name in ("name", "arguments"):
            fragment = _field(function, name)
            if fragment is not None:
                call["function"][name] += fragment

    async def consume(self, stream, on_event: Optional[EventCallback]):
        try:
            async for chunk in stream:
                if _field(chunk, "usage") is not None:
                    self.usage = _field(chunk, "usage")
                for choice in _field(chunk, "choices", []):
                    if _field(choice, "index", 0) != 0:
                        raise ProviderResponseError("Only one streamed choice is supported.")
                    delta = _field(choice, "delta")
                    content = _field(delta, "content")
                    tool_calls = _field(delta, "tool_calls") or []
                    if self.finish_reason is not None and (content or tool_calls):
                        raise ProviderResponseError(
                            "Response data arrived after the finish marker."
                        )
                    if content:
                        self.text.append(content)
                        if on_event is not None:
                            self.emitted = True
                            await on_event({"type": "model_text_delta", "text": content})
                    for call in tool_calls:
                        self.add_call(call)
                    if _field(choice, "finish_reason") is not None:
                        self.finish_reason = _field(choice, "finish_reason")
        except ProviderResponseError:
            raise
        except Exception as error:
            if self.emitted:
                raise StreamInterruptedError() from error
            raise
        finally:
            active_error = sys.exc_info()[1]
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception as close_error:
                    if active_error is not None:
                        active_error.add_note(
                            f"Stream cleanup also failed: {type(close_error).__name__}"
                        )
                    elif self.emitted:
                        raise StreamInterruptedError() from close_error
                    else:
                        raise

    def message(self):
        return {
            "role": "assistant",
            "content": "".join(self.text) or None,
            "tool_calls": [self.calls[index] for index in sorted(self.calls)] or None,
        }


async def request_message(
    llm, params, input_tokens: int, tools=None, on_event: Optional[EventCallback] = None
) -> ChatCompletionMessage:
    """Issue one request. Retry policy is owned by the public LLM methods."""
    assembly = _StreamAssembly()
    try:
        response = await llm.client.chat.completions.create(**params)
        if params.get("stream"):
            await assembly.consume(response, on_event)
            return validate_message(assembly.message(), assembly.finish_reason, tools)
        assembly.usage = _field(response, "usage")
        choices = _field(response, "choices") or []
        if len(choices) != 1:
            raise ProviderResponseError("Expected exactly one assistant response.")
        choice = choices[0]
        return validate_message(
            _field(choice, "message"), _field(choice, "finish_reason"), tools
        )
    finally:
        record_usage(llm, assembly.usage, input_tokens)


async def print_text_event(event: dict) -> None:
    """Retain the legacy text-stream console interface for ask/ask_with_images."""
    print(event["text"], end="", flush=True)
