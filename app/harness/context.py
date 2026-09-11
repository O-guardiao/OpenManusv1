"""Deterministic context selection over the complete legacy Message transcript.

Indices in notices are zero based. Selection never rewrites the transcript and
does not claim to summarize omitted material. Callers reserve system/tool schema
and completion tokens separately. External user-role carriers must have provenance.
"""

import json
from dataclasses import dataclass
from typing import Callable

from app.schema import Message


class ContextBudgetExceeded(ValueError):
    """Required context or a protocol-safe latest tool turn cannot fit."""


@dataclass
class _Block:
    indices: list[int]
    required: bool = False
    tool_turn: bool = False


def _blocks(messages: list[Message]) -> list[_Block]:
    blocks = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.tool_calls:
            ids = [call.id for call in message.tool_calls]
            if len(ids) != len(set(ids)):
                raise ContextBudgetExceeded("Duplicate tool call IDs in transcript")
            pending = set(ids)
            indices = [index]
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                result = messages[index]
                if result.tool_call_id not in pending:
                    break
                pending.remove(result.tool_call_id)
                indices.append(index)
                index += 1
            if pending:
                raise ContextBudgetExceeded(
                    f"Incomplete tool turn at transcript index {indices[0]}"
                )
            # Legacy image carriers are appended after all results in act().
            while index < len(messages):
                image = messages[index]
                if not (image.role == "user" and image.base64_image
                        and (image.content or "").startswith("Image returned by ")):
                    break
                indices.append(index)
                index += 1
            blocks.append(_Block(indices, tool_turn=True))
            continue
        if message.role != "tool":
            required = message.role == "system" or (
                message.role == "user" and message.provenance is None
            )
            blocks.append(_Block([index], required=required))
        # An already orphaned historical result is excluded, explicitly noticed.
        index += 1
    return blocks


def _ranges(indices: list[int]) -> str:
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)


class ContextWindow:
    @staticmethod
    def token_cost(messages: list[Message], *, count_tokens: Callable[[str], int]) -> int:
        """Conservative text serialization plus framing and image allowance.

        Without a provider image estimator, charge at least 16,384 tokens per
        image and its full encoded byte length. This is a local admission bound,
        not a claim of exact provider billing or arbitrary-model image capacity.
        """
        total = 16
        for message in messages:
            payload = message.to_dict()
            image = payload.pop("base64_image", None)
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            total += max(0, count_tokens(encoded)) + 32
            if image:
                total += max(16384, len(image.encode("utf-8")))
        return total

    @staticmethod
    def select(
        messages: list[Message],
        *,
        count_tokens: Callable[[str], int],
        max_tokens: int,
        reserve_tokens: int = 0,
        max_messages: int = 100,
    ) -> list[Message]:
        """Pin genuine human/system messages; retain complete recent tool blocks.

        Returns deep copies. The latest tool block is retained, using explicit
        excerpts of outputs/images when necessary. Earlier optional blocks are
        admitted newest first, atomically. Required messages are never excerpted.
        Synthetic omission notices occupy message and token budget too.
        """
        if max_tokens < 0 or reserve_tokens < 0 or max_messages < 1:
            raise ValueError("Context budgets must be nonnegative; max_messages >= 1")
        budget = max_tokens - reserve_tokens
        blocks = _blocks(messages)
        selected = {i: messages[i].model_copy(deep=True)
                    for block in blocks if block.required for i in block.indices}

        def render(chosen: dict[int, Message]) -> list[Message]:
            result = [chosen[i] for i in sorted(chosen)]
            omitted = [i for i in range(len(messages)) if i not in chosen]
            if omitted:
                result.append(Message.system_message(
                    "Context selection notice: omitted transcript indices "
                    f"{_ranges(omitted)} (zero based). Full originals remain in "
                    "the transcript; omitted content is not summarized."
                ))
            return result

        def fits(chosen: dict[int, Message]) -> bool:
            result = render(chosen)
            return len(result) <= max_messages and ContextWindow.token_cost(
                result, count_tokens=count_tokens
            ) <= budget

        if not fits(selected):
            raise ContextBudgetExceeded(
                "Human/system requirements plus omission notice exceed context budget"
            )

        optional = [block for block in blocks if not block.required]
        latest_tool = next((block for block in reversed(optional) if block.tool_turn), None)
        if latest_tool is not None:
            candidate = {**selected, **{i: messages[i].model_copy(deep=True)
                                       for i in latest_tool.indices}}
            if not fits(candidate):
                # Preserve all call arguments and result IDs; only output bodies
                # and image payloads may be abbreviated, with original provenance.
                def excerpt(limit: int) -> dict[int, Message]:
                    candidate = dict(selected)
                    for i in latest_tool.indices:
                        item = messages[i].model_copy(deep=True)
                        if item.role == "tool" or item.base64_image:
                            original = item.content or ""
                            if len(original) > limit or item.base64_image:
                                item.content = (
                                    original[:limit]
                                    + f"\n[Context excerpt; transcript index {i} "
                                    "(zero based). Content/image omitted; consult "
                                    "the original transcript for the full result.]"
                                )
                                item.base64_image = None
                        candidate[i] = item
                    return candidate

                candidate = excerpt(0)
                if not fits(candidate):
                    raise ContextBudgetExceeded(
                        "Latest complete tool turn cannot fit even as explicit excerpts"
                    )
                low, high = 0, max(len(messages[i].content or "") for i in latest_tool.indices)
                while low < high:
                    middle = (low + high + 1) // 2
                    trial = excerpt(middle)
                    if fits(trial):
                        low, candidate = middle, trial
                    else:
                        high = middle - 1
            selected = candidate

        for block in reversed(optional):
            if block is latest_tool:
                continue
            candidate = {**selected, **{i: messages[i].model_copy(deep=True)
                                       for i in block.indices}}
            if fits(candidate):
                selected = candidate
        return render(selected)
