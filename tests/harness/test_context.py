import copy

import pytest

from app.schema import Function, Memory, Message, MessageProvenance, ToolCall


def select(messages, **kwargs):
    from app.harness.context import ContextWindow

    return ContextWindow.select(messages, count_tokens=len, **kwargs)


def tool_block(index, size=20):
    calls = [ToolCall(id=f"{index}-{n}", function=Function(name="read", arguments="{}")) for n in range(2)]
    return [
        Message(role="assistant", tool_calls=calls),
        *[Message.tool_message("x" * size, "read", call.id) for call in calls],
    ]


def provenance():
    return MessageProvenance(source="remote", source_kind="tool", trust_tier="untrusted", carrier_type="text", content_sha256="a", normalized_sha256="b", taint_ids=["remote:1"])


def assert_pairs(messages):
    expected = []
    for message in messages:
        if message.tool_calls:
            assert not expected
            expected = [call.id for call in message.tool_calls]
        elif message.role == "tool":
            assert message.tool_call_id in expected
            expected.remove(message.tool_call_id)
        else:
            assert not expected
    assert not expected


def test_transcript_keeps_initial_request_and_late_correction():
    memory = Memory(max_messages=4)
    memory.add_message(Message.user_message("Build with no network access"))
    memory.add_messages([Message.assistant_message(str(n)) for n in range(120)])
    memory.add_message(Message.user_message("Correction: also preserve the license"))
    assert len(memory.messages) == 122
    result = select(memory.messages, max_tokens=1500, max_messages=10)
    assert result[0].content == "Build with no network access"
    assert any(m.content == "Correction: also preserve the license" for m in result)
    assert any("transcript" in (m.content or "") for m in result)


def test_multiple_tool_results_remain_atomic_when_old_blocks_are_omitted():
    messages = [Message.user_message("Do the task"), *tool_block(0), *tool_block(1)]
    result = select(messages, max_tokens=2500, max_messages=5)
    assert_pairs(result)
    assert {m.tool_call_id for m in result if m.role == "tool"} == {"1-0", "1-1"}


def test_large_recent_result_is_explicit_excerpt_without_transcript_mutation():
    messages = [Message.user_message("Inspect"), *tool_block(0, 20000)]
    messages[2].provenance = provenance()
    original = copy.deepcopy(messages)
    result = select(messages, max_tokens=2200)
    assert_pairs(result)
    outputs = [m for m in result if m.role == "tool"]
    assert all("transcript index" in m.content for m in outputs)
    assert outputs[0].provenance == original[2].provenance
    assert messages == original
    outputs[0].content = "mutated selection"
    assert messages == original


def test_required_human_content_never_silently_truncated():
    from app.harness.context import ContextBudgetExceeded

    with pytest.raises(ContextBudgetExceeded):
        select([Message.user_message("requirement" * 1000)], max_tokens=800)
    with pytest.raises(ContextBudgetExceeded):
        select([Message.user_message(str(n)) for n in range(4)], max_tokens=8000, max_messages=3)


def test_external_user_data_is_optional_but_system_and_human_are_preserved():
    messages = [Message.system_message("Rules"), Message.user_message("Request"), Message.user_message("untrusted" * 1000, provenance=provenance()), Message.assistant_message("done")]
    result = select(messages, max_tokens=900)
    assert [m.content for m in result[:2]] == ["Rules", "Request"]
    assert not any(m.content == messages[2].content for m in result)


def test_associated_tool_image_is_grouped_and_budgeted_as_image():
    messages = [Message.user_message("Inspect"), *tool_block(0), Message.user_message("Image returned by read:", base64_image="AA==", provenance=provenance())]
    original = copy.deepcopy(messages)
    result = select(messages, max_tokens=2000)
    assert_pairs(result)
    image = next(m for m in result if "Image returned by" in (m.content or ""))
    assert image.base64_image is None
    assert "transcript index 4" in image.content
    assert image.provenance == messages[4].provenance
    assert messages == original


def test_budget_includes_reservation_and_serialization():
    from app.harness.context import ContextWindow

    result = select([Message.user_message("Hi"), *tool_block(0, 2000)], max_tokens=1500, reserve_tokens=200)
    assert ContextWindow.token_cost(result, count_tokens=len) <= 1300


def test_orphan_tool_result_is_omitted_with_explicit_reference():
    result = select([Message.user_message("Hi"), Message.tool_message("orphan", "read", "missing")], max_tokens=1000)
    assert not any(m.role == "tool" for m in result)
    assert any("1" in (m.content or "") and "transcript" in m.content for m in result)


def test_memory_listener_runs_before_append_and_failure_preserves_memory():
    initial = Message.user_message("loaded")
    memory = Memory(messages=[initial])
    seen = []
    memory.set_listener(lambda message: seen.append((message.content, len(memory.messages))))
    memory.add_messages([Message.user_message("one"), Message.user_message("two")])
    assert seen == [("one", 1), ("two", 2)]
    def fail(message):
        raise OSError("disk full")
    memory.set_listener(fail)
    with pytest.raises(OSError):
        memory.add_message(Message.user_message("not saved"))
    assert len(memory.messages) == 3
    memory.set_listener(None)
    memory.add_message(Message.user_message("no listener"))
    assert len(memory.messages) == 4


def test_incomplete_tool_turn_cannot_send_invalid_protocol_to_provider():
    from app.harness.context import ContextBudgetExceeded

    with pytest.raises(ContextBudgetExceeded, match="Incomplete tool turn"):
        select([Message.user_message("Hi"), *tool_block(0)[:2]], max_tokens=9000)


def test_real_human_image_is_required_and_never_replaced_by_excerpt():
    from app.harness.context import ContextBudgetExceeded

    image = Message.user_message("Use this drawing", base64_image="AA==")
    with pytest.raises(ContextBudgetExceeded):
        select([image], max_tokens=2000)
    result = select([image], max_tokens=20000)
    assert result[0].base64_image == "AA=="
    assert result[0] is not image


def test_loaded_memory_does_not_serialize_listener():
    memory = Memory(messages=[Message.user_message("loaded")])
    memory.set_listener(lambda message: None)
    assert "_listener" not in memory.model_dump()
