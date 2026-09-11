"""Provider boundary tests with real SDK payloads and an offline transport."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import stop_after_attempt, wait_none

from app.exceptions import TokenLimitExceeded
from app.llm import ConservativeByteTokenizer, LLM, TokenCounter


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}},
    "required": ["query"], "additionalProperties": False,
}}}]
METHODS = ["ask", "ask_with_images", "ask_tool"]


def completion(*, content="ready", arguments=None, finish="stop", usage=True):
    message = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [{"id": "call_one", "type": "function", "function": {
            "name": "lookup", "arguments": arguments,
        }}]
    return ChatCompletion.model_validate({
        "id": "chat_offline", "object": "chat.completion", "created": 1,
        "model": "gpt-4o", "choices": [{"index": 0, "message": message,
            "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        if usage else None,
    })


def chunk(*, text=None, calls=None, finish=None, usage=False):
    return ChatCompletionChunk.model_validate({
        "id": "chat_offline", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-4o", "choices": [] if usage else [{"index": 0,
            "delta": {"content": text, "tool_calls": calls}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        if usage else None,
    })


class OfflineStream:
    def __init__(self, *items):
        self.items = iter(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = next(self.items, None)
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        self.closed = True


class OfflineTransport:
    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def llm_for(*outcomes):
    llm = object.__new__(LLM)
    llm.model = "gpt-4o"
    llm.api_type = "openai"
    llm.max_tokens = 500
    llm.temperature = 0
    llm.max_input_tokens = None
    llm.total_input_tokens = 0
    llm.total_completion_tokens = 0
    llm.tokenizer = ConservativeByteTokenizer()
    llm.token_counter = TokenCounter(llm.tokenizer)
    transport = OfflineTransport(*outcomes)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=transport))
    return llm, transport


async def invoke(llm, method, **kwargs):
    call = getattr(LLM, method).retry_with(wait=wait_none(), stop=stop_after_attempt(2))
    if method == "ask_with_images":
        kwargs["images"] = ["data:image/png;base64,AA=="]
    if method == "ask_tool":
        kwargs.setdefault("tools", TOOLS)
    kwargs.setdefault("stream", False)
    return await call(llm, messages=[{"role": "user", "content": "hello"}], **kwargs)


def status_error(code):
    response = httpx.Response(code, request=httpx.Request("POST", "https://offline.invalid"))
    return APIStatusError("offline status", response=response, body=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("error", [ValueError("invalid"), TokenLimitExceeded("budget"),
    status_error(400), status_error(401), status_error(403), status_error(404),
    status_error(422), asyncio.CancelledError()])
async def test_permanent_errors_and_cancel_are_not_retried(method, error):
    llm, transport = llm_for(error, completion())
    with pytest.raises(type(error)):
        await invoke(llm, method)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("error", [status_error(408), status_error(429), status_error(500),
    status_error(503), APIConnectionError(request=httpx.Request("POST", "https://offline.invalid")),
    APITimeoutError(request=httpx.Request("POST", "https://offline.invalid"))])
async def test_transient_errors_retry_and_return_response(method, error):
    llm, transport = llm_for(error, completion())
    result = await invoke(llm, method)
    assert (result.content if method == "ask_tool" else result) == "ready"
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_stream_assembles_interleaved_tools_and_awaits_text_events():
    stream = OfflineStream(
        chunk(text="Searching "),
        chunk(calls=[{"index": 1, "id": "second", "type": "function", "function": {
            "name": "look", "arguments": '{"query":'}},
            {"index": 0, "id": "first", "type": "function", "function": {
            "name": "lookup", "arguments": '{"query":"one'}}]),
        chunk(text="now", calls=[{"index": 0, "function": {"arguments": '"}'}},
            {"index": 1, "function": {"name": "up", "arguments": '"two"}'}}]),
        chunk(finish="tool_calls"), chunk(usage=True),
    )
    llm, transport = llm_for(stream)
    events = []

    async def emit(event):
        await asyncio.sleep(0)
        events.append(event)

    result = await invoke(llm, "ask_tool", stream=True, on_event=emit)
    assert result.content == "Searching now"
    assert [(call.id, call.function.name, call.function.arguments) for call in result.tool_calls] == [
        ("first", "lookup", '{"query":"one"}'), ("second", "lookup", '{"query":"two"}')]
    assert events == [{"type": "model_text_delta", "text": "Searching "},
                      {"type": "model_text_delta", "text": "now"}]
    assert llm.total_input_tokens == 12
    assert llm.total_completion_tokens == 7
    assert transport.requests[0]["stream"] is True
    assert "on_event" not in transport.requests[0]
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("arguments,finish", [('{"query":"x"}', "length"),
    ('{"query":"x"}', "content_filter"), ('{"query":', "tool_calls"),
    ('[]', "tool_calls"), ('{"query":NaN}', "tool_calls"),
    ('{"query":"x","query":"y"}', "tool_calls")])
async def test_incomplete_or_invalid_json_tools_are_never_returned(streaming, arguments, finish):
    outcome = OfflineStream(chunk(calls=[{"index": 0, "id": "call_one", "type": "function",
        "function": {"name": "lookup", "arguments": arguments}}]), chunk(finish=finish)) if streaming else completion(
            content=None, arguments=arguments, finish=finish)
    llm, transport = llm_for(outcome)
    with pytest.raises(ValueError):
        await invoke(llm, "ask_tool", stream=streaming)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
async def test_absent_usage_is_explicit_and_does_not_fabricate_totals(method):
    llm, _ = llm_for(completion(usage=False))
    await invoke(llm, method)
    assert llm.last_usage["status"] == "unavailable"
    assert llm.last_usage["prompt_tokens"] is None
    assert llm.last_usage["completion_tokens"] is None
    assert llm.total_input_tokens == 0
    assert llm.total_completion_tokens == 0


@pytest.mark.asyncio
async def test_completed_text_only_stream_without_usage_is_valid():
    llm, _ = llm_for(OfflineStream(chunk(text="Done"), chunk(finish="stop")))
    result = await invoke(llm, "ask_tool", stream=True)
    assert result.content == "Done"
    assert result.tool_calls is None
    assert llm.last_usage["status"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
async def test_partial_visible_stream_failure_is_not_retried(method):
    stream = OfflineStream(chunk(text="Already visible"), status_error(503))
    llm, transport = llm_for(stream, completion())
    events = []

    async def emit(event):
        events.append(event)

    kwargs = {"on_event": emit} if method == "ask_tool" else {}
    with pytest.raises(Exception, match="stream_interrupted"):
        await invoke(llm, method, stream=True, **kwargs)
    assert len(transport.requests) == 1
    assert stream.closed


@pytest.mark.asyncio
async def test_bedrock_tool_stream_is_explicitly_unsupported():
    llm, transport = llm_for(completion())
    llm.api_type = "aws"
    with pytest.raises(ValueError, match="stream.*[Bb]edrock|[Bb]edrock.*stream"):
        await invoke(llm, "ask_tool", stream=True)
    assert not transport.requests


@pytest.mark.asyncio
async def test_close_failure_after_visible_text_is_not_retried():
    class FailingClose(OfflineStream):
        async def close(self):
            raise status_error(503)

    stream = FailingClose(chunk(text="Visible"), chunk(finish="stop"))
    llm, transport = llm_for(stream, completion())

    async def emit(event):
        pass

    with pytest.raises(Exception, match="stream_interrupted"):
        await invoke(llm, "ask_tool", stream=True, on_event=emit)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_cancelled_stream_remains_cancelled_when_close_also_fails():
    class FailingClose(OfflineStream):
        async def close(self):
            raise status_error(503)

    llm, transport = llm_for(FailingClose(asyncio.CancelledError()), completion())
    with pytest.raises(asyncio.CancelledError):
        await invoke(llm, "ask_tool", stream=True)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_stream_without_finish_marker_is_not_a_success():
    llm, transport = llm_for(OfflineStream(chunk(text="Partial")))
    with pytest.raises(ValueError, match="finish"):
        await invoke(llm, "ask_tool", stream=True)
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_unknown_usage_still_reserves_input_budget():
    llm, transport = llm_for(completion(usage=False), completion())
    llm.max_input_tokens = llm.count_message_tokens([{"role": "user", "content": "hello"}])
    await invoke(llm, "ask")
    assert llm.last_usage["budget_status"] == "estimated"
    with pytest.raises(TokenLimitExceeded):
        await invoke(llm, "ask")
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_transient_retry_does_not_append_images_twice_or_mutate_input():
    llm, transport = llm_for(status_error(503), completion())
    messages = [{"role": "user", "content": "hello"}]
    call = LLM.ask_with_images.retry_with(wait=wait_none(), stop=stop_after_attempt(2))
    result = await call(llm, messages=messages, images=["https://offline.invalid/image.png"])
    assert result == "ready"
    assert messages == [{"role": "user", "content": "hello"}]
    assert transport.requests[0]["messages"] == transport.requests[1]["messages"]
    assert len(transport.requests[1]["messages"][0]["content"]) == 2


@pytest.mark.asyncio
async def test_stream_can_retry_before_any_text_is_visible():
    failed = OfflineStream(status_error(503))
    llm, transport = llm_for(failed, OfflineStream(chunk(text="Done"), chunk(finish="stop")))
    events = []

    async def emit(event):
        events.append(event)

    result = await invoke(llm, "ask_tool", stream=True, on_event=emit)
    assert result.content == "Done"
    assert events == [{"type": "model_text_delta", "text": "Done"}]
    assert len(transport.requests) == 2
    assert failed.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "content_filter"])
async def test_truncated_error_is_machine_readable(finish):
    llm, _ = llm_for(completion(arguments='{"query":"x"}', finish=finish))
    with pytest.raises(ValueError) as caught:
        await invoke(llm, "ask_tool")
    assert caught.value.code == "provider_truncated"
