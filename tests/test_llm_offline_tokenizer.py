from __future__ import annotations

from app import llm as llm_module


def test_unknown_model_uses_conservative_offline_tokenizer_when_cache_is_missing(
    monkeypatch,
):
    def unknown_model(model):
        raise KeyError(model)

    def unavailable_cache(name):
        raise OSError("offline fixture")

    monkeypatch.setattr(llm_module.tiktoken, "encoding_for_model", unknown_model)
    monkeypatch.setattr(llm_module.tiktoken, "get_encoding", unavailable_cache)

    tokenizer = llm_module.load_tokenizer("unmapped-model")

    assert tokenizer.encode("aé") == list("aé".encode("utf-8"))


def test_unknown_model_prefers_cached_cl100k_encoding(monkeypatch):
    sentinel = object()

    def unknown_model(model):
        raise KeyError(model)

    monkeypatch.setattr(llm_module.tiktoken, "encoding_for_model", unknown_model)
    monkeypatch.setattr(
        llm_module.tiktoken,
        "get_encoding",
        lambda name: sentinel if name == "cl100k_base" else None,
    )

    assert llm_module.load_tokenizer("unmapped-model") is sentinel
