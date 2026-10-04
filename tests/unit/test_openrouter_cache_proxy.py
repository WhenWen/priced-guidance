from tech_tree_arena.runtime.openrouter_cache_proxy import prepare_openrouter_body


def test_prepare_openrouter_body_enables_anthropic_cache_and_sticky_routing() -> None:
    request = {
        "model": "anthropic/claude-fable-5",
        "prompt_cache_key": "codex-thread-123",
        "input": [{"role": "user", "content": "hello"}],
    }

    prepared, diagnostics = prepare_openrouter_body(request, cache_ttl="1h")

    assert "cache_control" not in request
    assert prepared["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert prepared["session_id"].startswith("codex-")
    assert len(prepared["session_id"]) == 70
    assert diagnostics["prompt_cache_key_present"] is True


def test_prepare_openrouter_body_has_stable_fallback_session_id() -> None:
    first = {
        "model": "anthropic/claude-fable-5",
        "instructions": "stable",
        "input": [{"role": "user", "content": "first"}],
    }
    resumed = {
        **first,
        "input": [
            *first["input"],
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
    }

    prepared_first, _ = prepare_openrouter_body(first, cache_ttl="5m")
    prepared_resumed, _ = prepare_openrouter_body(resumed, cache_ttl="5m")

    assert prepared_first["session_id"] == prepared_resumed["session_id"]
    assert prepared_first["cache_control"] == {"type": "ephemeral"}


def test_prepare_openrouter_body_preserves_explicit_cache_configuration() -> None:
    request = {
        "model": "anthropic/claude-fable-5",
        "session_id": "caller-session",
        "cache_control": {"type": "ephemeral"},
    }

    prepared, _ = prepare_openrouter_body(request, cache_ttl="1h")

    assert prepared["session_id"] == "caller-session"
    assert prepared["cache_control"] == {"type": "ephemeral"}
