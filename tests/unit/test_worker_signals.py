import signal

from tech_tree_arena.runtime import worker


def test_worker_ignores_terminal_signals_owned_by_parent(monkeypatch) -> None:
    installed: dict[signal.Signals, object] = {}

    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    worker._ignore_parent_owned_signals()

    assert installed == {
        signal.SIGINT: signal.SIG_IGN,
        signal.SIGTERM: signal.SIG_IGN,
    }
