import signal

from tech_tree_arena import cli


def test_cli_defers_sigint_and_sigterm_to_safe_engine_boundary(monkeypatch) -> None:
    installed: dict[signal.Signals, object] = {}

    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.__setitem__(signum, handler),
    )

    cli._install_terminate_handler()

    assert installed == {
        signal.SIGINT: cli._terminate_as_interrupt,
        signal.SIGTERM: cli._terminate_as_interrupt,
    }
    assert cli._terminate_delivered is False
    assert cli._terminate_requested is False
    assert cli._runner_active is False
