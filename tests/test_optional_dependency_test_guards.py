"""Guard optional-dependency tests from hard import failures in minimal CI."""

from __future__ import annotations

from pathlib import Path


def test_ppo_agent_test_guards_torch_before_importing_strategy_module():
    path = Path(__file__).resolve().parents[1] / "tests" / "test_ppo_agent.py"
    text = path.read_text(encoding="utf-8")

    assert 'pytest.importorskip("torch")' in text
    assert text.index('pytest.importorskip("torch")') < text.index(
        "from strategies.market_making.ppo_agent import"
    )
