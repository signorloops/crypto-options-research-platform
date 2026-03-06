#!/usr/bin/env python3
"""Auto-prefill objective weekly manual-status fields from generated artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.manual_status_utils import (
    apply_manual_status_updates,
    build_manual_status_markdown,
    default_manual_status_template,
    normalize_manual_status,
)
from scripts.governance.report_utils import (
    load_optional_json_object as _load_optional_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_autofill(decision: dict[str, Any], attribution: dict[str, Any]) -> dict[str, bool]:
    decision_text = str(decision.get("decision", "")).strip()
    rollback_raw = decision.get("rollback")
    rollback_map = rollback_raw if isinstance(rollback_raw, dict) else {}
    rollback_ref = str(rollback_map.get("reference", "")).strip()

    summary_raw = attribution.get("summary")
    summary_map = summary_raw if isinstance(summary_raw, dict) else {}
    strategies = _as_int(summary_map.get("strategies")) or 0
    missing_entries = _as_int(summary_map.get("missing_entries"))
    attribution_complete = strategies > 0 and missing_entries == 0

    return {
        "rollback_decision_recorded": bool(decision_text),
        "change_and_rollback_recorded": bool(decision_text and rollback_ref),
        "pnl_attribution_confirmed": attribution_complete,
    }


def _apply_prefill(
    status: dict[str, Any], prefill: dict[str, bool]
) -> tuple[dict[str, Any], list[str]]:
    return apply_manual_status_updates(
        status,
        manual_updates={key: True for key, flag in prefill.items() if flag},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-prefill objective weekly manual-status fields."
    )
    parser.add_argument(
        "--decision-json",
        default="artifacts/weekly-decision-log.json",
        help="Path to weekly decision log JSON.",
    )
    parser.add_argument(
        "--attribution-json",
        default="artifacts/weekly-pnl-attribution.json",
        help="Path to weekly PnL attribution JSON.",
    )
    parser.add_argument(
        "--manual-status-json",
        default="artifacts/weekly-manual-status.json",
        help="Path to manual status JSON (also used as default output path).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path. Defaults to --manual-status-json.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown checklist path. Disabled when empty.",
    )
    args = parser.parse_args()

    decision = _load_optional_json(Path(args.decision_json).resolve())
    attribution = _load_optional_json(Path(args.attribution_json).resolve())

    manual_status_path = Path(args.manual_status_json).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else manual_status_path

    existing_status = _load_optional_json(manual_status_path)
    status = normalize_manual_status(existing_status) if existing_status else default_manual_status_template()
    prefill = _derive_autofill(decision, attribution)
    updated_status, changed_keys = _apply_prefill(status, prefill)

    _write_json(output_path, updated_status)
    if args.output_md:
        _write_markdown(
            Path(args.output_md).resolve(),
            build_manual_status_markdown(
                status=updated_status,
                decision=decision,
                changed_keys=changed_keys,
            ),
        )

    changed_text = ", ".join(changed_keys) if changed_keys else "none"
    print("Weekly manual status prefill: " f"updated_keys={changed_text}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
