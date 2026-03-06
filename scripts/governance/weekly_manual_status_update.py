#!/usr/bin/env python3
"""Apply explicit manual signoff updates to weekly manual status artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.manual_status_utils import (
    MANUAL_KEYS,
    ROLE_KEYS,
    apply_manual_status_updates,
    build_manual_status_markdown,
    default_manual_status_template,
    normalize_manual_status,
    parse_bool_text,
)
from scripts.governance.report_utils import (
    load_optional_json_object as _load_optional_json,
    write_json as _write_json,
    write_markdown as _write_markdown,
)


def _parse_check_assignment(raw: str) -> tuple[str, bool]:
    key, sep, value = str(raw).partition("=")
    if sep != "=":
        raise argparse.ArgumentTypeError("manual check updates must use key=value")
    normalized_key = key.strip()
    if normalized_key not in MANUAL_KEYS:
        raise argparse.ArgumentTypeError(f"unsupported manual check key: {normalized_key}")
    try:
        return normalized_key, parse_bool_text(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_signoff_assignment(raw: str) -> tuple[str, str]:
    role, sep, signer = str(raw).partition("=")
    if sep != "=":
        raise argparse.ArgumentTypeError("signoff updates must use role=value")
    normalized_role = role.strip()
    if normalized_role not in ROLE_KEYS:
        raise argparse.ArgumentTypeError(f"unsupported signoff role: {normalized_role}")
    cleaned_signer = signer.strip()
    if not cleaned_signer:
        raise argparse.ArgumentTypeError("signoff value cannot be empty; use --clear-signoff")
    return normalized_role, cleaned_signer


def main() -> int:
    parser = argparse.ArgumentParser(description="Update weekly manual status with explicit checks.")
    parser.add_argument(
        "--manual-status-json",
        default="artifacts/weekly-manual-status.json",
        help="Path to manual status JSON (also used as default output path).",
    )
    parser.add_argument(
        "--decision-json",
        default="artifacts/weekly-decision-log.json",
        help="Path to weekly decision log JSON for markdown context.",
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
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        type=_parse_check_assignment,
        help="Manual check update in key=true|false form. Repeatable.",
    )
    parser.add_argument(
        "--signoff",
        action="append",
        default=[],
        type=_parse_signoff_assignment,
        help="Role signoff update in role=name form. Repeatable.",
    )
    parser.add_argument(
        "--clear-signoff",
        action="append",
        default=[],
        choices=ROLE_KEYS,
        help="Clear a role signoff. Repeatable.",
    )
    args = parser.parse_args()

    decision = _load_optional_json(Path(args.decision_json).resolve())
    manual_status_path = Path(args.manual_status_json).resolve()
    output_path = Path(args.output_json).resolve() if args.output_json else manual_status_path
    existing_status = _load_optional_json(manual_status_path)
    status = normalize_manual_status(existing_status) if existing_status else default_manual_status_template()

    signoff_updates = dict(args.signoff)
    for role in args.clear_signoff:
        signoff_updates[role] = ""

    updated_status, changed_keys = apply_manual_status_updates(
        status,
        manual_updates=dict(args.check),
        signoff_updates=signoff_updates,
    )

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
    print(f"Weekly manual status update: changed_keys={changed_text}, output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
