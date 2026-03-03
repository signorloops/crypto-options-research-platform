"""
Quick benchmark for crypto option pricing model zoo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is importable when running as a standalone script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research.pricing.model_zoo import CryptoOptionModelZoo, OptionQuote


def _build_synthetic_quotes(
    spot: float,
    rate: float,
    sigma: float,
    seed: int,
    n_per_bucket: int,
) -> list[OptionQuote]:
    rng = np.random.default_rng(seed)
    zoo = CryptoOptionModelZoo()
    maturities = [7.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0, 120.0 / 365.0]
    strikes = [0.8 * spot, 0.9 * spot, 1.0 * spot, 1.1 * spot, 1.2 * spot]

    quotes: list[OptionQuote] = []
    for t in maturities:
        for k in strikes:
            for _ in range(max(n_per_bucket, 1)):
                # Use Bates approximation as "market" generator to include both SV and jumps.
                base = zoo.price_option(
                    model="bates",
                    spot=spot,
                    strike=k,
                    maturity=t,
                    rate=rate,
                    sigma=sigma,
                    is_call=True,
                    model_params={
                        "kappa": 1.8,
                        "theta": 0.32,
                        "v0": 0.45,
                        "rho": -0.55,
                        "jump_intensity": 1.2,
                        "jump_mean": -0.06,
                        "jump_std": 0.28,
                    },
                )
                noisy = max(0.0, base * (1.0 + rng.normal(0.0, 0.01)))
                quotes.append(
                    OptionQuote(
                        spot=spot,
                        strike=float(k),
                        maturity=float(t),
                        rate=rate,
                        market_price=float(noisy),
                        is_call=True,
                    )
                )
    return quotes


def save_quotes_json(path: str, quotes: list[OptionQuote]) -> None:
    payload = [
        {
            "spot": float(q.spot),
            "strike": float(q.strike),
            "maturity": float(q.maturity),
            "rate": float(q.rate),
            "market_price": float(q.market_price),
            "is_call": bool(q.is_call),
        }
        for q in quotes
    ]
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_quotes_json(path: str) -> list[OptionQuote]:
    raw = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("quotes json must be a list")

    quotes: list[OptionQuote] = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("quote row must be an object")
        quotes.append(
            OptionQuote(
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                maturity=float(row["maturity"]),
                rate=float(row["rate"]),
                market_price=float(row["market_price"]),
                is_call=bool(row.get("is_call", True)),
            )
        )
    return quotes


def run_benchmark(
    seed: int = 42,
    n_per_bucket: int = 1,
    quotes: list[OptionQuote] | None = None,
) -> pd.DataFrame:
    """Run model zoo benchmark on synthetic or preloaded option quotes."""
    spot = 50000.0
    rate = 0.02
    sigma = 0.60
    quote_rows = (
        quotes
        if quotes is not None
        else _build_synthetic_quotes(
            spot=spot,
            rate=rate,
            sigma=sigma,
            seed=seed,
            n_per_bucket=n_per_bucket,
        )
    )

    zoo = CryptoOptionModelZoo()
    table = zoo.benchmark(
        quotes=quote_rows,
        sigma=sigma,
        model_params_by_model={
            "merton_jump_diffusion": {
                "jump_intensity": 1.0,
                "jump_mean": -0.05,
                "jump_std": 0.25,
            },
            "kou_jump": {
                "jump_intensity": 1.0,
                "p_up": 0.35,
                "eta1": 12.0,
                "eta2": 8.0,
            },
            "heston": {
                "kappa": 1.8,
                "theta": 0.32,
                "v0": 0.45,
                "rho": -0.55,
            },
            "bates": {
                "kappa": 1.8,
                "theta": 0.32,
                "v0": 0.45,
                "rho": -0.55,
                "jump_intensity": 1.2,
                "jump_mean": -0.06,
                "jump_std": 0.28,
            },
        },
    )
    if table.empty:
        return table
    return table.sort_values(["rmse", "mae", "model"], ascending=[True, True, True]).reset_index(
        drop=True
    )


def evaluate_benchmark_quality_gates(
    table: pd.DataFrame,
    expected_best_model: str = "bates",
    max_best_rmse: float = 120.0,
) -> list[str]:
    violations: list[str] = []
    if table.empty:
        return ["Benchmark table is empty"]

    ordered = table.sort_values(["rmse", "mae", "model"], ascending=[True, True, True])
    best = ordered.iloc[0]
    best_model = str(best.get("model", "")).strip()
    best_rmse = float(best.get("rmse", 0.0))

    if expected_best_model and best_model.lower() != expected_best_model.lower():
        violations.append(
            f"Unexpected best model: expected={expected_best_model}, actual={best_model}"
        )
    if best_rmse > float(max_best_rmse):
        violations.append(f"Best model RMSE too high: {best_rmse:.6f} > {float(max_best_rmse):.6f}")
    return violations


def save_benchmark_json(
    path: str,
    source: str,
    quotes: list[OptionQuote],
    table: pd.DataFrame,
) -> None:
    ordered = table.sort_values(["rmse", "mae", "model"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    rows: list[dict[str, Any]] = []
    for row in ordered.to_dict(orient="records"):
        rows.append(
            {
                "model": str(row.get("model", "")),
                "rmse": float(row.get("rmse", 0.0)),
                "mae": float(row.get("mae", 0.0)),
                "mean_abs_iv_error": float(row.get("mean_abs_iv_error", 0.0)),
                "n_quotes": int(float(row.get("n_quotes", 0))),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quotes_source": source,
        "n_quotes": int(len(quotes)),
        "results": rows,
    }
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_benchmark_markdown(
    source: str,
    quotes: list[OptionQuote],
    table: pd.DataFrame,
    violations: list[str],
) -> str:
    ordered = table.sort_values(["rmse", "mae", "model"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    lines: list[str] = [
        "# Pricing Model Zoo Benchmark",
        "",
        f"- Quotes source: `{source}`",
        f"- Quotes count: `{len(quotes)}`",
        "",
        "## Ranking",
        "",
        "| Rank | Model | RMSE | MAE | Mean Abs IV Error | N Quotes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(ordered.to_dict(orient="records"), start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('model', '')} | {float(row.get('rmse', 0.0)):.6f} | "
            f"{float(row.get('mae', 0.0)):.6f} | {float(row.get('mean_abs_iv_error', 0.0)):.6f} | "
            f"{int(float(row.get('n_quotes', 0)))} |"
        )

    lines.extend(["", "## Quality Gates", ""])
    if violations:
        lines.append("- FAIL")
        for violation in violations:
            lines.append(f"- {violation}")
    else:
        lines.append("- PASS")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run crypto option pricing model-zoo benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data.")
    parser.add_argument(
        "--n-per-bucket",
        type=int,
        default=1,
        help="Number of quotes per strike/maturity bucket.",
    )
    parser.add_argument(
        "--quotes-json",
        type=str,
        default="",
        help="Optional input quotes JSON path. When omitted, synthetic quotes are used.",
    )
    parser.add_argument(
        "--save-quotes-json",
        type=str,
        default="",
        help="Optional output path to persist the quotes used by benchmark.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional benchmark JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Optional benchmark markdown output path.",
    )
    parser.add_argument(
        "--expected-best-model",
        type=str,
        default="bates",
        help="Expected best-ranked model name for quality gate.",
    )
    parser.add_argument(
        "--max-best-rmse",
        type=float,
        default=120.0,
        help="Max allowed RMSE for best-ranked model.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when quality gates fail.",
    )
    args = parser.parse_args()

    if args.quotes_json:
        quotes = load_quotes_json(args.quotes_json)
        source = f"json:{args.quotes_json}"
    else:
        quotes = _build_synthetic_quotes(
            spot=50000.0,
            rate=0.02,
            sigma=0.60,
            seed=args.seed,
            n_per_bucket=args.n_per_bucket,
        )
        source = f"synthetic:seed={args.seed},n_per_bucket={args.n_per_bucket}"

    table = run_benchmark(quotes=quotes)
    violations = evaluate_benchmark_quality_gates(
        table=table,
        expected_best_model=args.expected_best_model,
        max_best_rmse=float(args.max_best_rmse),
    )

    if args.save_quotes_json:
        save_quotes_json(args.save_quotes_json, quotes)
    if args.output_json:
        save_benchmark_json(path=args.output_json, source=source, quotes=quotes, table=table)
    if args.output_md:
        markdown = render_benchmark_markdown(
            source=source,
            quotes=quotes,
            table=table,
            violations=violations,
        )
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

    print(table.to_string(index=False))
    if violations:
        for violation in violations:
            print(f"QUALITY GATE FAILED: {violation}")
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
