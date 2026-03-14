"""Compatibility shim — preserves existing import paths.

All logic has moved to execution.dashboard.*
"""

from __future__ import annotations

import os
import sys

from execution.dashboard import create_dashboard_app
from execution.dashboard.data_helpers import available_result_files
from execution.dashboard.pages.deviation import (
    build_cross_market_deviation_report,
)
from data.quote_integration import (
    build_cex_defi_deviation_dataset_live,
)

__all__ = [
    "create_dashboard_app",
    "available_result_files",
    "build_cross_market_deviation_report",
    "build_cex_defi_deviation_dataset_live",
]


# Propagate monkeypatch: when tests set an attribute on this shim module
# (e.g. build_cex_defi_deviation_dataset_live), forward it to the actual
# module that uses it so the patch takes effect.
_PATCH_FORWARD = {
    "build_cex_defi_deviation_dataset_live": "execution.dashboard.pages.deviation",
}

_this = sys.modules[__name__]
_original_setattr = type(_this).__setattr__


class _ShimModule(type(_this)):
    """Module subclass that propagates monkeypatch setattr to actual modules."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        target = _PATCH_FORWARD.get(name)
        if target and target in sys.modules:
            setattr(sys.modules[target], name, value)


_this.__class__ = _ShimModule


app = create_dashboard_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "execution.research_dashboard:app",
        host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
        port=int(os.getenv("DASHBOARD_PORT", "8501")),
        reload=False,
    )
