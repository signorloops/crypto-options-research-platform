# validation_scripts archive

This directory stores legacy one-off validation scripts that are not currently
referenced by active workflows, Make targets, tests, or documentation.

Archived in 2026-03 simplification batch:
- deep_math_verification.py
- gamma_deep_verification.py
- gamma_derivation_analysis.py
- gamma_sign_analysis.py
- greeks_conversion_analysis.py
- inverse_options_math_validation.py
- inverse_pnl_math_validation.py
- iv_convergence_analysis.py
- long_term_algorithms_benchmark.py
- put_call_parity_verification.py
- theta_derivation_verification.py
- var_vol_shock_analysis.py

If any archived script needs to be reactivated, move it back to
`validation_scripts/` and add explicit wiring in docs/workflows.
