# Inverse Options PnL Mathematical Validation Report

Generated: 2026-02-08 16:33:35.499740

============================================================


## PnL Formula - ✓ PASSED
----------------------------------------
- trials: 1000
- zero_crossing_error: 0.0000000000
- limit_error: 0.0000500000
- pnl_up: 0.0000018182
- pnl_down: -0.0000022222
- asymmetry_ratio: 1.2222222222

## Non-linear Characteristics - ✓ PASSED
----------------------------------------
- asymmetry_ratio: 1.2222222222
- is_concave: True
- limit_error: 0.0000500000

## Extreme Scenarios - ✓ PASSED
----------------------------------------

### Scenarios:

**flash_crash:**
  - pnl: -0.009980
  - sign_correct: True
  - pnl_pct: -499.000000

**flash_pump:**
  - pnl: 0.000018
  - sign_correct: True
  - pnl_pct: 0.900000

**moderate_decline:**
  - pnl: -0.000009
  - sign_correct: True
  - pnl_pct: -0.428571

**moderate_rise:**
  - pnl: 0.000005
  - sign_correct: True
  - pnl_pct: 0.230769

## Delta Correction - ✓ PASSED
----------------------------------------
- analytical_delta: 0.0000000002
- numerical_delta: 0.0000000002
- absolute_error: 0.0000000000
- relative_error: 0.0000000331

## Inverse vs Linear Comparison - ✓ PASSED
----------------------------------------
- max_profit_inverse: 0.5000000000
- max_loss_inverse: -1.0000000000
- max_profit_linear: 0.5000000000
- max_loss_linear: -1.0000000000
- profit_asymmetry: 2.0000000000