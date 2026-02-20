# Stress Test Report

Generated: 2026-02-08T08:17:37.756180

======================================================================


## Sustained Load - ✓ PASSED
----------------------------------------
- target_tps: 100
- actual_tps: 1306.2453
- iterations: 500
- duration_seconds: 0.3828
- errors: 0
- latency_mean_ms: 0.7557
- latency_p95_ms: 1.7032
- latency_p99_ms: 2.0244

## Flash Crash Scenario - ✓ PASSED
----------------------------------------
- iterations: 200
- latency_mean_ms: 0.8431
- latency_max_ms: 7.0120
- circuit_violations: 193