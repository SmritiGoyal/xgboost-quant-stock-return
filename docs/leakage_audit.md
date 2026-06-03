# Leakage Audit: Idiosyncratic-Volatility Feature

This note documents a look-ahead bias found in the idiosyncratic-volatility
feature (`vol_12m`), how it was detected, its impact on the headline results,
and the one-line fix. It is included because catching and correcting a leak is
part of the research process, not something to hide.

## Summary

| | Sharpe | Monthly alpha | Alpha t-stat | Market beta |
|---|---:|---:|---:|---:|
| Before fix (leaked) | 2.53 | +4.36% | 13.24 | +0.73 |
| After fix (correct) | 1.03 | +2.19% | 6.08 | -0.43 |

The leak accounted for essentially the entire difference between a
suspiciously strong result and an honest one.

## The bug

The feature was computed as a trailing 12-month standard deviation of returns:

```python
merged["vol_12m"] = g["RET"].rolling(12).std().reset_index(level=0, drop=True)
```

`pandas` `rolling(12)` is right-aligned: the window at month *t* spans
`[t-11, t]` and therefore **includes month t's own return**. The model's
prediction target is the (cross-sectionally demeaned) return of month *t*. So
the feature was partly constructed from the very quantity it was being used to
predict — a textbook look-ahead leak.

Every other feature in the pipeline is correctly lagged to information
available at the start of month *t*:

- `reversal_1m` uses `RET.shift(1)`
- `ret_2_12` (momentum) spans `[t-12, t-2]`, skipping `t-1`
- `new_issue` applies an extra `.shift(1)` as a conservative buffer
- Compustat fundamentals are shifted forward 5 months for filing lag

`vol_12m` was the lone exception.

## How it was found

A feature ablation (`src/run_ablation.py`) was run to measure how much the two
self-proposed signals (`reversal_1m`, `vol_12m`) added over an 8-feature
academic baseline. The first ablation produced an implausible result:

```
baseline_no_proposed   Sharpe 0.94
full_model             Sharpe 2.53   (+169% from two features)
proposed_only          Sharpe 0.64
```

Two features nearly tripling the Sharpe of an 8-feature model is not credible.
That prompted a line-by-line review of how the two proposed features were
constructed, which surfaced the missing lag on `vol_12m`.

## The fix

Lag the feature by one month so its window ends at *t-1*:

```python
merged["vol_12m"] = g["RET"].rolling(12).std().reset_index(level=0, drop=True)
merged["vol_12m"] = merged.groupby("PERMNO")["vol_12m"].shift(1)
```

## Post-fix ablation

After the fix, the same ablation gives an honest, modest contribution from the
proposed signals:

| Feature set | Features | Ann. Sharpe | Monthly alpha | Alpha t-stat |
|---|---:|---:|---:|---:|
| Baseline (academic + controls) | 8 | 0.94 | +2.14% | 5.65 |
| Full model (+ proposed signals) | 10 | 1.03 | +2.19% | 6.08 |
| Proposed signals only | 2 | 0.38 | +0.92% | 3.05 |

The proposed signals add ~+0.09 Sharpe (about +10%) over the academic
baseline — complementary, not dominant. The baseline (which never contained
`vol_12m`) was unaffected by the leak, so its 0.94 was correct all along; the
leak lived entirely in the full and proposed-only configurations.

## Takeaway

The corrected long-short spread has an annualized Sharpe of 1.03 with a
strongly significant alpha (t = 6.08) over 300 out-of-sample months. It is a
frictionless gross result driven mainly by the short leg, with a negative
market beta — a respectable research finding, not a deployable strategy. The
headline figure is the leak-free one.
