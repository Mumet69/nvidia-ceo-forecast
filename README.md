# Forecast Polymarket's next NVIDIA CEO prices — make.fun entry

Entry for the [make.fun](https://make.fun) bounty
`polymarket-nvidia-ceo-forecast-01`: forecast the Yes price of every trading
market of Polymarket's event "Next CEO of NVIDIA?" at every minute of the 4
hours after each checkpoint, as five quantiles (5%, 25%, 50%, 75%, 95%).

## Program

`main.py` — Python 3.13, standard library only, no network, deterministic.
It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Model

For each market, the quantiles of its price over the next 240 minutes, from
the empirical distribution of its own past h-step returns:

    q_tau(h) = x + k_tau * (emp_q_tau(h) - emp_q_50(h))

- `emp_q_tau(h)` is the tau-quantile of the market's h-step returns over its
  last 480 minutes, at horizons h in {1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60,
  87, 126, 183, 240}, interpolated in log(h) between grid points.
- Recentered on the empirical median, so the forecast median is the last
  price: these prices show no usable drift (adding drift, in either
  direction, hurts).
- The middle quantiles are narrowed (k = 0.4): the price usually does not
  move at all, so the true 25%/75% quantiles sit much closer to the last
  price than a random walk says. The tails use k = 0.8.
- Asymmetric: positive offsets (price up) are scaled by 0.75, negative
  offsets (price down) by 1.5 — moves down are bigger than moves up.
- Everything is clamped to [0, 1] and kept non-decreasing. A market with
  fewer than 3 prices falls back to the house persistence baseline.

## Local practice score

Scored locally on the kit's six practice cases (p00–p05) against answers
fetched from Polymarket's public price history
(`GET https://clob.polymarket.com/prices-history?...&fidelity=1`), gridded
exactly as the house grids them — the fetched grid reproduces all 214,513
input prices with zero error, and the same scorer reproduces the house
baseline's documented practice loss of 0.363589 exactly.

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.4695    | 0.5361         |
| p01  | 0.3053    | 0.3804         |
| p02  | 0.4062    | 0.4084         |
| p03  | 0.3143    | 0.3686         |
| p04  | 0.2006    | 0.2988         |
| p05  | 0.1074    | 0.2382         |
| **mean** | **0.2895** | **0.3636** |

Mean pinball loss **0.2895** vs the house baseline's **0.363589**
(-20%). Runtime ~0.12 s per case (limit 8 s).

## Run

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
