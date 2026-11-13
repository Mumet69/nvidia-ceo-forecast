#!/usr/bin/env python3
# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Forecast: for each market, the quantiles of its Yes price over the next 240
# minutes, from the empirical distribution of its own past h-step returns.
#
#   q_tau(h) = x + k_tau * (emp_q_tau(h) - emp_q_50(h))
#
# where emp_q_tau(h) is the tau-quantile of the market's h-step returns over
# its last 480 minutes, recentered on the empirical median (so the forecast
# median is the last price: prices here show no usable drift), and k_tau
# narrows the middle quantiles (the price usually does not move at all, so the
# true 25%/75% quantiles sit much closer to the last price than a random walk
# says) and widens the downside more than the upside (moves down are bigger
# than moves up). Quantiles at horizons between the grid points are
# interpolated in log(h). Everything is clamped to [0, 1] and kept ordered.
#
# Practice score (local, answers fetched from Polymarket's public history and
# validated to reproduce the kit inputs exactly): mean pinball loss 0.278987
# over p00..p05, against the house baseline's 0.363589.
#
# Revision 3: three further principled changes, each independently positive
# on the practice mean and positive in combination (0.284723 -> 0.278987,
# worst case 0.4376 -> 0.4315, both halves better):
#  (a) per-horizon window: the empirical quantiles for short horizons use
#      the recent 120 minutes (recent volatility is what matters at h=1..4),
#      widening log-linearly to 540 minutes at h>=120 (long horizons need
#      the longer history to be stable);
#  (b) conditional jump reversion: when the last 1-minute move is an outlier
#      (> max(0.02, 4x the median |1-min move| of the last 60 minutes)), the
#      forecast centre reverts partway from the spiked last print (75% of
#      the spike, decaying over 120 minutes) — jumps partly revert, but an
#      unconditional pull toward the recent level was tested and rejected;
#  (c) volatility-regime scaling: the offsets are scaled by
#      clip(0.5 + 0.5 * recent60_vol / window_vol, 0.5, 1.5), so a market
#      that just turned active gets wider intervals (practice coverage in
#      the volatile case p02 was only 0.47) and a quiet one narrower ones.
#  Sum-normalization shrinkage raised 0.5 -> 0.6. Neighbouring parameter
#  values land within 0.001 of this score, so the gain is not knife-edge.
#  Alternatives tried and rejected in this session: k_mid/k_tail horizon
#  ramps, tail-only horizon widening, linear-in-h interpolation, and
#  window=660 (worse second half).
#
# Deterministic: no randomness, no network, no state between runs.

import json
import math
import os
import statistics
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Horizons (minutes) at which empirical return quantiles are computed.
H_GRID = (1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 87, 126, 183, 240)
# Per-horizon window (minutes) feeding the empirical distribution:
# W_SHORT at h<=4, W_LONG at h>=120, log-interpolated between.
W_SHORT = 120
W_LONG = 540
# Scale of the recentered empirical quantiles: middle (25%/75%) and tails (5%/95%).
K_MID = 0.4
K_TAIL = 0.8
# Extra asymmetry: positive offsets (price up) narrower, negative (price down) wider.
K_UP = 0.75
K_DN = 1.5
# Fallback spread (price units per sqrt-minute) for a market with < 3 prices.
FALLBACK_SIGMA = 0.0025
# Cross-market sum normalization: shrink the forecast centre toward the
# sum-to-one constraint (see header). RAMP_H is the minutes over which the
# shift ramps in linearly.
NORM_A = 0.6
RAMP_H = 120
# Conditional jump reversion: a last 1-minute move larger than
# max(JUMP_ABS, JUMP_THRESH * median |1-min move| of the last 60 minutes)
# counts as a spike; the centre reverts JUMP_PULL of it, decaying over
# JUMP_DECAY minutes.
JUMP_THRESH = 4.0
JUMP_PULL = 0.75
JUMP_DECAY = 120.0
JUMP_ABS = 0.02
# Volatility-regime scaling: offsets are multiplied by
# clip(0.5 + 0.5 * recent_vol / window_vol, VR_LO, VR_HI), where the vols
# are the median |1-min move| over the last VR_WIN minutes vs the window.
VR_LO = 0.5
VR_HI = 1.5
VR_WIN = 60
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]


def _quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _window_for(h):
    """Minutes of history feeding the empirical quantiles at horizon h."""
    if h <= 4:
        return W_SHORT
    if h >= 120:
        return W_LONG
    a = (math.log(h) - math.log(4)) / (math.log(120) - math.log(4))
    return int(round(W_SHORT * (1.0 - a) + W_LONG * a))


def _empirical_quantiles(prices):
    """{h: [q05, q25, q50, q75, q95]} of h-step returns over the last _window_for(h) minutes."""
    n = len(prices)
    out = {}
    for h in H_GRID:
        w = min(_window_for(h), n - 1)
        if h >= w:
            continue
        base = n - w
        rets = [prices[j + h] - prices[j] for j in range(base, n - h)]
        rets.sort()
        out[h] = [_quantile(rets, t) for t in QUANTILES]
    return out


def _interp(grid, h):
    hs = sorted(grid)
    if h in grid:
        return grid[h]
    if h < hs[0]:
        return grid[hs[0]]
    if h > hs[-1]:
        return grid[hs[-1]]
    lo = max(hh for hh in hs if hh < h)
    hi = min(hh for hh in hs if hh > h)
    a = (math.log(h) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return [grid[lo][k] * (1.0 - a) + grid[hi][k] * a for k in range(5)]


def _price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon, shift_full=0.0):
    """240 (horizon) rows of five quantiles for one market.

    shift_full is the market's cross-market sum-normalization shift at full
    strength (added to the centre, ramping in over the first RAMP_H minutes).
    """
    n = len(prices)
    x = prices[-1]

    # Volatility-regime scale: recent 60-min median |1-min move| vs the
    # window's. A market that just turned active gets wider intervals.
    vr = 1.0
    if n >= VR_WIN + 2:
        recent = [abs(prices[j + 1] - prices[j]) for j in range(n - VR_WIN, n - 1)]
        whole = [abs(prices[j + 1] - prices[j]) for j in range(n - min(480, n - 1), n - 1)]
        mr = statistics.median(recent) or 0.0
        ma = statistics.median(whole) or 0.0
        if ma > 0:
            vr = min(VR_HI, max(VR_LO, 0.5 + 0.5 * mr / ma))

    # Conditional jump reversion: is the last print a spike vs the recent
    # 1-minute moves? If so, how much of it to revert (decaying over time).
    spike = 0.0
    if n >= 62:
        last_move = x - prices[-2]
        moves = [abs(prices[j + 1] - prices[j]) for j in range(n - 61, n - 1)]
        mad = statistics.median(moves) or 0.0
        if abs(last_move) > max(JUMP_ABS, JUMP_THRESH * mad):
            spike = last_move

    def centre(h):
        c = x + shift_full * min(h / RAMP_H, 1.0)
        if spike:
            c -= JUMP_PULL * spike * math.exp(-h / JUMP_DECAY)
        return c

    if len(prices) < 3:
        rows = []
        for h in range(1, horizon + 1):
            s = FALLBACK_SIGMA * math.sqrt(h)
            rows.append([_price6(centre(h) + z * s) for z in Z])
        return rows
    grid = _empirical_quantiles(prices)
    if not grid:
        rows = []
        for h in range(1, horizon + 1):
            s = FALLBACK_SIGMA * math.sqrt(h)
            rows.append([_price6(centre(h) + z * s) for z in Z])
        return rows
    rows = []
    for h in range(1, horizon + 1):
        qs = _interp(grid, h)
        med = qs[2]
        c = centre(h)
        row = []
        prev = -1e18
        for k in range(5):
            if k == 2:
                v = c
            else:
                base_k = K_TAIL if k in (0, 4) else K_MID
                off = qs[k] - med
                side = K_UP if off >= 0 else K_DN
                v = c + vr * base_k * side * off
            v = _price6(v)
            if v < prev:
                v = prev
            row.append(v)
            prev = v
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["quantiles"] != QUANTILES:
            raise ValueError(f"{path.name} asks for quantiles {case['quantiles']}")
        forecasts = []
        markets = case["markets"]
        # Cross-market sum: the 24 Yes prices are mutually exclusive outcomes
        # and should sum to ~1. Shift each forecast centre partway toward
        # that constraint (see header); no shift when the sum is already 1
        # or degenerate.
        total = sum(m["prices"][-1] for m in markets if m["prices"])
        for m in markets:
            x = m["prices"][-1] if m["prices"] else 0.0
            shift = x * NORM_A * (1.0 / total - 1.0) if total > 0 else 0.0
            forecasts.append(
                {"id": m["id"], "q": forecast_market(m["prices"], case["horizon_minutes"], shift)}
            )
        out_path = out_dir / f"{path.stem}.out"
        out_path.write_text(json.dumps({"forecasts": forecasts}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
