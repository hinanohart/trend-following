"""
Golden vectors for the trend following core.

Independent of both implementations: standard library only, written from the
definitions rather than from the shortcuts the libraries use

    ewmac      = ewm(span, adjust = False) of the cumulative return index, flat before the start
    normaliser = explicit sum of squared crossover kernel weights over the valid observations
    signal     = explicit blend kernel over the root of its weighted sum of squared returns
    sizing     = .7 ewm_vol + .3 of its 10 year average
    tsmom      = explicit window sums

Floats are written with repr, so they round-trip exactly.

    $ python3 testdata/generate.py
"""

import math
import random
from itertools import accumulate, starmap
from pathlib import Path

T = 400
EWMA_SPAN = 10.
VOL_SPAN = 35.
FAST, SLOW = 16., 64.
LOOKBACK = 20
SPEEDS = ((8., 32.), (16., 64.), (32., 128.), (64., 256.))
TARGET_VOL, CAP, PERIODS, WARMUP = .15, 2., 252, 75
LONG_RUN = .3
COST = 2e-4
EPS = 1e-12

COLUMNS = ('r', 'ewma', 'ewm_vol', 'ewmac', 'tsmom', 'position', 'ret_lag1', 'ret_lag2_cost', 'signal')

# helpers

def beta(span):
    return (span - 1.) / (span + 1.)

def zeroed(xs):
    return list(map(lambda x: x if math.isfinite(x) else 0., xs))

def valid(xs):
    return list(map(lambda x: float(math.isfinite(x)), xs))

def valid_count(xs):
    return list(map(int, accumulate(valid(xs))))

def ewm_adjust_false(xs, span):
    # h_t = b h_{t-1} + (1 - b) x_t with h_{-1} = 0
    b = beta(span)
    return list(accumulate(xs, lambda h, x: b * h + (1. - b) * x, initial = 0.))[1:]

def weighted_mean(xs, mask, span):
    # pandas ewm(span, adjust = True): sum_i b^i x_{t-i} / sum_i b^i over valid observations
    b = beta(span)
    num = accumulate(xs, lambda h, x: b * h + x)
    den = accumulate(mask, lambda h, m: b * h + m)
    return list(map(lambda nd: nd[0] / max(nd[1], EPS), zip(num, den)))

# the pieces

def ewma(xs, span):
    return weighted_mean(zeroed(xs), valid(xs), span)

def ewm_vol(rs, span):
    squares = list(map(lambda r: r * r, zeroed(rs)))
    return list(map(lambda v: math.sqrt(max(v, EPS)), weighted_mean(squares, valid(rs), span)))

def kernel(fast, slow):
    # weight of r_{t-i} in the crossover:  b_s^(i+1) - b_f^(i+1)
    return list(map(lambda i: beta(slow) ** (i + 1) - beta(fast) ** (i + 1), range(T)))

def crossover(rs, fast, slow):
    index = list(accumulate(zeroed(rs)))
    return list(map(lambda f, s: f - s, ewm_adjust_false(index, fast), ewm_adjust_false(index, slow)))

def ewmac(rs, fast, slow):
    # squared kernel weights summed over the lags that were valid
    w, mask = kernel(fast, slow), valid(rs)
    var = lambda t: math.fsum(map(lambda i: w[i] ** 2 * mask[t - i], range(t + 1)))
    return list(map(lambda m, t: m / math.sqrt(max(var(t), EPS)), crossover(rs, fast, slow), range(T)))

def tsmom(rs, lookback):
    r0, mask = zeroed(rs), valid(rs)
    window = lambda xs, t: math.fsum(xs[max(0, t - lookback + 1):t + 1])
    return list(map(lambda t: window(r0, t) / math.sqrt(max(window(mask, t), 1.)), range(T)))

def tstat(rs):
    # blend kernel W_i = sum_k w_k,i / sd_k, sd_k over the valid lags, then sum_i W_i r_(t-i) / sqrt(sum_i W_i^2 r_(t-i)^2)
    r0, mask, kernels = zeroed(rs), valid(rs), list(map(lambda speed: kernel(*speed), SPEEDS))

    def at(t):
        lags = range(t + 1)
        sd = list(map(lambda w: math.sqrt(max(math.fsum(map(lambda i: w[i] ** 2 * mask[t - i], lags)), EPS)), kernels))
        blend = list(map(lambda i: math.fsum(map(lambda w, s: w[i] / s, kernels, sd)), lags))
        num = math.fsum(map(lambda i: blend[i] * r0[t - i], lags))
        den = math.fsum(map(lambda i: (blend[i] * r0[t - i]) ** 2, lags))
        return num / math.sqrt(max(den, EPS))

    return list(map(at, range(T)))

def sizing_vol(rs):
    vol = ewm_vol(rs, VOL_SPAN)
    valid = list(map(lambda r: float(math.isfinite(r)), rs))
    long_run = weighted_mean(list(map(lambda v, m: v * m, vol, valid)), valid, 10. * PERIODS)
    return list(map(lambda v, l: (1. - LONG_RUN) * v + LONG_RUN * l, vol, long_run))

def positions(rs, signal):
    size = lambda s, v, n: min(max(s, -CAP), CAP) * TARGET_VOL / (max(v, EPS) * math.sqrt(PERIODS)) if n > WARMUP else 0.
    return list(map(size, signal, sizing_vol(rs), valid_count(rs)))

def strategy_returns(pos, rs, cost, lag):
    held = ([0.] * lag + pos)[:T]
    previous = [0.] + held[:-1]
    return list(map(lambda w, p, r: w * r - cost * abs(w - p), held, previous, zeroed(rs)))

def sharpe(rets):
    mean = math.fsum(rets) / len(rets)
    var = math.fsum(map(lambda x: (x - mean) ** 2, rets)) / len(rets)
    return mean / math.sqrt(var + EPS) * math.sqrt(PERIODS)

# cases

def gaussian(seed, mu = 0., sigma = .01):
    rng = random.Random(seed)
    return list(map(lambda _: rng.gauss(mu, sigma), range(T)))

def with_gaps(rs):
    missing = set(range(50)) | set(range(100, T, 37))
    return list(map(lambda t, r: math.nan if t in missing else (math.inf if t == 222 else r), range(T), rs))

CASES = dict(
    iid = gaussian(0),
    trend = gaussian(1, mu = .002),
    zeros = [0.] * T,
    gaps = with_gaps(gaussian(2)),
)

def columns(rs):
    signal = tstat(rs)
    pos = positions(rs, signal)
    return (
        rs,
        ewma(rs, EWMA_SPAN),
        ewm_vol(rs, VOL_SPAN),
        ewmac(rs, FAST, SLOW),
        tsmom(rs, LOOKBACK),
        pos,
        strategy_returns(pos, rs, 0., 1),
        strategy_returns(pos, rs, COST, 2),
        signal,
    )

def write(path, name, rs):
    cols = columns(rs)
    header = f'# generated by testdata/generate.py, do not edit. case={name} T={T} ewma_span={EWMA_SPAN} vol_span={VOL_SPAN} ewmac={FAST}/{SLOW} tsmom={LOOKBACK} speeds={SPEEDS} target_vol={TARGET_VOL} cap={CAP} periods={PERIODS} warmup={WARMUP} long_run={LONG_RUN} cost={COST}'
    rows = map(lambda row: ','.join(map(repr, row)), zip(*cols))
    path.write_text('\n'.join((header, ','.join(COLUMNS), *rows)) + '\n')
    return name, sharpe(cols[6]), sharpe(cols[7])

if __name__ == '__main__':
    here = Path(__file__).parent
    summary = list(map(lambda item: write(here / f'{item[0]}.csv', *item), CASES.items()))
    lines = map(lambda row: ','.join((row[0], repr(row[1]), repr(row[2]))), summary)
    (here / 'sharpe.csv').write_text('\n'.join(('# generated by testdata/generate.py, do not edit', 'case,ret_lag1,ret_lag2_cost', *lines)) + '\n')
