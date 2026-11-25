"""
minimal core of systematic trend following

every rule here is a causal linear filter of past returns. the reference wiring reads its blended filter as a t-statistic -
the filter over the square root of its own weighted sum of squared returns - so no volatility estimate enters the signal
breakouts (where price sits in a window's range) are not linear filters and are not covered
returns are simple excess returns per period with time last, any leading dimensions (assets, batch) are carried along
float32 or float64 only - half precision quantises the decays near 1, so every exponentially weighted piece refuses it
for back-adjusted futures a return is the adjusted price change over the unadjusted price of the contract held
non-finite returns (nan, inf) are missing - they add nothing and only let time pass. an exact zero is an observation (a flat
print): mark known stale prints nan yourself - treating every zero as missing inflates the volatility of tick-sized markets
time passes exactly for about 240 missing rows of the signal and 534 of the volatility (span 35); past that the 1e-12
guard against dividing by zero acts, so split series at longer gaps

nothing here is tuned to make money. spans, caps and targets chosen on the same data
they are evaluated on will overfit - no setting here is a strategy. sources and measurements: EVIDENCE.md
"""

from collections import namedtuple
from functools import partial, reduce as fold
from itertools import product

import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce

# constants

EPS = 1e-12

Intermediates = namedtuple('Intermediates', ('vol', 'signal'))

# helpers

# spans as in pandas - a centre of mass c is span 2c + 1, a halflife h is beta = .5 ** (1 / h)

def beta_from_span(span):
    return (span - 1.) / (span + 1.)

def spans_to_betas(spans, like):
    spans = torch.as_tensor(spans, dtype = like.dtype, device = like.device)
    return rearrange(beta_from_span(spans), '... -> (...) 1')

def shift(t, k = 1):
    return F.pad(t, (k, 0))[..., :t.shape[-1]]

def valid_count(x):
    return x.isfinite().cumsum(dim = -1).to(x.dtype)

# h_t = beta * h_{t-1} + b_t along time, by recursive doubling
# log2(t) steps, exact, causal, differentiable in both b and beta. folded, so one step at a time is held in memory

def double(state, k):
    h, beta = state
    return h + beta * shift(h, k), beta * beta

def doubling_scan(b, beta):
    strides = map(partial(pow, 2), range((b.shape[-1] - 1).bit_length()))
    return fold(double, strides, (b, beta))[0]

# the same scan in blocks of `block` steps - doubling inside every block at once, then across the block ends, then every block
# adds its carry. as many steps as before (log2(block) + log2(t / block)), but t log2(block) work instead of t log2(t)

def scan(b, beta, block = 16):
    assert b.dtype in (torch.float32, torch.float64), 'float32 or float64 only - half precision quantises decays near 1'
    beta = rearrange(torch.as_tensor(beta, dtype = b.dtype, device = b.device), '... -> ... 1')
    decay = beta ** torch.arange(1, block + 1, dtype = b.dtype, device = b.device)
    local = doubling_scan(rearrange(F.pad(b, (0, -b.shape[-1] % block)), '... (n l) -> ... n l', l = block), beta)
    carry = shift(doubling_scan(local[..., -1], decay[..., -1]))
    return rearrange(local + carry[..., None] * decay, '... n l -> ... (n l)')[..., :b.shape[-1]]

# sum_i b^(i + 1) x_{t-i}, the decayed sums every crossover, its variance and its t-statistic are built from

def decayed(x, beta):
    return beta * scan(x, beta)

# spans as one flat python list - equal spans give equal rows, so each distinct one is scanned once
# found in python, not by unique, so every shape is static

def flat(spans):
    spans = spans.tolist() if hasattr(spans, 'tolist') else spans
    return sum(map(flat, spans), []) if isinstance(spans, (tuple, list)) else [spans]

def masked_ewma(x, valid, span):
    beta = beta_from_span(span)
    return scan(x, beta) / scan(valid.to(x.dtype), beta).clamp(min = EPS)

# exponentially weighted mean over valid observations - pandas ewm(span, adjust = True)

def ewma(x, span):
    valid = x.isfinite()
    return masked_ewma(x.where(valid, 0.), valid, span)

# ex-ante volatility - zero mean exponentially weighted rms of returns
# one bad print still counts - set bad prints nan first

def ewm_vol(r, span = 35.):
    valid = r.isfinite()
    return masked_ewma(r.where(valid, 0.).square(), valid, span).clamp(min = EPS).sqrt()

# moving average crossover of the cumulative return index, written on returns directly
#   slow_beta * S_slow(r) - fast_beta * S_fast(r),   S_b(x)_t = sum_i b^i x_{t-i}
# divided by its standard deviation per unit return volatility over the valid observations seen
#   sum_i valid_{t-i} (slow_beta^(i + 1) - fast_beta^(i + 1))^2 - the same decayed sums, run on the validity mask
# -> unit variance on iid returns from the first step and through any gaps, in return units (divide by volatility for a z-score)
# the kernel's mean lag is (fast + slow) / 2 - 1 periods, tsmom's (lookback - 1) / 2
# spans may be sequences, giving a speeds dimension (..., k, t)

def ewmac(r, fast = 16., slow = 64.):
    single = all(map(lambda span: torch.as_tensor(span).ndim == 0, (fast, slow)))
    fast, slow = map(partial(spans_to_betas, like = r), (fast, slow))

    valid = r.isfinite()
    r, valid = map(lambda t: rearrange(t, '... t -> ... 1 t'), (r.where(valid, 0.), valid.to(r.dtype)))

    crossover = decayed(r, slow) - decayed(r, fast)
    var = decayed(valid, slow * slow) + decayed(valid, fast * fast) - 2. * decayed(valid, slow * fast)

    out = crossover / var.clamp(min = EPS).sqrt()

    return rearrange(out, '... 1 t -> ... t') if single else out

# t-statistic of an equal risk blend of crossovers - one kernel W_i = sum_k (slow_k^(i + 1) - fast_k^(i + 1)) / sd_k, with sd_k
# the iid standard deviation of speed k as in ewmac, over sqrt(sum_i W_i^2 r_{t-i}^2) (heteroskedasticity consistent)
# -> unit variance whenever the signs of returns are symmetric given their sizes: any volatility path, any gaps, no volatility
# estimate. one return alone gives +-1, so a lone outlier moves it by about one and a large value needs a persistent trend
# sign dependent volatility (the leverage effect of equities) bends it, as it bends every normalisation, and serial correlation
# rho scales its sd by about sqrt((1 + rho) / (1 - rho)). with a box kernel it is the rolling t-statistic of returns
# scalar spans give one crossover's t-statistic, sequences the blend's - the variance adds its d (d + 1) / 2 decayed sums one at
# a time (d distinct decays): the default TrendFollowing peaks at about 0.2 kB per float32 input element on large batches (0.14
# on CUDA), up to 0.8 kB with backward on CPU - most of it memory glibc keeps, the tensors alive need about 0.18 kB - so split
# long batches by rows (rows are independent). on a GPU a lone series pays for the d (d + 1) / 2 sums in launches - batch the
# series

def ewmac_tstat(r, fast = 16., slow = 64.):
    spans = (slow, fast)
    fast, slow = map(partial(spans_to_betas, like = r), (fast, slow))
    betas = torch.cat((slow, fast))

    valid = r.isfinite()
    r, valid = map(lambda t: rearrange(t, '... t -> ... 1 t'), (r.where(valid, 0.), valid.to(r.dtype)))

    sd = (decayed(valid, slow * slow) + decayed(valid, fast * fast) - 2. * decayed(valid, slow * fast)).clamp(min = EPS).sqrt()
    weight = torch.cat((1. / sd, -1. / sd), dim = -2)

    # equal spans share one decayed sum and add their weights - by index_add, as a matmul sums in a batch dependent order (on
    # CUDA three or more equal spans may add in any order). decays that require grad keep every row

    spans = list(range(betas.shape[0])) if betas.requires_grad else flat(spans)
    distinct = list(dict.fromkeys(spans))
    betas, index = betas[list(map(spans.index, distinct))], torch.tensor(list(map(distinct.index, spans)), device = r.device)
    weight = torch.zeros_like(weight[..., :len(distinct), :]).index_add(-2, index, weight)

    trend = reduce(weight * decayed(r, betas), '... u t -> ... t', 'sum')

    # the variance sum_ab w_a w_b decayed(r^2, beta_a beta_b) over pairs a <= b, folded - one pair's rows in memory at a time

    square = r[..., 0, :].square()
    term = lambda a, b: (2. - (a == b)) * weight[..., a, :] * weight[..., b, :] * decayed(square, betas[a] * betas[b])
    var = fold(lambda var, ab: var + term(*ab), filter(lambda ab: ab[0] <= ab[1], product(range(len(distinct)), repeat = 2)), square.new_zeros(()))

    return trend / var.clamp(min = EPS).sqrt()

# time series momentum - sum of the last `lookback` returns
# over the square root of how many were valid, in return units like ewmac
# the running sum keeps a huge bad tick forever in float32 - use float64 when outliers are possible

def tsmom(r, lookback = 252):
    valid = r.isfinite()
    total, count = map(lambda t: t.cumsum(dim = -1), (r.where(valid, 0.), valid.to(r.dtype)))
    window_total, window_count = map(lambda t: t - shift(t, lookback), (total, count))
    return window_total / window_count.clamp(min = 1.).sqrt()

# volatility targeted position - signal in standard deviations, capped at 2, times target / annualised vol
# target_vol is the risk of a one sd signal
# no leverage limit: clamp the position after. a falling vol estimate (flat or stale prices, calm regimes) raises leverage
# annualising by sqrt(periods) assumes serially uncorrelated returns
# spans, warmup and the 10 year anchor count rows: periods must be the data's rows per year (weekend rows make ~300, so 252
# sizes positions ~9% too large) and other bar lengths need their spans rescaled

def position(signal, vol, target_vol = .15, cap = 2., periods = 252):
    assert cap >= 0, 'cap must not be negative'

    return signal.clamp(-cap, cap) * target_vol / (vol.clamp(min = EPS) * periods ** .5)

# the position decided at close t earns the return of t + lag - the only place the lag lives. lag 1 trades at the close the
# signal saw, lag 2 a day later
# cost is per unit of traded notional (2bp = 2e-4), entry included, no carry or financing. a per asset or
# per day cost tensor broadcasts. the trade ignores the drift of the held notional with price

def strategy_returns(pos, r, cost = 0., lag = 1):
    assert lag >= 1, 'lag must be at least 1, otherwise the strategy trades on returns it has not seen yet'

    held = shift(pos, lag)
    return held * r.where(r.isfinite(), 0.) - cost * (held - shift(held, 1)).abs()

# sqrt(periods) annualising is moved by serial correlation of the strategy's returns
# the zero positions of the warmup pull it towards 0, so slice them off

def sharpe(rets, periods = 252, eps = EPS, dim = -1):
    var, mean = torch.var_mean(rets, dim = dim, correction = 0)
    return mean / (var + eps).sqrt() * periods ** .5

# reference wiring of the pieces - returns in, volatility targeted positions out
# the signal is the t-statistic of an equal risk blend of ewmac speeds, so volatility only sizes the position
# the sizing volatility moves `long_run` of the way to its 10 year average over valid rows, blending volatilities
# the long run part damps the leverage calm or stale spells build
# positions are per asset, each aiming at target_vol alone - portfolio risk scaling and trade buffers are left to the user
# trading part of the way to the position each day is an ewma of the positions
# speeds, cap, vol_span and long_run follow published practice, target_vol only scales - none fitted to returns (EVIDENCE.md)

class TrendFollowing(Module):
    def __init__(
        self,
        *,
        speeds = ((8, 32), (16, 64), (32, 128), (64, 256)),
        vol_span = 35.,
        long_run = .3,
        target_vol = .15,
        cap = 2.,
        periods = 252,
        warmup = 75
    ):
        super().__init__()
        self.speeds = speeds

        self.vol_span = vol_span
        self.long_run = long_run
        self.target_vol = target_vol
        self.cap = cap
        self.periods = periods
        self.warmup = warmup

    def forward(
        self,
        returns,
        return_intermediates = False
    ):
        fast, slow = zip(*self.speeds)

        signal = ewmac_tstat(returns, fast, slow)

        vol = ewm_vol(returns, self.vol_span)
        vol = vol.lerp(ewma(vol.where(returns.isfinite(), torch.nan), 10. * self.periods), self.long_run)

        pos = position(signal, vol, self.target_vol, self.cap, self.periods)
        pos = pos.where(valid_count(returns) > self.warmup, 0.)

        if not return_intermediates:
            return pos

        return pos, Intermediates(vol, signal)
