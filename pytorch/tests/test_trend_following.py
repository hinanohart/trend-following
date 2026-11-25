import ast
import csv
from itertools import chain
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck
from torch.autograd.functional import jacobian

import trend_following_pytorch
from trend_following_pytorch import (
    ewma,
    ewm_vol,
    ewmac,
    ewmac_tstat,
    tsmom,
    position,
    strategy_returns,
    sharpe,
    TrendFollowing
)
from trend_following_pytorch.trend_following import scan, beta_from_span

TESTDATA = Path(__file__).parents[2] / 'testdata'
CASES = ('iid', 'trend', 'zeros', 'gaps')

# helpers

def returns(*shape, seed = 0, dtype = torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator = gen, dtype = dtype) * 1e-2

def random_signs(*shape, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, shape, generator = gen).double() * 2. - 1.

def load(path):
    lines = filter(lambda line: not line.startswith('#'), path.read_text().splitlines())
    header, *rows = csv.reader(lines)
    columns = map(lambda col: torch.tensor(list(map(float, col)), dtype = torch.float64), zip(*rows))
    return dict(zip(header, columns))

def compute(r):
    pos, (_, signal) = TrendFollowing()(r, return_intermediates = True)
    return dict(
        ewma = ewma(r, 10.),
        ewm_vol = ewm_vol(r, 35.),
        ewmac = ewmac(r, 16., 64.),
        tsmom = tsmom(r, 20),
        position = pos,
        ret_lag1 = strategy_returns(pos, r),
        ret_lag2_cost = strategy_returns(pos, r, cost = 2e-4, lag = 2),
        signal = signal
    )

def kernel_tstat(r, fast, slow):
    # explicit lower triangular kernel matrix in float64, independent of scan
    t = r.shape[-1]
    lag = (torch.arange(t)[:, None] - torch.arange(t)[None, :]).double()
    power = lag.clamp(min = 0.) + 1.
    w = torch.where(lag >= 0, beta_from_span(slow) ** power - beta_from_span(fast) ** power, 0.)
    r = r.nan_to_num(0.)
    return (r @ w.T) / (r.square() @ w.square().T).clamp(min = 1e-12).sqrt()

# scan

@pytest.mark.parametrize('t', (1, 2, 257))
@pytest.mark.parametrize('beta', (0., .5, .99))
def test_scan_matches_decay_matrix(t, beta):
    b = returns(3, t)

    lag = torch.arange(t)
    lag = (lag[:, None] - lag[None, :]).double()
    decay = torch.where(lag >= 0, beta ** lag.clamp(min = 0), 0.)
    expected = b @ decay.T

    assert torch.allclose(scan(b, beta), expected, rtol = 1e-12, atol = 1e-15)
    assert torch.allclose(scan(b.float(), beta), expected.float(), rtol = 1e-5, atol = 1e-6)

# golden vectors from testdata/generate.py, independent of this implementation

@pytest.mark.parametrize('case', CASES)
def test_golden_float64(case):
    golden = load(TESTDATA / f'{case}.csv')
    ours = compute(golden['r'])
    off = list(filter(lambda key: not torch.allclose(ours[key], golden[key], rtol = 1e-9, atol = 1e-12), ours))
    assert not off, off

@pytest.mark.parametrize('case', CASES)
def test_golden_float32(case):
    golden = load(TESTDATA / f'{case}.csv')
    ours = compute(golden['r'].float())
    close = lambda key: torch.allclose(ours[key].double(), golden[key], rtol = 1e-4, atol = 1e-6 * golden[key].abs().max().item())
    off = list(filter(lambda key: not close(key), ours))
    assert not off, off

def test_golden_sharpe():
    header, *rows = csv.reader(filter(lambda line: not line.startswith('#'), (TESTDATA / 'sharpe.csv').read_text().splitlines()))
    expected = dict(map(lambda row: (row[0], torch.tensor(list(map(float, row[1:])), dtype = torch.float64)), rows))
    ours = dict(map(lambda case: (case, compute(load(TESTDATA / f'{case}.csv')['r'])), CASES))
    got = dict(map(lambda case: (case, sharpe(torch.stack((ours[case]['ret_lag1'], ours[case]['ret_lag2_cost'])))), CASES))
    assert all(map(lambda case: torch.allclose(got[case], expected[case], rtol = 1e-9, atol = 1e-12), CASES))

# causality - nothing depends on the future

CAUSAL = dict(
    ewma = lambda r: ewma(r, 10.),
    ewm_vol = lambda r: ewm_vol(r, 10.),
    ewmac = lambda r: ewmac(r, 4., 16.),
    ewmac_tstat = lambda r: ewmac_tstat(r, (4., 8.), (16., 32.)),
    tsmom = lambda r: tsmom(r, 5),
    trend_following = TrendFollowing(warmup = 3)
)

@pytest.mark.parametrize('name', CAUSAL.keys())
def test_causal(name):
    jac = jacobian(CAUSAL[name], returns(32))
    assert (jac.triu(diagonal = 1) == 0.).all()

def test_strategy_returns_never_see_the_return_they_trade_on():
    r = returns(32)
    jac = jacobian(lambda pos: strategy_returns(pos, r, cost = 2e-4), returns(32, seed = 1))
    assert (jac.triu() == 0.).all()

def test_lag_zero_is_refused():
    with pytest.raises(AssertionError):
        strategy_returns(returns(8), returns(8), lag = 0)

# gradients

def test_gradcheck_through_returns_and_spans():
    r = returns(2, 40).requires_grad_()
    spans = torch.tensor([4., 16.], dtype = torch.float64, requires_grad = True)
    assert gradcheck(lambda r, fast: ewmac(r, fast, fast * 4.), (r, spans))
    assert gradcheck(lambda r, fast: ewmac_tstat(r, fast, fast * 4.), (r, spans))

def test_gradcheck_through_the_reference_wiring():
    r = returns(2, 40).requires_grad_()
    assert gradcheck(lambda r: sharpe(strategy_returns(TrendFollowing(warmup = 3)(r), r, cost = 2e-4)), (r,))

def test_missing_data_is_masked_and_gradients_stay_finite():
    r = returns(2, 200)
    r[:, :50] = torch.nan
    r[:, 120] = torch.nan
    r[0, 150] = torch.inf
    r.requires_grad_()
    spans = torch.tensor([8., 16.], dtype = torch.float64, requires_grad = True)

    pos, (vol, signal) = TrendFollowing()(r, return_intermediates = True)
    outputs = (pos, vol, signal, ewmac(r, spans, spans * 4.), ewmac_tstat(r, spans, spans * 4.), ewma(r, 10.), tsmom(r, 20), strategy_returns(pos, r))
    assert all(map(lambda t: t.isfinite().all(), outputs))

    sum(map(lambda t: t.sum(), outputs), r.new_zeros(())).backward()
    assert r.grad.isfinite().all() and spans.grad.isfinite().all()

    n = r.isfinite().cumsum(dim = -1)
    assert (pos[n <= 75] == 0.).all() and (pos[n > 75] != 0.).any()

# normalisation

def test_ewmac_has_unit_variance_from_the_first_step():
    r = returns(20_000, 300)
    var = (ewmac(r, 16., 64.) / 1e-2)[:, (0, 19, 299)].var(dim = 0)
    assert torch.allclose(var, torch.ones_like(var), atol = 4 * (2 / 20_000) ** .5)

def test_ewmac_has_unit_variance_through_gaps():
    r = returns(20_000, 300)
    r[:, torch.arange(300) % 7 >= 5] = torch.nan
    r[:, 150:230] = torch.nan
    var = (ewmac(r, 16., 64.) / 1e-2)[:, (149, 230, 240, 299)].var(dim = 0)
    assert torch.allclose(var, torch.ones_like(var), atol = 4 * (2 / 20_000) ** .5)

# the t-statistic has unit second moment whenever signs are symmetric given sizes - any volatility path, gaps, spikes, stale prints

def test_tstat_has_unit_variance_under_any_volatility():
    n, t = 10_000, 320
    size = torch.full((t,), 1e-2, dtype = torch.float64)
    size[150:] *= 4.
    size[250:280] = 0.
    size[300] *= 50.

    r = size * (returns(n, t, seed = 1).abs() + 1e-3) * random_signs(n, t, seed = 2)
    r[:, 200:220] = torch.nan

    second = ewmac_tstat(r, (8., 32.), (32., 128.))[:, (0, 151, 160, 200, 225, 281, 301, 319)].square().mean(dim = 0)
    assert torch.allclose(second, torch.ones_like(second), atol = 4 * (2 / n) ** .5)

def test_a_lone_return_moves_the_tstat_by_one():
    r = torch.zeros(60, dtype = torch.float64)
    r[:5] = torch.nan
    r[10] = -1e3

    z = ewmac_tstat(r, (8., 16.), (32., 64.))
    assert (z[:10] == 0.).all() and torch.allclose(z[10:], torch.full_like(z[10:], -1.), rtol = 1e-12)

def test_single_speed_signal_is_the_kernel_tstat():
    r = returns(3, 300)
    r[1, :40] = torch.nan
    r[2, 100:130] = torch.nan
    _, (_, signal) = TrendFollowing(speeds = ((8.3, 32.7),))(r, return_intermediates = True)
    assert torch.allclose(signal, kernel_tstat(r, 8.3, 32.7), rtol = 1e-9, atol = 1e-12)

def test_speed_blend_has_unit_variance():
    r = returns(2_000, 300)
    _, (_, signal) = TrendFollowing()(r, return_intermediates = True)
    var = signal[:, (0, 19, 299)].var(dim = 0)
    assert torch.allclose(var, torch.ones_like(var), atol = 4 * (2 / 2_000) ** .5)

def test_scale_invariance():
    r = returns(3, 300)
    model = TrendFollowing()
    pos, (_, signal) = model(r, return_intermediates = True)
    scaled_pos, (_, scaled_signal) = model(r * 3., return_intermediates = True)
    assert torch.allclose(signal, scaled_signal) and torch.allclose(pos, scaled_pos * 3.)

def test_batch_equals_rows():
    r = returns(5, 400)
    r[1, :30] = torch.nan
    r[2, :170] = torch.nan
    r[3, 200:230] = torch.nan
    r[4, :1] = torch.nan

    model = TrendFollowing()
    assert torch.allclose(model(r), torch.stack(list(map(model, r))), rtol = 1e-12, atol = 0.)

# the signal needs no volatility estimate, so a volatility jump does not move its variance

@pytest.mark.parametrize('sigmas', ((5e-3, 2e-2), (2e-2, 5e-3)))
def test_signal_variance_stays_flat_across_a_vol_step(sigmas):
    sigma = torch.full((600,), sigmas[0], dtype = torch.float64)
    sigma[300:] = sigmas[1]

    r = returns(1000, 600) / 1e-2 * sigma
    _, (_, signal) = TrendFollowing()(r, return_intermediates = True)

    var = signal[:, (302, 310, 330, 360, 420, 599)].var(dim = 0)
    assert ((var > .8) & (var < 1.2)).all(), var

# the sizing volatility leans on its long run average, so a stale spell cannot blow up leverage

def test_stale_prices_raise_leverage_less_than_the_plain_estimate():
    r = returns(200, 1000)
    r[:, 700:760] = 0.
    _, (vol, _) = TrendFollowing()(r, return_intermediates = True)
    plain = ewm_vol(r, 35.)
    assert ((vol[:, 699] / vol[:, 759]) < (plain[:, 699] / plain[:, 759])).all()

# missing returns add nothing: rows before a series starts (a panel of assets with different start dates) change no later position

def test_leading_missing_returns_change_nothing():
    r = returns(3, 600)
    padded = torch.cat((torch.full((3, 300), torch.nan, dtype = r.dtype), r), dim = -1)
    model = TrendFollowing()
    assert torch.allclose(model(padded)[:, 300:], model(r), rtol = 1e-12, atol = 1e-15)

def test_position_is_capped_and_vol_targeted():
    pos = position(torch.tensor([-5., 1., 5.]), torch.full((3,), .01), target_vol = .1, cap = 2., periods = 100)
    assert torch.allclose(pos, torch.tensor([-2., 1., 2.]))

# shapes - time last, any leading dims

@pytest.mark.parametrize('shape', ((300,), (4, 300), (2, 4, 300)))
def test_shapes(shape):
    r = returns(*shape)
    assert TrendFollowing()(r).shape == shape
    assert ewmac(r).shape == shape
    assert ewmac(r, (8., 16.), (32., 64.)).shape == (*shape[:-1], 2, shape[-1])
    assert ewmac_tstat(r).shape == ewmac_tstat(r, (8., 16.), (32., 64.)).shape == shape

# pandas parity

def test_ewma_matches_pandas():
    pd = pytest.importorskip('pandas')

    r = returns(500)
    r[:30] = torch.nan
    r[100:105] = torch.nan

    expected = torch.from_numpy(pd.Series(r.numpy()).ewm(span = 10., adjust = True, ignore_na = False).mean().to_numpy())
    finite = expected.isfinite()
    assert torch.allclose(ewma(r, 10.)[finite], expected[finite], rtol = 1e-12, atol = 1e-15)

# the package is loop free - functional all the way down

def test_package_has_no_loops():
    files = list(Path(trend_following_pytorch.__file__).parent.glob('*.py'))
    assert len(files) == 2

    nodes = chain.from_iterable(map(lambda path: ast.walk(ast.parse(path.read_text())), files))
    loops = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    assert not any(map(lambda node: isinstance(node, loops), nodes))

# half precision quantises decays near 1, and float16 underflows small variances, so it is refused

@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16))
@pytest.mark.parametrize('name', ('ewma', 'ewm_vol', 'ewmac', 'ewmac_tstat', 'trend_following'))
def test_half_precision_is_refused(name, dtype):
    with pytest.raises(AssertionError):
        CAUSAL[name](returns(32).to(dtype))

# equal spans are found in python, so shapes are static - one graph under torch.compile, and vmap matches the batch call

@pytest.mark.skipif(
    torch.__version__ < '2.11',
    reason = 'one graph is tested on torch 2.11 - the dynamo of torch 2.0 cannot trace map over a partial'
)
def test_compiles_to_one_graph():
    r = returns(3, 300)
    r[1, :40] = torch.nan
    model = TrendFollowing()
    assert torch.equal(torch.compile(model, fullgraph = True, backend = 'eager')(r), model(r))

def test_vmap_equals_the_batch_call():
    r = returns(3, 300)
    r[1, :40] = torch.nan
    model = TrendFollowing()
    assert torch.equal(torch.func.vmap(model)(r), model(r))

# a negative cap is refused, as Rust's f64::clamp refuses it

def test_negative_cap_is_refused():
    with pytest.raises(AssertionError):
        position(torch.ones(3), torch.ones(3), cap = -1.)

# batch == rows bit for bit - forward and both intermediates, both precisions

def outputs(r):
    pos, (vol, signal) = TrendFollowing()(r, return_intermediates = True)
    return torch.stack((pos, vol, signal))

@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_batch_equals_rows_bit_for_bit(dtype):
    r = returns(5, 400, dtype = dtype)
    r[1, :30] = torch.nan
    r[2, :170] = torch.nan
    r[3, 200:230] = torch.nan
    assert torch.equal(outputs(r), torch.stack(list(map(outputs, r)), dim = 1))
