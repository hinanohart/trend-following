//! Minimal core of systematic trend following.
//!
//! Every rule here is a causal linear filter of past returns. The reference wiring reads its blended
//! filter as a t-statistic, the filter over the square root of its own weighted sum of squared
//! returns, so no volatility estimate enters the signal. Each piece is a small state machine with
//! `update(x) -> y`, and has a lazy batch form over any iterator of returns:
//!
//! - [`Ewma`] / [`ewma`]: exponentially weighted mean over valid observations (pandas `adjust = True`)
//! - [`EwmVol`] / [`ewm_vol`]: zero mean exponentially weighted RMS of returns, the ex-ante volatility
//! - [`Ewmac`] / [`ewmac`]: moving average crossover of the cumulative return index, unit variance on iid returns
//! - [`EwmacTstat`] / [`ewmac_tstat`]: t-statistic of an equal risk blend of crossovers, unit variance
//!   under any volatility path when the signs of returns are symmetric
//! - [`Tsmom`] / [`tsmom`]: time series momentum, in the same units as [`Ewmac`]
//! - [`position`], [`strategy_returns`], [`sharpe`]: sizing and accounting
//! - [`TrendFollowing`]: reference wiring, returns in, volatility targeted positions out, with its [`Intermediates`]
//!
//! Returns are simple excess returns per period; for back-adjusted futures a return is the adjusted
//! price change over the unadjusted price of the contract held. Non-finite returns (`NaN`, `±inf`) are
//! missing: they add nothing and only let time pass. An exact zero is an observation (a flat print):
//! mark known stale prints `NaN` yourself, since treating every zero as missing inflates the volatility
//! of tick-sized markets. Time passes exactly for about 240 missing rows of the signal and 534 of the
//! volatility (span 35); past that the `1e-12` guard against dividing by zero acts, so split series at
//! longer gaps. Breakouts (where price sits in a window's range) are not
//! linear filters and are not covered.
//!
//! Nothing here is tuned to make money. Spans, caps and targets chosen on the same data they are
//! evaluated on will overfit. No setting here is a strategy. Sources and measurements: EVIDENCE.md.
//!
//! ```
//! use trend_following::{Config, TrendFollowing, strategy_returns};
//!
//! let returns = [0.01, -0.004, 0.007, f64::NAN, 0.002, -0.003];
//! let mut model = TrendFollowing::new(&Config::default());
//! let positions: Vec<f64> = returns.iter().map(|&r| model.update(r)).collect();
//! let pnl: Vec<f64> = strategy_returns(&positions, &returns, 2e-4, 1).collect();
//!
//! assert!(positions.iter().chain(&pnl).all(|x| x.is_finite()));
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::VecDeque;
use std::iter;

/// Guard for divisions and square roots.
pub const EPS: f64 = 1e-12;

/// Decay factor `beta = 1 - alpha` of an exponentially weighted average with pandas-style `span`
/// (`alpha = 2 / (span + 1)`). A centre of mass `c` is span `2c + 1`; a halflife `h` is
/// `beta = 0.5^(1/h)`.
///
/// ```
/// assert_eq!(trend_following::beta_from_span(3.0), 0.5);
/// ```
///
/// # Panics
///
/// If `span` is below 1, infinite or `NaN`: `beta` would leave `[0, 1)`.
#[must_use]
pub const fn beta_from_span(span: f64) -> f64 {
    assert!(
        span >= 1.0 && span < f64::INFINITY,
        "span must be finite and at least 1"
    );
    (span - 1.0) / (span + 1.0)
}

/// The value if finite, else zero, and whether it was finite.
fn split(x: f64) -> (f64, bool) {
    if x.is_finite() {
        (x, true)
    } else {
        (0.0, false)
    }
}

/// Exponentially weighted mean over valid observations, pandas `ewm(span, adjust = True)`.
///
/// A missing value adds nothing but still decays older weights, so the mean carries over a gap.
/// Before the first valid observation the mean is `0`.
///
/// ```
/// use trend_following::Ewma;
///
/// let mut ewma = Ewma::new(3.0);
/// assert_eq!(ewma.update(1.0), 1.0);
/// assert_eq!(ewma.update(4.0), 3.0); // (0.5 * 1 + 4) / (0.5 + 1)
/// assert_eq!(ewma.update(f64::NAN), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ewma {
    beta: f64,
    num: f64,
    den: f64,
}

impl Ewma {
    /// A new mean with pandas-style `span`.
    ///
    /// # Panics
    ///
    /// If `span` is below 1, infinite or `NaN`.
    #[must_use]
    pub const fn new(span: f64) -> Self {
        Self {
            beta: beta_from_span(span),
            num: 0.0,
            den: 0.0,
        }
    }

    /// Feeds one observation and returns the current mean.
    pub fn update(&mut self, x: f64) -> f64 {
        let (x, valid) = split(x);
        self.push(x, valid)
    }

    fn push(&mut self, x: f64, valid: bool) -> f64 {
        self.num = self.beta * self.num + x;
        self.den = self.beta * self.den + f64::from(u8::from(valid));
        self.num / self.den.max(EPS)
    }
}

/// Ex-ante volatility: zero mean exponentially weighted RMS of returns.
///
/// One bad print still counts, so set bad prints to `NaN` first.
///
/// ```
/// use trend_following::EwmVol;
///
/// let mut vol = EwmVol::new(60.0);
/// assert_eq!(vol.update(-0.02), 0.02);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwmVol(Ewma);

impl EwmVol {
    /// A new estimate with pandas-style `span`.
    ///
    /// # Panics
    ///
    /// If `span` is below 1, infinite or `NaN`.
    #[must_use]
    pub const fn new(span: f64) -> Self {
        Self(Ewma::new(span))
    }

    /// Feeds one return and returns the current volatility, never below `sqrt(EPS)`.
    pub fn update(&mut self, r: f64) -> f64 {
        let (r, valid) = split(r);
        self.0.push(r * r, valid).max(EPS).sqrt()
    }
}

/// Moving average crossover of the cumulative return index, computed on returns directly.
///
/// `slow_beta * S_slow(r) - fast_beta * S_fast(r)` with `S_b(x)_t = Σ_i b^i x_{t-i}`, divided by its
/// standard deviation per unit return volatility over the valid observations seen,
/// `Σ_i valid_{t-i} (slow_beta^(i+1) - fast_beta^(i+1))²`, kept by the same recursions run on the
/// validity mask. That gives unit variance on iid returns from the first step and through any gaps,
/// in return units: divide by volatility for a z-score.
///
/// The kernel's mean lag is `(fast + slow) / 2 - 1` periods, [`Tsmom`]'s `(lookback - 1) / 2`.
///
/// ```
/// use trend_following::Ewmac;
///
/// let mut ewmac = Ewmac::new(16.0, 64.0);
/// assert!((ewmac.update(0.01) - 0.01).abs() < 1e-15); // one return: the crossover is that return
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ewmac {
    betas: (f64, f64),
    sums: (f64, f64),
    masses: (f64, f64, f64),
}

impl Ewmac {
    /// A new crossover of a `fast` and a `slow` pandas-style span.
    ///
    /// # Panics
    ///
    /// If a span is below 1, infinite or `NaN`.
    #[must_use]
    pub const fn new(fast: f64, slow: f64) -> Self {
        Self {
            betas: (beta_from_span(fast), beta_from_span(slow)),
            sums: (0.0, 0.0),
            masses: (0.0, 0.0, 0.0),
        }
    }

    /// Feeds one return and returns the normalised crossover.
    pub fn update(&mut self, r: f64) -> f64 {
        let (crossover, sd) = self.step(r);
        crossover / sd
    }

    /// Feeds one return and returns the crossover and its standard deviation per unit return
    /// volatility over the valid observations seen.
    fn step(&mut self, r: f64) -> (f64, f64) {
        let (r, valid) = split(r);
        let (fast, slow) = self.betas;

        let seen = f64::from(u8::from(valid));
        let (masses, mass) = (self.masses, |decay: f64, m: f64| decay * m + seen);

        self.sums = (fast * self.sums.0 + r, slow * self.sums.1 + r);
        self.masses = (
            mass(fast * fast, masses.0),
            mass(slow * slow, masses.1),
            mass(slow * fast, masses.2),
        );

        let crossover = slow * self.sums.1 - fast * self.sums.0;
        let risk = fast * fast * self.masses.0 + slow * slow * self.masses.1
            - 2.0 * slow * fast * self.masses.2;
        (crossover, risk.max(EPS).sqrt())
    }
}

/// t-statistic of an equal risk blend of [`Ewmac`] speeds: one kernel
/// `W_i = Σ_k (slow_k^(i+1) - fast_k^(i+1)) / sd_k`, with `sd_k` the iid standard deviation of speed `k`
/// as in [`Ewmac`], over `sqrt(Σ_i W_i² r_(t-i)²)` (heteroskedasticity consistent).
///
/// Unit variance whenever the signs of returns are symmetric given their sizes: any volatility path,
/// any gaps, no volatility estimate. One return alone gives ±1, so a lone outlier moves it by about
/// one and a large value needs a persistent trend. Sign dependent volatility (the leverage effect of
/// equities) bends it, as it bends every normalisation, and serial correlation `rho` scales its sd by
/// about `sqrt((1 + rho) / (1 - rho))`. With a box kernel it is the rolling t-statistic of returns.
/// State grows as `u (u + 1) / 2` for `u` distinct spans.
///
/// ```
/// use trend_following::EwmacTstat;
///
/// let mut tstat = EwmacTstat::new(&[(8.0, 32.0), (16.0, 64.0)]);
/// assert_eq!(tstat.update(f64::NAN), 0.0);
/// assert!((tstat.update(-1e3) + 1.0).abs() < 1e-12); // a lone return is one standard deviation
/// assert!((tstat.update(0.0) + 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct EwmacTstat {
    crossovers: Vec<Ewmac>,
    owners: Vec<usize>,
    weights: Vec<f64>,
    decays: Vec<f64>,
    squares: Vec<f64>,
}

impl EwmacTstat {
    /// A new t-statistic of the `(fast, slow)` pandas-style spans, blended with equal risk.
    ///
    /// # Panics
    ///
    /// If a span is below 1, infinite or `NaN`.
    #[must_use]
    pub fn new(speeds: &[(f64, f64)]) -> Self {
        // distinct spans in order of first appearance, the slow ones first (as Python)
        let (mut spans, mut owners) = (Vec::new(), Vec::new());
        for span in speeds.iter().map(|s| s.1).chain(speeds.iter().map(|s| s.0)) {
            let known = spans
                .iter()
                .position(|&k: &f64| k.to_bits() == span.to_bits());
            owners.push(known.unwrap_or(spans.len()));
            if known.is_none() {
                spans.push(span);
            }
        }
        let betas: Vec<f64> = spans.into_iter().map(beta_from_span).collect();
        let decays: Vec<f64> = betas
            .iter()
            .enumerate()
            .flat_map(|(a, &left)| betas[a..].iter().map(move |&right| left * right))
            .collect();

        Self {
            crossovers: speeds
                .iter()
                .map(|&(fast, slow)| Ewmac::new(fast, slow))
                .collect(),
            owners,
            weights: vec![0.0; betas.len()],
            squares: vec![0.0; decays.len()],
            decays,
        }
    }

    /// Feeds one return and returns the t-statistic.
    pub fn update(&mut self, r: f64) -> f64 {
        self.weights.fill(0.0);
        let (slows, fasts) = self.owners.split_at(self.crossovers.len());
        let trend: f64 = self
            .crossovers
            .iter_mut()
            .zip(slows.iter().zip(fasts))
            .map(|(crossover, (&slow, &fast))| {
                let (value, sd) = crossover.step(r);
                // equal spans add their weights into one (as Python)
                self.weights[slow] += 1.0 / sd;
                self.weights[fast] -= 1.0 / sd;
                value / sd
            })
            .sum();

        let (square, weights) = (split(r).0.powi(2), &self.weights);
        // pairs a <= b, the ones off the diagonal twice (as Python)
        let pairs = weights.iter().enumerate().flat_map(|(a, &left)| {
            let factors = iter::once(1.0).chain(iter::repeat(2.0));
            factors
                .zip(&weights[a..])
                .map(move |(factor, &right)| factor * left * right)
        });
        let var: f64 = self
            .squares
            .iter_mut()
            .zip(&self.decays)
            .zip(pairs)
            .map(|((sum, &decay), pair)| {
                *sum = decay * (*sum + square);
                pair * *sum
            })
            .sum();

        trend / var.max(EPS).sqrt()
    }
}

/// Time series momentum: the sum of the last `lookback` returns
/// over the square root of how many of them were valid, in return units like [`Ewmac`].
///
/// ```
/// use trend_following::Tsmom;
///
/// let mut tsmom = Tsmom::new(2);
/// tsmom.update(0.03);
/// tsmom.update(0.01);
/// let momentum = tsmom.update(0.01); // the 0.03 has left the window
/// assert!((momentum - 0.02 / 2f64.sqrt()).abs() < 1e-15);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Tsmom {
    window: VecDeque<(f64, bool)>,
    lookback: usize,
    sum: f64,
    count: u32,
}

impl Tsmom {
    /// A new momentum over the last `lookback` periods.
    #[must_use]
    pub fn new(lookback: usize) -> Self {
        Self {
            window: VecDeque::new(),
            lookback,
            sum: 0.0,
            count: 0,
        }
    }

    /// Feeds one return and returns the normalised window sum.
    pub fn update(&mut self, r: f64) -> f64 {
        let (r, valid) = split(r);
        self.window.push_back((r, valid));
        self.sum += r;
        self.count += u32::from(valid);

        if self.window.len() > self.lookback
            && let Some((old, was_valid)) = self.window.pop_front()
        {
            self.sum -= old;
            self.count -= u32::from(was_valid);
        }

        self.sum / f64::from(self.count.max(1)).sqrt()
    }
}

/// [`Ewma`] over a whole series, lazily.
///
/// ```
/// let means: Vec<f64> = trend_following::ewma([1.0, 4.0], 3.0).collect();
/// assert_eq!(means, [1.0, 3.0]);
/// ```
///
/// # Panics
///
/// If `span` is below 1, infinite or `NaN`.
pub fn ewma(xs: impl IntoIterator<Item = f64>, span: f64) -> impl Iterator<Item = f64> {
    xs.into_iter()
        .scan(Ewma::new(span), |state, x| Some(state.update(x)))
}

/// [`EwmVol`] over a whole series, lazily.
///
/// ```
/// let vol: Vec<f64> = trend_following::ewm_vol([0.02, -0.02], 60.0).collect();
/// assert!((vol[1] - 0.02).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If `span` is below 1, infinite or `NaN`.
pub fn ewm_vol(returns: impl IntoIterator<Item = f64>, span: f64) -> impl Iterator<Item = f64> {
    returns
        .into_iter()
        .scan(EwmVol::new(span), |state, r| Some(state.update(r)))
}

/// [`Ewmac`] over a whole series, lazily.
///
/// ```
/// let trend: Vec<f64> = trend_following::ewmac([0.01; 3], 16.0, 64.0).collect();
/// assert!(trend.iter().all(|&y| y > 0.0));
/// ```
///
/// # Panics
///
/// If a span is below 1, infinite or `NaN`.
pub fn ewmac(
    returns: impl IntoIterator<Item = f64>,
    fast: f64,
    slow: f64,
) -> impl Iterator<Item = f64> {
    returns
        .into_iter()
        .scan(Ewmac::new(fast, slow), |state, r| Some(state.update(r)))
}

/// [`EwmacTstat`] over a whole series, lazily.
///
/// ```
/// let z: Vec<f64> = trend_following::ewmac_tstat([0.01, 0.02], &[(8.0, 32.0)]).collect();
/// assert!((z[0] - 1.0).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// If a span is below 1, infinite or `NaN`.
pub fn ewmac_tstat(
    returns: impl IntoIterator<Item = f64>,
    speeds: &[(f64, f64)],
) -> impl Iterator<Item = f64> {
    returns
        .into_iter()
        .scan(EwmacTstat::new(speeds), |state, r| Some(state.update(r)))
}

/// [`Tsmom`] over a whole series, lazily.
///
/// ```
/// let momentum: Vec<f64> = trend_following::tsmom([0.01, -0.01, 0.02], 2).collect();
/// assert_eq!(momentum[2], 0.01 / 2f64.sqrt());
/// ```
pub fn tsmom(returns: impl IntoIterator<Item = f64>, lookback: usize) -> impl Iterator<Item = f64> {
    returns
        .into_iter()
        .scan(Tsmom::new(lookback), |state, r| Some(state.update(r)))
}

/// Volatility targeted position: `signal` in standard deviations, capped at `±cap`, times
/// `target_vol / (vol * sqrt(periods))`.
///
/// `target_vol` is the risk of a one standard deviation signal. There is no leverage limit: clamp the
/// position after. A falling volatility estimate (flat or stale prices, calm regimes) raises leverage
/// after warm-up. Annualising by `sqrt(periods)` assumes serially uncorrelated returns.
///
/// ```
/// assert_eq!(trend_following::position(5.0, 0.01, 0.1, 2.0, 100.0), 2.0);
/// ```
///
/// # Panics
///
/// If `cap` is negative or `NaN`, like [`f64::clamp`].
#[must_use]
pub fn position(signal: f64, vol: f64, target_vol: f64, cap: f64, periods: f64) -> f64 {
    signal.clamp(-cap, cap) * target_vol / (vol.clamp(EPS, f64::INFINITY) * periods.sqrt())
}

/// Strategy returns: the position decided at close `t` earns the return of `t + lag`, minus `cost`
/// per unit of traded notional (`2bp = 2e-4`, entry included, no carry or financing). The trade
/// ignores the drift of the held notional with price.
/// The only place the lag lives: lag 1 trades at the close the signal saw, lag 2 a day later.
///
/// ```
/// let pnl: Vec<f64> = trend_following::strategy_returns(&[1.0, 1.0], &[0.01, 0.02], 0.0, 1).collect();
/// assert_eq!(pnl, [0.0, 0.02]);
/// ```
///
/// # Panics
///
/// If `lag` is `0`: the strategy would trade on returns it has not seen yet.
/// If `positions` and `returns` differ in length.
pub fn strategy_returns(
    positions: &[f64],
    returns: &[f64],
    cost: f64,
    lag: usize,
) -> impl Iterator<Item = f64> {
    assert_eq!(positions.len(), returns.len(), "one position per return");
    assert!(
        lag >= 1,
        "lag must be at least 1, otherwise the strategy trades on returns it has not seen yet"
    );

    let held = || iter::repeat_n(0.0, lag).chain(positions.iter().copied());

    held()
        .zip(iter::once(0.0).chain(held()))
        .zip(returns)
        .map(move |((held, previous), &r)| held * split(r).0 - cost * (held - previous).abs())
}

/// Annualised Sharpe ratio, `mean / sqrt(var + EPS) * sqrt(periods)` with the population variance.
/// `NaN` for an empty slice. Serial correlation of the strategy's returns moves the `sqrt(periods)`
/// annualising. The zero positions of the warm-up pull it towards 0, so leave them out.
///
/// ```
/// assert!(trend_following::sharpe(&[0.01, 0.03], 252.0) > 0.0);
/// ```
#[must_use]
pub fn sharpe(returns: &[f64], periods: f64) -> f64 {
    let (count, total) = returns
        .iter()
        .fold((0.0, 0.0), |(count, total), &r| (count + 1.0, total + r));
    let mean = total / count;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / count;
    mean / (var + EPS).sqrt() * periods.sqrt()
}

/// Settings of [`TrendFollowing`], none fitted to returns: speeds, cap, `vol_span` and `long_run` follow
/// published practice (sources in EVIDENCE.md); `target_vol` only scales.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// `(fast, slow)` spans of the crossovers, blended with equal risk.
    pub speeds: Vec<(f64, f64)>,
    /// Span of the volatility estimate.
    pub vol_span: f64,
    /// Share of the way the sizing volatility moves to its 10 year average over valid returns,
    /// blending volatilities.
    pub long_run: f64,
    /// Annualised volatility the position aims for.
    pub target_vol: f64,
    /// Cap on the signal, in standard deviations.
    pub cap: f64,
    /// Periods per year. Spans, warm-up and the 10 year anchor count rows, so this must be the
    /// data's rows per year (weekend rows make about 300, and 252 then sizes positions about 9% too
    /// large); other bar lengths need their spans rescaled.
    pub periods: f64,
    /// Valid observations to see before the first non-zero position.
    pub warmup: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            speeds: vec![(8.0, 32.0), (16.0, 64.0), (32.0, 128.0), (64.0, 256.0)],
            vol_span: 35.0,
            long_run: 0.3,
            target_vol: 0.15,
            cap: 2.0,
            periods: 252.0,
            warmup: 75,
        }
    }
}

/// What [`TrendFollowing::update`] computed on the way to its position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Intermediates {
    /// Sizing volatility per period: [`EwmVol`] moved `long_run` of the way to its 10 year average.
    pub vol: f64,
    /// t-statistic of the blended speeds, from [`EwmacTstat`], before the cap.
    pub signal: f64,
}

/// Reference wiring of the pieces: returns in, volatility targeted positions out, one step at a time.
///
/// The signal is the [`EwmacTstat`] of the speeds, so no volatility enters it. Volatility only sizes,
/// through [`position`], moving `long_run` of the way to its 10 year average over valid returns (see
/// [`Config::long_run`]); the long run part damps the leverage that calm or stale spells build. Zero
/// until more than `warmup` valid returns have been seen.
///
/// Positions are per asset, each aiming at `target_vol` alone: portfolio risk scaling and trade
/// buffers are left to the user. Trading part of the way to the position each day is an [`Ewma`] of
/// the positions.
///
/// ```
/// use trend_following::{Config, Intermediates, TrendFollowing};
///
/// let mut model = TrendFollowing::new(&Config::default());
/// let position = model.update(0.01);
/// let Intermediates { vol, signal } = model.intermediates();
///
/// assert_eq!(position, 0.0); // nothing is held during warm-up
/// assert!((vol - 0.01).abs() < 1e-15); // one return so far: the volatility is its size
/// assert!((signal - 1.0).abs() < 1e-12); // so the signal is one standard deviation up
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TrendFollowing {
    vol: EwmVol,
    average: Ewma,
    long_run: f64,
    trend: EwmacTstat,
    count: u32,
    target_vol: f64,
    cap: f64,
    periods: f64,
    warmup: u32,
    last: Intermediates,
}

impl TrendFollowing {
    /// A new model with the given settings.
    ///
    /// # Panics
    ///
    /// If a span of `config` (`10 * periods` for the long run average included) is below 1,
    /// infinite or `NaN`.
    #[must_use]
    pub fn new(config: &Config) -> Self {
        Self {
            vol: EwmVol::new(config.vol_span),
            average: Ewma::new(10.0 * config.periods),
            long_run: config.long_run,
            trend: EwmacTstat::new(&config.speeds),
            count: 0,
            target_vol: config.target_vol,
            cap: config.cap,
            periods: config.periods,
            warmup: config.warmup,
            last: Intermediates::default(),
        }
    }

    /// Feeds one return and returns the position to hold over the next period.
    pub fn update(&mut self, r: f64) -> f64 {
        let current = self.vol.update(r);
        let average = self
            .average
            .update(if r.is_finite() { current } else { f64::NAN });
        let vol = current + self.long_run * (average - current);
        let signal = self.trend.update(r);

        self.count = self.count.saturating_add(u32::from(r.is_finite()));
        self.last = Intermediates { vol, signal };

        if self.count > self.warmup {
            position(signal, vol, self.target_vol, self.cap, self.periods)
        } else {
            0.0
        }
    }

    /// Volatility and signal behind the last position, like Python's `return_intermediates`.
    #[must_use]
    pub const fn intermediates(&self) -> Intermediates {
        self.last
    }
}
