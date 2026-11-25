# Trend Following: evidence

How the core works, how to use it safely, where each default comes from, and the evidence behind each choice.

## How a position is made

1. **Trend.** Four moving average crossovers, each a fast minus a slow exponentially weighted average of the price index built from the returns, are blended with equal risk into one signal: a t-statistic that says how many standard deviations the recent trend is from zero. No volatility estimate enters it.
2. **Cap.** The signal is clipped to ±2.
3. **Volatility.** The typical size of a return: an exponentially weighted root mean square of the returns (span 35), moved 0.3 of the way to its 10 year average.
4. **Position.** `position = clamp(signal, -cap, cap) × target_vol / (vol × sqrt(periods))`, zero during the warm-up. A one standard deviation signal aims at `target_vol` annual risk, and a wilder market gets a smaller position.

A position is the notional held per unit of capital: the strategy's return is position × the asset's return. 1.5 holds 1.5 times the capital long; -0.5 holds half of it short.

## Notes for use

- Input: simple excess returns, one per period (in PyTorch, time is the last axis). For back-adjusted futures, a return is the adjusted price change over the unadjusted price of the contract held.
- `NaN` or `±inf` means missing: it adds nothing and time passes. An exact zero counts as a real (flat) print. Set bad or stale prints to `NaN` first: one 10 standard deviation print cuts the next position to 0.47 of the clean one and keeps it under 0.9 of it for 52 rows.
- Positions are zero until more than `warmup` valid returns have been seen, so leave those rows out of any accounting. With no missing returns at the start, P&L begins at row `warmup + lag`.
- The position set at the close of row t earns the return of row t + `lag`. `lag = 1` trades at the close the signal saw; `lag = 2` trades a day later.
- `periods` is the number of rows per year: 252 for trading days. Daily data that keeps weekend rows has about 300, and 252 then makes positions about 9% too large. For other bar lengths, rescale the spans too.
- Gaps of up to about 240 missing rows are handled exactly (534 for the volatility alone), in float32 and float64. Longer gaps reach the `1e-12` guard against division by zero, so split the series there. The longest inner gap in the 138 futures below is 142 rows.
- PyTorch takes float32 or float64 (Rust uses f64). Half precision rounds the decays near 1: in bfloat16, span 256 acts as span 511.
- Not included: portfolio risk scaling, trade buffers and breakout rules (see Signals, and Sizing and accounting).

## Defaults and where they come from

None is fitted to returns.

| Setting | Default | Source or reason |
|---|---|---|
| `speeds` (fast, slow spans) | (8, 32), (16, 64), (32, 128), (64, 256) | Carver's crossovers with slow = 4 × fast (Carver, 2015). Together they cover trends of one to twelve months, like the 1, 3 and 12 month mix of Hurst, Ooi & Pedersen (2017). |
| `vol_span` | 35 | pysystemtrade's default (Carver). Its half-life is about 12 days, like a 25 day simple average. |
| `long_run` | 0.3 | pysystemtrade's default: the volatility moves 0.3 of the way to its 10 year average. |
| `cap` | 2 standard deviations | Industry practice (Kurth, Eisler, Rej & Bouchaud, 2026). |
| `target_vol` | 0.15 | No published source; it only scales the positions. It is per asset, while published targets are for whole portfolios: 16% in pysystemtrade, 10% in Hurst, Ooi & Pedersen (2017). |
| `warmup` | 75 valid returns | This project's choice: about twice the volatility span, so the volatility has settled. The signal needs no warm-up. The first positions after it ran about 1.6% under target. |
| `periods` | 252 | Rows per year of daily data without weekend rows. |
| `tsmom` lookback | 252 | The 12 month time series momentum of Moskowitz, Ooi & Pedersen (2012). |
| `lag` | 1 | Trade at the close the signal saw (Moskowitz, Ooi & Pedersen, 2012). Index rules trade a day or more later (S&P DJI rules in Morgan Stanley, 2025, and Citigroup, 2025). |
| `cost` | 0 | Costs differ by market, so none is assumed. It is per unit of traded notional: 2 basis points is `2e-4`. |

Spans follow pandas: decay `beta = (span - 1) / (span + 1)`, and `ewma` is pandas `ewm(span, adjust = True)`.

## PyTorch or Rust

- Both pass the same golden files (`testdata/*.csv`, written by `testdata/generate.py` from the definitions, standard library only). They agree to about 9 significant digits (tolerance `1e-12 + 1e-9 × value`).
- PyTorch has no Python loop over time. The time recursion is a parallel scan (recursive doubling, Hillis & Steele, 1986), run in blocks.
- PyTorch memory: the default `TrendFollowing` peaks at about 0.2 kB per float32 input element (0.14 kB on CUDA). With backward on CPU it reaches 0.8 kB, mostly freed memory that glibc keeps. Rows are independent, so split long batches by rows.
- PyTorch on a GPU: the default speeds have 6 distinct spans (32 and 64 repeat), so the variance needs 21 decayed sums, each with its own kernel launches. A lone series pays for all of them: 10,000 returns take 38 ms forward on a T4 GPU. Batch many series together.
- `torch.compile(fullgraph = True)` captures the default `TrendFollowing` as one graph (tested on torch 2.11).
- Rust: besides `update`, every piece has a lazy iterator form over a whole series.

## Evidence

Two kinds of support appear below: published work (see References) and measurements made while building the library. The measurements are descriptive. The test suite does not re-run them, and none is a performance claim.

Data used for the measurements:

- Six simulated return processes. A few volatility checks used other simulated returns.
- 138 daily futures series from the price files of Robert Carver's pysystemtrade (`data/futures/multiple_prices_csv`, to 2024-03; GPL-3 data, used for measurement only, not shipped).
- Returns are taken within one contract, so roll days are missing. Weekend rows are dropped.
- Data errors were set missing by rules fixed before the run: spikes that reverse, prices at or below zero, tenfold scale breaks, runs of five or more zero returns.
- A series was dropped if errors exceeded 0.5% of its returns or fewer than 2520 valid returns remained. Of two series whose returns correlated 0.97 or more over 500 or more common days (a contract and its mini or micro twin), only the one with more valid returns was kept.

### The effect itself

- Pooled over many markets and two centuries, the trend effect is strong (Lempérière et al., 2014).
- Back to 1880, a diversified time series momentum strategy was consistently profitable over the 137 years studied (Hurst, Ooi & Pedersen, 2017).
- Asset by asset the evidence is weak, and the pooled t-statistic falls below bootstrap critical values (Huang, Li, Wang & Zhou, 2020).
- Short trends have withered: trends of a few days (Lempérière et al., 2014), and fast crossovers of days to weeks since about 2009 (Kurth et al., 2026).
- Spans, caps and targets chosen on the same data they are tested on will overfit (Bailey & López de Prado, 2014).

### Signals

- Every rule here is a causal linear filter of past returns. This one frame covers time series momentum and moving average crossovers (Levine & Pedersen, 2016).
- Breakouts are not covered. A breakout of window `n` correlates 0.92–0.94 with the crossover `(n / 5, 4n / 5)` (Carver, 2016).
- Mean lag (the average age of the weights) is `(fast + slow) / 2 - 1` for a crossover and `(lookback - 1) / 2` for time series momentum. The default speeds look back 19, 39, 79 and 159 days on average. On iid returns (independent, identically distributed) they best track time series momentum over 28, 57, 115 and 232 days (correlation 0.92).
- The t-statistic divides by a heteroskedasticity consistent standard error (White, 1980), so it stays in standard deviations when volatility changes.
- Serial correlation `rho` scales the t-statistic's standard deviation by about `sqrt((1 + rho) / (1 - rho))`: 3% at the median `|rho|` (0.03) of the 138 futures, 8% at their 90% quantile.
- On the 138 futures the t-statistic's mean square is 1.17 at the median (trends and serial dependence). With random signs it is 1.00.
- Not taken: dividing the raw crossover by a lagging volatility, which dips after volatility jumps; and filtering `r_t / vol_(t-1)` first (Dao et al., 2016; Sepp & Lucic, 2026), which needs a fitted scale and swings with regimes.

### Volatility

- Positions are re-sized daily, so sizing needs next day's variance. Range estimators (open, high, low, close) cut noise, not lag, and cut time series momentum turnover by only about 8% (Baltas & Kosowski, 2015). So the core takes closing returns.
- How spans were compared: the standard deviation of log(63 day realised risk / target), where lower is steadier. That test cannot see a constant bias, so every choice also had to keep realised / target within 5%, and was judged on turnover.
- Span 35 held risk steadier than span 75 on the six simulated processes and the 138 futures (down 0.035–0.054; 0.042 on the futures), for 7–10% more position turnover.
- Span 121 (Moskowitz, Ooi & Pedersen, 2012) was less steady on 5 of the 6 processes. Spans 32 (RiskMetrics decay 0.94; J.P. Morgan / Reuters, 1996) and 58 (half-life 20 days; Harvey et al., 2018) ran 4–9% above target; span 35 ran 2–4% above.
- Squares, not absolute returns. On Student t returns (4 degrees of freedom), absolute returns ran 12% above target (squares 4%), because their normal-case constant `sqrt(pi / 2)` no longer holds. They also turned over 22–24% more on three simulated processes. Clipping at 5 standard deviations was no steadier.
- Demeaning (Moskowitz, Ooi & Pedersen, 2012) moves the estimate by under 1%.
- The 10 year anchor of the long run blend held risk steadier than a 20 year one on the 138 futures.
- The long run part damps the leverage that calm or stale spells build: 1.4 times after 20 flat days, 1.8 without it.
- The goal is steady risk, not the best one day forecast. A variance blend with decay 0.94 and a GARCH implied long run share of 1/6 forecast next day's variance better on 86% of the 138 futures (the mean gain is not significant) and passed the same tests, but it would change three settings.
- Volatility target indices use other forms: the greater of two exponentially weighted volatilities (half-lives 10 and 23 days in an S&P DJI index, Morgan Stanley, 2025; decays 0.94 and 0.97 in the Morgan Stanley Amplitude Index, Morgan Stanley, 2026), or the lesser (an S&P DJI index, Citigroup, 2025).
- Tested per asset here, the greater of half-lives 10 and 23 with the long run blend ran 5.5% under target on equity futures and was less steady. Without the blend it was a little steadier (by 0.012–0.015) for about 4% more turnover, too little to change the default.

### Sizing and accounting

- Carver's cap, 20 at an average absolute forecast of 10 (Carver, 2015), is a little tighter: 1.6 standard deviations for normal forecasts.
- Realised risk differs from `target_vol` because the capped signal varies in size: 0.96 of it with no trend, 1.10 on the 138 futures (median, quartiles 1.05–1.19).
- Other published responses replace the cap: the sign (Moskowitz, Ooi & Pedersen, 2012), `tanh` saturation (Lempérière et al., 2014), and the reverting `y * exp(-y^2 / 4) / 0.89`, largest at `|y| = sqrt 2` (Baz et al., 2015, as given by Lim, Zohren & Roberts, 2019). Each is a one line change before `position`.
- Strong signals earn less than a linear response predicts. Part of it is volatility underestimated when it mean reverts; Carver attributes the rest to exhausted trends (Carver, 2020). The cap limits the bet on them.
- Annualising by `sqrt(periods)` assumes serially uncorrelated returns. Daily futures returns come close (variance ratios near 1 at 21 and 63 days), but a strategy's own returns are serially correlated, which makes `sharpe` about 5% too high at the median of the 138 futures (Lo, 2002).
- `strategy_returns` ignores the trades that come from the held notional drifting with price. They are under 10% of turnover.
- Left to the user: portfolio risk scaling (a covariance as in Hurst, Ooi & Pedersen, 2017, or Carver's diversification multiplier) and trade buffers (both in Carver, 2015). Trading part of the way to the position each day (Gârleanu & Pedersen, 2013) is an exponentially weighted average of the positions.

## References

- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5).
- Baltas, N. & Kosowski, R. (2015). Demystifying Time-Series Momentum Strategies: Volatility Estimators, Trading Rules and Pairwise Correlations. Working paper, version of 1 October 2015, SSRN 2140091, https://papers.ssrn.com/abstract=2140091.
- Baz, J., Granger, N., Harvey, C. R., Le Roux, N. & Rattray, S. (2015). Dissecting Investment Strategies in the Cross Section and Time Series. Working paper, SSRN 2695101, https://papers.ssrn.com/abstract=2695101.
- Carver, R. (2015). Systematic Trading. Harriman House.
- Carver, R. (2016). A simple breakout trading rule (pysystemtrade). Blog post, 20 May 2016, https://qoppac.blogspot.com/2016/05/a-simple-breakout-trading-rule.html.
- Carver, R. (2020). Forecast linearity and forecasting mean reverting volatility. Blog post, 1 September 2020, https://qoppac.blogspot.com/2020/09/forecast-linearity-and-forecasting-mean.html.
- Carver, R. pysystemtrade, https://github.com/robcarver17/pysystemtrade (default settings in `sysdata/config/defaults.yaml`; ideas only, no code is copied).
- Citigroup (2025). Pricing supplement (SEC Form 424B2), Annex A, an S&P Dow Jones Indices index, https://www.sec.gov/Archives/edgar/data/831001/000095010325012637/dp235271_424b2-us2514099d.htm.
- Dao, T.-L., Nguyen, T.-T., Deremble, C., Lempérière, Y., Bouchaud, J.-P. & Potters, M. (2016). Tail protection for long investors: Trend convexity at work. arXiv:1607.02410.
- Gârleanu, N. & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. Journal of Finance 68(6).
- Harvey, C. R., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O. (2018). The Impact of Volatility Targeting. Journal of Portfolio Management 45(1).
- Hillis, W. D. & Steele, G. L. (1986). Data parallel algorithms. Communications of the ACM 29(12).
- Huang, D., Li, J., Wang, L. & Zhou, G. (2020). Time series momentum: Is it there? Journal of Financial Economics 135(3), 774–794.
- Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A Century of Evidence on Trend-Following Investing. Journal of Portfolio Management 44(1).
- J.P. Morgan / Reuters (1996). RiskMetrics Technical Document.
- Kurth, J. G., Eisler, Z., Rej, A. & Bouchaud, J.-P. (2026). Is Trend Still Your Friend?: A Microstructural Account of the Demise of Short-Term Trend-Following. arXiv:2607.01550.
- Lempérière, Y., Deremble, C., Seager, P., Potters, M. & Bouchaud, J.-P. (2014). Two centuries of trend following. Journal of Investment Strategies 3(3). arXiv:1404.3274.
- Levine, A. & Pedersen, L. H. (2016). Which Trend Is Your Friend? Financial Analysts Journal 72(3).
- Lim, B., Zohren, S. & Roberts, S. (2019). Enhancing Time-Series Momentum Strategies Using Deep Neural Networks. Journal of Financial Data Science 1(4). arXiv:1904.04912.
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal 58(4).
- Morgan Stanley (2025). Pricing supplement (SEC Form 424B2), Annex A: S&P 500 Daily Risk Control 5% USD Excess Return Index, https://www.sec.gov/Archives/edgar/data/895421/000183988225030452/ms8270_424b2-16456.htm.
- Morgan Stanley (2026). Pricing supplement (SEC Form 424B2): Morgan Stanley Amplitude Index, https://www.sec.gov/Archives/edgar/data/895421/000095010326009777/dp249236_424b2-msamp5.htm.
- Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics 104(2).
- Sepp, A. & Lucic, V. (2026). The Science and Practice of Trend-Following Systems. arXiv:2607.19497.
- White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. Econometrica 48(4).
