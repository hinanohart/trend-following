//! Golden vectors from `testdata/generate.py` (independent of both implementations), plus properties.

use std::collections::HashMap;
use std::{iter, panic};

use trend_following::{
    Config, Ewma, Ewmac, EwmacTstat, TrendFollowing, ewm_vol, ewma, ewmac, ewmac_tstat, sharpe,
    strategy_returns, tsmom,
};

const CASES: [(&str, &str); 4] = [
    ("iid", include_str!("../../testdata/iid.csv")),
    ("trend", include_str!("../../testdata/trend.csv")),
    ("zeros", include_str!("../../testdata/zeros.csv")),
    ("gaps", include_str!("../../testdata/gaps.csv")),
];

const SHARPE: &str = include_str!("../../testdata/sharpe.csv");

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split(',').collect())
}

fn columns(csv: &str) -> HashMap<String, Vec<f64>> {
    let mut rows = rows(csv);
    let header = rows.next().expect("header");
    let values: Vec<Vec<f64>> = rows
        .map(|row| row.iter().map(|v| v.parse().expect("float")).collect())
        .collect();

    header
        .iter()
        .enumerate()
        .map(|(i, &name)| (name.to_owned(), values.iter().map(|row| row[i]).collect()))
        .collect()
}

fn run(returns: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut model = TrendFollowing::new(&Config::default());
    returns
        .iter()
        .map(|&r| (model.update(r), model.intermediates().signal))
        .unzip()
}

fn positions(returns: &[f64]) -> Vec<f64> {
    run(returns).0
}

fn ours(returns: &[f64]) -> HashMap<&'static str, Vec<f64>> {
    let (pos, signal) = run(returns);
    let r = || returns.iter().copied();

    HashMap::from([
        ("ewma", ewma(r(), 10.0).collect()),
        ("ewm_vol", ewm_vol(r(), 35.0).collect()),
        ("ewmac", ewmac(r(), 16.0, 64.0).collect()),
        ("tsmom", tsmom(r(), 20).collect()),
        (
            "ret_lag1",
            strategy_returns(&pos, returns, 0.0, 1).collect(),
        ),
        (
            "ret_lag2_cost",
            strategy_returns(&pos, returns, 2e-4, 2).collect(),
        ),
        ("position", pos),
        ("signal", signal),
    ])
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 + 1e-9 * b.abs()
}

fn returns(seed: u64, len: usize) -> Vec<f64> {
    // splitmix64 -> uniform in [-0.01, 0.01)
    iter::successors(Some(seed), |s| Some(s.wrapping_add(0x9E37_79B9_7F4A_7C15)))
        .skip(1)
        .take(len)
        .map(|s| {
            let z = (s ^ (s >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let unit = f64::from(u32::try_from((z ^ (z >> 31)) >> 32).expect("32 bits"))
                / f64::from(u32::MAX);
            (unit - 0.5) * 0.02
        })
        .collect()
}

#[test]
fn golden_vectors() {
    for (case, csv) in CASES {
        let golden = columns(csv);
        for (name, values) in ours(&golden["r"]) {
            let expected = &golden[name];
            assert_eq!(values.len(), expected.len(), "{case}/{name}: length");
            if let Some(t) = values
                .iter()
                .zip(expected)
                .position(|(&a, &b)| !close(a, b))
            {
                panic!(
                    "{case}/{name} at t={t}: {} vs golden {}",
                    values[t], expected[t]
                );
            }
        }
    }
}

#[test]
fn golden_sharpe() {
    let expected: HashMap<&str, Vec<f64>> = rows(SHARPE)
        .skip(1)
        .map(|row| {
            (
                row[0],
                row[1..].iter().map(|v| v.parse().expect("float")).collect(),
            )
        })
        .collect();

    for (case, csv) in CASES {
        let ours = ours(&columns(csv)["r"]);
        let got = [
            sharpe(&ours["ret_lag1"], 252.0),
            sharpe(&ours["ret_lag2_cost"], 252.0),
        ];
        assert!(
            got.iter().zip(&expected[case]).all(|(&a, &b)| close(a, b)),
            "{case}: {got:?} vs {:?}",
            expected[case]
        );
    }
}

#[test]
fn streaming_equals_batch() {
    let r = returns(1, 500);
    let mut state = Ewmac::new(16.0, 64.0);
    let streamed: Vec<u64> = r.iter().map(|&x| state.update(x).to_bits()).collect();
    let batched: Vec<u64> = ewmac(r.iter().copied(), 16.0, 64.0)
        .map(f64::to_bits)
        .collect();
    assert_eq!(streamed, batched);

    let speeds = Config::default().speeds;
    let mut state = EwmacTstat::new(&speeds);
    let streamed: Vec<u64> = r.iter().map(|&x| state.update(x).to_bits()).collect();
    let batched: Vec<u64> = ewmac_tstat(r.iter().copied(), &speeds)
        .map(f64::to_bits)
        .collect();
    assert_eq!(streamed, batched);
}

#[test]
fn a_lone_return_moves_the_tstat_by_one() {
    let mut r = vec![0.0; 60];
    r[..5].iter_mut().for_each(|x| *x = f64::NAN);
    r[10] = -1e3;

    let z: Vec<f64> = ewmac_tstat(r, &Config::default().speeds).collect();
    assert!(z[..10].iter().all(|&x| x == 0.0));
    assert!(z[10..].iter().all(|&x| (x + 1.0).abs() < 1e-12));
}

#[test]
fn the_future_never_changes_the_past() {
    let r = returns(2, 400);
    let mut changed = r.clone();
    changed[200..].iter_mut().for_each(|x| *x = -*x * 3.0);

    let bits = |xs: &[f64]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (pos, changed_pos) = (positions(&r), positions(&changed));
    assert_eq!(bits(&pos[..200]), bits(&changed_pos[..200]));

    let pnl: Vec<f64> = strategy_returns(&pos, &r, 2e-4, 1).collect();
    let mut moved = pos.clone();
    moved[200..].iter_mut().for_each(|p| *p += 1.0);
    let moved_pnl: Vec<f64> = strategy_returns(&moved, &r, 2e-4, 1).collect();
    assert_eq!(bits(&pnl[..=200]), bits(&moved_pnl[..=200]));
}

#[test]
fn missing_data_and_flat_markets_stay_finite() {
    let mut r = returns(3, 300);
    r[..50].iter_mut().for_each(|x| *x = f64::NAN);
    r[120] = f64::INFINITY;

    let pos = positions(&r);
    assert!(pos.iter().all(|p| p.is_finite()));
    assert!(
        pos[..110].iter().all(|&p| p == 0.0),
        "no position before warm-up on valid data"
    );
    assert!(positions(&[0.0; 300]).iter().all(|&p| p == 0.0));
}

#[test]
fn leading_missing_returns_change_nothing() {
    let r = columns(CASES[0].1)["r"].clone();
    let padded: Vec<f64> = iter::repeat_n(f64::NAN, 300)
        .chain(r.iter().copied())
        .collect();
    assert_eq!(positions(&padded)[300..], positions(&r)[..]);
}

#[test]
#[should_panic(expected = "lag must be at least 1")]
fn lag_zero_is_refused() {
    let _ = strategy_returns(&[1.0], &[0.01], 0.0, 0);
}

#[test]
fn spans_below_one_or_not_finite_are_refused() {
    let refused = |span: f64| {
        [
            panic::catch_unwind(|| Ewma::new(span)).is_err(),
            panic::catch_unwind(|| Ewmac::new(span, 64.0)).is_err(),
            panic::catch_unwind(|| EwmacTstat::new(&[(8.0, span)])).is_err(),
            panic::catch_unwind(|| {
                TrendFollowing::new(&Config {
                    vol_span: span,
                    ..Config::default()
                })
            })
            .is_err(),
        ]
    };
    let invalid = [0.5, -1.0, -3.0, f64::NAN, f64::INFINITY];
    assert!(invalid.into_iter().all(|span| refused(span) == [true; 4]));
    assert_eq!(
        refused(1.0),
        [false; 4],
        "span 1 is beta 0: price minus its average"
    );
}
