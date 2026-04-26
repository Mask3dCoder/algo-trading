use pyo3::prelude::*;
use std::collections::BTreeMap;

#[pyclass]
#[derive(Clone)]
pub struct OrderBook {
    #[pyo3(get)]
    symbol: String,
    bids: BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
    asks: BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
}

// Placeholder — full implementation in T-0147
