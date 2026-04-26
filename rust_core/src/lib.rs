use pyo3::prelude::*;

mod orderbook;
mod indicators;
mod matching;

#[pymodule]
fn quantcore_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<orderbook::OrderBook>()?;
    m.add_function(wrap_pyfunction!(indicators::sma, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::ema, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::rsi, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::atr, m)?)?;
    Ok(())
}
