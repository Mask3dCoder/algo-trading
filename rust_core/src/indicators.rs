use pyo3::prelude::*;

/// Simple Moving Average
#[pyfunction]
fn sma(data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    if period == 0 || data.len() < period {
        return Ok(vec![None; data.len()]);
    }
    let mut result = vec![None; data.len()];
    let mut sum = 0.0;
    for i in 0..data.len() {
        sum += data[i];
        if i >= period {
            sum -= data[i - period];
        }
        if i >= period - 1 {
            result[i] = Some(sum / period as f64);
        }
    }
    Ok(result)
}

/// Exponential Moving Average
#[pyfunction]
fn ema(data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    if period == 0 || data.is_empty() {
        return Ok(vec![None; data.len()]);
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut result = vec![None; data.len()];
    let mut prev: Option<f64> = None;
    for i in 0..data.len() {
        if i < period - 1 {
            continue;
        }
        match prev {
            None => {
                let sum: f64 = data[..period].iter().sum();
                prev = Some(sum / period as f64);
            }
            Some(p) => {
                prev = Some(data[i] * k + p * (1.0 - k));
            }
        }
        result[i] = prev;
    }
    Ok(result)
}

/// Relative Strength Index
#[pyfunction]
fn rsi(data: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    if period == 0 || data.len() < period + 1 {
        return Ok(vec![None; data.len()]);
    }
    let mut result = vec![None; data.len()];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;
    for i in 1..=period {
        let change = data[i] - data[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;
    if avg_loss == 0.0 {
        result[period] = Some(100.0);
    } else {
        let rs = avg_gain / avg_loss;
        result[period] = Some(100.0 - 100.0 / (1.0 + rs));
    }
    for i in (period + 1)..data.len() {
        let change = data[i] - data[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { change.abs() } else { 0.0 };
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        if avg_loss == 0.0 {
            result[i] = Some(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            result[i] = Some(100.0 - 100.0 / (1.0 + rs));
        }
    }
    Ok(result)
}

/// Average True Range
#[pyfunction]
fn atr(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> PyResult<Vec<Option<f64>>> {
    let n = high.len().min(low.len()).min(close.len());
    if period == 0 || n < 2 {
        return Ok(vec![None; n]);
    }
    let mut tr = vec![0.0; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    let mut result = vec![None; n];
    let mut sum: f64 = tr[..period].iter().sum();
    result[period - 1] = Some(sum / period as f64);
    for i in period..n {
        let prev = result[i - 1].unwrap();
        result[i] = Some((prev * (period as f64 - 1.0) + tr[i]) / period as f64);
    }
    Ok(result)
}
