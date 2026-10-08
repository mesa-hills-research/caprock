//! Measurement helpers: percentiles, memory from /proc, directory sizes, page-cache
//! dropping.

use std::path::Path;

/// The `q`-quantile (0..=1) of `values` (nearest rank on a sorted copy).
pub fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len());
    v[rank - 1]
}

/// The median (lower middle).
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[(v.len() - 1) / 2]
}

/// A field of `/proc/self/status` in bytes (`VmHWM`: peak resident set; `VmRSS`:
/// current). `None` off Linux.
pub fn proc_status_bytes(field: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field)
            && let Some(rest) = rest.strip_prefix(':')
        {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Peak resident memory of this process so far.
pub fn peak_rss() -> Option<u64> {
    proc_status_bytes("VmHWM")
}

/// Current resident memory.
pub fn current_rss() -> Option<u64> {
    proc_status_bytes("VmRSS")
}

/// Total size of the regular files under `dir` (recursively).
pub fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_dir() {
                    total += dir_bytes(&e.path());
                } else {
                    total += m.len();
                }
            }
        }
    }
    total
}

/// Drops the page cache (needs root: writes 3 to /proc/sys/vm/drop_caches after a
/// sync). Returns whether it worked.
pub fn drop_caches() -> bool {
    let _ = std::process::Command::new("sync").status();
    std::fs::write("/proc/sys/vm/drop_caches", "3\n").is_ok()
}

/// Seconds since `start`.
pub fn secs(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles() {
        let v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(quantile(&v, 0.5), 50.0);
        assert_eq!(quantile(&v, 0.99), 99.0);
        assert_eq!(quantile(&v, 1.0), 100.0);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(quantile(&[], 0.5), 0.0);
    }
}
