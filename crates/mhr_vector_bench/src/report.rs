//! Markdown tables of suite results.

use serde_json::Value;

fn f(v: &Value, digits: usize) -> String {
    match v.as_f64() {
        Some(x) => format!("{x:.digits$}"),
        None => "-".into(),
    }
}

fn mb(v: &Value) -> String {
    match v.as_f64() {
        Some(x) => format!("{:.0}", x / 1048576.0),
        None => "-".into(),
    }
}

/// A markdown report of `run`'s results.
pub fn markdown(report: &Value) -> String {
    let mut s = format!("# {}\n\n", report["suite"].as_str().unwrap_or("suite"));
    s.push_str("## Builds\n\n| dataset | base | config | build s | vectors/s | peak RSS MB | disk MB | segments | open warm s | open cold s |\n|---|---|---|---|---|---|---|---|---|---|\n");
    let parts = report["parts"].as_array().cloned().unwrap_or_default();
    for p in &parts {
        let b = &p["build"];
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p["dataset"].as_str().unwrap_or("?"),
            p["base"],
            p["config"].as_str().unwrap_or("?"),
            f(&b["build_seconds"], 1),
            f(&b["vectors_per_second"], 0),
            mb(&b["peak_rss"]),
            mb(&p["disk_bytes"]),
            b["stats"]["segments"],
            f(&p["open_warm"]["open_seconds"], 3),
            f(&p["open_cold"]["open_seconds"], 3),
        ));
        if let Some(e) = p["error"].as_str() {
            s.push_str(&format!("\nError: `{}`\n\n", e.replace('`', "'")));
        }
    }
    s.push_str("\n## Queries (single thread unless noted)\n\n| dataset | config | k | beam | recall | QPS | p50 ms | p99 ms | QPS (N threads) | violations |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for p in &parts {
        for r in p["queries"]["beams"]
            .as_array()
            .cloned()
            .unwrap_or_default()
        {
            let multi = r["qps_threads"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .map(|(t, q)| format!("{t}: {}", f(q, 0)))
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p["dataset"].as_str().unwrap_or("?"),
                p["config"].as_str().unwrap_or("?"),
                p["k"],
                if r["beam"].is_null() {
                    "default".into()
                } else {
                    r["beam"].to_string()
                },
                f(&r["recall"], 4),
                f(&r["qps_1"], 0),
                f(&r["p50_ms"], 3),
                f(&r["p99_ms"], 3),
                multi,
                r["violations"],
            ));
        }
    }
    s.push_str("\n## Filtered queries (default beam)\n\n| dataset | config | filter | matching | recall | QPS | p50 ms | p99 ms | violations |\n|---|---|---|---|---|---|---|---|---|\n");
    for p in &parts {
        for r in p["queries"]["filters"]
            .as_array()
            .cloned()
            .unwrap_or_default()
        {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p["dataset"].as_str().unwrap_or("?"),
                p["config"].as_str().unwrap_or("?"),
                r["filter"].as_str().unwrap_or("?"),
                r["matching"],
                f(&r["recall"], 4),
                f(&r["qps_1"], 0),
                f(&r["p50_ms"], 3),
                f(&r["p99_ms"], 3),
                r["violations"],
            ));
        }
    }
    s.push_str("\n## Updates\n\n| dataset | config | inserts/s | apply(100) p50 ms | p99 ms | commit s | deletes/s | recall after | after maintain | after compact | violations |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for p in &parts {
        let u = &p["updates"];
        if u.is_null() {
            continue;
        }
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            p["dataset"].as_str().unwrap_or("?"),
            p["config"].as_str().unwrap_or("?"),
            f(&u["insert_vectors_per_second"], 0),
            f(&u["apply_100_p50_ms"], 2),
            f(&u["apply_100_p99_ms"], 2),
            f(&u["commit_seconds"], 3),
            f(&u["deletes_per_second"], 0),
            f(&u["recall_after_updates"], 4),
            f(&u["recall_after_maintain"], 4),
            f(&u["recall_after_compact"], 4),
            u["violations_after_updates"],
        ));
        if let Some(e) = u["error"].as_str() {
            s.push_str(&format!("\nError: `{}`\n\n", e.replace('`', "'")));
        }
    }
    s
}
