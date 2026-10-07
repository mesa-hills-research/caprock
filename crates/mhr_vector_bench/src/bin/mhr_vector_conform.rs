//! `mhr_vector_conform`: the conformance runner (the swarm's oracle).
//!
//! ```text
//! mhr_vector_conform list
//! mhr_vector_conform run [--jobs N] [--timeout SECS] [--filter SUBSTRING] [--out FILE]
//! mhr_vector_conform one <check id>
//! ```
//!
//! `run` starts one child process per check (a crash or hang is pinned on its check),
//! `--jobs` at a time, and prints one JSON line per check:
//! `{"check", "status": pass|fail|panic|timeout|error, "signature", "detail", "seconds"}`,
//! then a summary line. The signature is what the swarm buckets by:
//! - fail: `<check id>: <failure kind>` (kinds are stable: `recall-below`,
//!   `exact-mismatch`, `too-few`, `live-count`, `bytes-differ`, ...);
//! - panic: `todo: <unit>` for an unimplemented stub (`todo!("U07 ...")`), otherwise
//!   `panic: <file>: <message up to its first digit>`;
//! - timeout: `timeout: <check id>`.
//!
//! Exit status: 0 when every selected check passed, 1 otherwise.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mhr_vector_bench::args::Args;
use mhr_vector_bench::checks;
use serde_json::{Value, json};

static PANIC: Mutex<Option<(String, String)>> = Mutex::new(None);

fn panic_signature(message: &str, file: &str) -> String {
    if let Some(rest) = message.strip_prefix("not yet implemented: ")
        && let Some(unit) = rest.split_whitespace().next()
        && unit.starts_with('U')
    {
        return format!("todo: {unit}");
    }
    let stem: String = message
        .chars()
        .take_while(|c| !c.is_ascii_digit())
        .take(80)
        .collect();
    format!("panic: {file}: {}", stem.trim())
}

fn one(id: &str) -> Value {
    let Some(check) = checks::all().into_iter().find(|c| c.id == id) else {
        return json!({"check": id, "status": "error", "signature": "unknown-check", "detail": "no such check"});
    };
    std::panic::set_hook(Box::new(|info| {
        let message = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_default();
        let file = info
            .location()
            .map(|l| l.file().to_string())
            .unwrap_or_default();
        let mut slot = PANIC.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some((message, file));
        }
    }));
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (check.run)()));
    let seconds = started.elapsed().as_secs_f64();
    match outcome {
        Ok(Ok(summary)) => {
            json!({"check": id, "status": "pass", "signature": null, "detail": summary, "seconds": seconds})
        }
        Ok(Err(f)) => json!({
            "check": id, "status": "fail", "signature": format!("{id}: {}", f.kind),
            "detail": f.detail, "seconds": seconds,
        }),
        Err(_) => {
            let (message, file) = PANIC
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .clone()
                .unwrap_or_default();
            json!({
                "check": id, "status": "panic", "signature": panic_signature(&message, &file),
                "detail": format!("{file}: {message}"), "seconds": seconds,
            })
        }
    }
}

struct Running {
    id: String,
    child: Child,
    started: Instant,
}

fn finish(r: Running, timed_out: bool) -> Value {
    let seconds = r.started.elapsed().as_secs_f64();
    if timed_out {
        let mut child = r.child;
        let _ = child.kill();
        let _ = child.wait();
        return json!({"check": r.id, "status": "timeout", "signature": format!("timeout: {}", r.id), "detail": format!("killed after {seconds:.0} s"), "seconds": seconds});
    }
    match r.child.wait_with_output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            match stdout
                .lines()
                .rev()
                .find_map(|l| serde_json::from_str::<Value>(l).ok())
            {
                Some(v) => v,
                None => json!({
                    "check": r.id, "status": "panic", "signature": format!("crash: {}", r.id),
                    "detail": format!("{}: {}", out.status, String::from_utf8_lossy(&out.stderr).lines().rev().take(5).collect::<Vec<_>>().join(" | ")),
                    "seconds": seconds,
                }),
            }
        }
        Err(e) => {
            json!({"check": r.id, "status": "error", "signature": "runner", "detail": e.to_string(), "seconds": seconds})
        }
    }
}

fn run(a: &Args) -> i32 {
    let jobs: usize = a.num("jobs", 4).unwrap_or(4);
    let timeout = Duration::from_secs(a.num("timeout", 600u64).unwrap_or(600));
    let filter = a.get("filter").unwrap_or("").to_string();
    let ids: Vec<String> = checks::all()
        .into_iter()
        .map(|c| c.id)
        .filter(|id| id.contains(&filter))
        .collect();
    let exe = std::env::current_exe().expect("own path");
    let mut out: Box<dyn Write> = match a.get("out") {
        Some(p) => Box::new(std::fs::File::create(p).expect("output file")),
        None => Box::new(std::io::stdout()),
    };
    let mut queue = ids.into_iter();
    let mut running: Vec<Running> = Vec::new();
    let (mut pass, mut total) = (0, 0);
    loop {
        while running.len() < jobs.max(1) {
            let Some(id) = queue.next() else { break };
            let child = Command::new(&exe)
                .args(["one", &id])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("spawn a check");
            running.push(Running {
                id,
                child,
                started: Instant::now(),
            });
        }
        if running.is_empty() {
            break;
        }
        let mut i = 0;
        while i < running.len() {
            let done = matches!(running[i].child.try_wait(), Ok(Some(_)));
            let late = running[i].started.elapsed() > timeout;
            if done || late {
                let r = running.swap_remove(i);
                let v = finish(r, late && !done);
                total += 1;
                pass += (v["status"] == "pass") as usize;
                writeln!(out, "{}", serde_json::to_string(&v).unwrap()).ok();
                out.flush().ok();
            } else {
                i += 1;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    writeln!(
        out,
        "{}",
        json!({"summary": {"passed": pass, "total": total}})
    )
    .ok();
    if pass == total { 0 } else { 1 }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = Args::parse(raw.get(1..).unwrap_or(&[]), &[]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let code = match raw.first().map(String::as_str) {
        Some("list") => {
            for c in checks::all() {
                println!("{}", c.id);
            }
            0
        }
        Some("one") => {
            let Some(id) = args.positional.first() else {
                eprintln!("usage: mhr_vector_conform one <check id>");
                std::process::exit(2);
            };
            let v = one(id);
            let ok = v["status"] == "pass";
            println!("{}", serde_json::to_string(&v).unwrap());
            if ok { 0 } else { 1 }
        }
        Some("run") => run(&args),
        _ => {
            eprintln!(
                "usage: mhr_vector_conform list | run [--jobs N --timeout S --filter X --out FILE] | one <id>"
            );
            2
        }
    };
    std::process::exit(code);
}
