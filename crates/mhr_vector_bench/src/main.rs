//! `mhr_vector_bench`: datasets, ground truth, measurements and the speed score.

use mhr_vector_bench::args::Args;
use mhr_vector_bench::commands;

const USAGE: &str = "\
usage: mhr_vector_bench <command> [options]   (JSON on stdout)

  datasets  [--datasets ROOT]                           list converted datasets
  synth     --out DIR [--dim D --n N --queries Q --shape clustered|gaussian|duplicates|anisotropic
            --metric cosine|dot|l2 --seed S]            write a synthetic dataset
  gt        --dataset NAME [--base N --queries Q --filters a,b --threads T]
                                                        compute and cache ground truth
  build     --dataset NAME --index DIR [--config NAME|JSON --base N --rows N --threads T
            --budget BYTES --batch N --commit-every N --delta N --engine mhr|oracle]
  query     --dataset NAME --index DIR [--k 10 --beams 16,32 --qps-threads 1,8 --filters a,b
            --repeats 3 --recall-threads 4]
  open      --dataset NAME --index DIR [--drop-caches]  open in this process, one query
  update    --dataset NAME --index DIR --rows N [...]  insert rows N.., delete 10%, recall
  run       --suite FILE --out DIR [--work DIR --keep]  a whole suite: results.json + results.md
  prepare   --workload FILE --index-root DIR            ground truth + builds for `score`
  score     --workload FILE --index-root DIR            {\"primary\", \"components\", \"detail\"}
  build-score --workload FILE --index-root DIR          the build phase's number (vectors/s)

Filters (--filters, workload \"filters\"): the standard set (none, p50, p10, p01, p001, go,
p10-python, p01-or-p001, p50-d7, nothing), or path:<prefix>, or lang:<language>.

Common: --datasets ROOT (default $MHRV_DATASETS or ./datasets), --engine mhr|oracle.";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = raw.first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let args = match Args::parse(&raw[1..], &["drop-caches", "keep"]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let result = match command.as_str() {
        "datasets" => commands::datasets(&args),
        "synth" => commands::synth(&args),
        "gt" => commands::gt(&args),
        "build" => commands::build(&args),
        "query" => commands::query(&args),
        "open" => commands::open(&args),
        "update" => commands::update(&args),
        "run" => commands::run(&args),
        "prepare" => commands::prepare(&args),
        "score" => commands::score(&args),
        "build-score" => commands::build_score(&args),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            return;
        }
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    };
    match result {
        Ok(v) => println!("{}", serde_json::to_string(&v).unwrap()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
