//! A small argument parser: `command --key value --flag positional`.

use std::collections::BTreeMap;

/// Parsed arguments: options (`--key value` or `--key=value`), flags (`--key` alone,
/// for names in `flags`), and positionals.
#[derive(Debug, Default)]
pub struct Args {
    pub options: BTreeMap<String, String>,
    pub flags: Vec<String>,
    pub positional: Vec<String>,
}

impl Args {
    /// Parses `raw`; names listed in `flags` take no value.
    pub fn parse(raw: &[String], flags: &[&str]) -> Result<Args, String> {
        let mut out = Args::default();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    out.options.insert(k.to_string(), v.to_string());
                } else if flags.contains(&name) {
                    out.flags.push(name.to_string());
                } else {
                    let v = raw
                        .get(i + 1)
                        .ok_or_else(|| format!("--{name} needs a value"))?;
                    out.options.insert(name.to_string(), v.clone());
                    i += 1;
                }
            } else {
                out.positional.push(a.clone());
            }
            i += 1;
        }
        Ok(out)
    }

    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.options.get(name).map(String::as_str)
    }

    pub fn required(&self, name: &str) -> Result<&str, String> {
        self.get(name)
            .ok_or_else(|| format!("--{name} is required"))
    }

    pub fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{name}: cannot parse {v:?}")),
        }
    }

    /// A comma-separated list of numbers.
    pub fn list<T: std::str::FromStr + Clone>(
        &self,
        name: &str,
        default: &[T],
    ) -> Result<Vec<T>, String> {
        match self.get(name) {
            None => Ok(default.to_vec()),
            Some(v) => v
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.trim()
                        .parse()
                        .map_err(|_| format!("--{name}: cannot parse {s:?}"))
                })
                .collect(),
        }
    }
}
