//! Whole-command snapshots on a made-up Allegro-like cluster, like `tests/snapshots/` of
//! allegro-buddy 2.7: the real `bab` binary (with the `test-hooks` clock, groups and host
//! name) against `tests/fake-slurm/`, a tiny fake Slurm that replays recorded answers.
//!
//! `tests/snapshots/<scenario>.txt` is stdout followed by stderr at `COLUMNS=160`, the
//! scratch directory shown as `@TMP@`. They are byte-exact output of the Python reference
//! (renamed to `bab`), see the `equivalence-harness-3` tag. `UPDATE_SNAPSHOTS=1` rewrites them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

struct Case {
    name: &'static str,
    args: &'static [&'static str],
    /// The user's answer at a (pretended) terminal prompt.
    input: Option<&'static str>,
    exit: i32,
    /// Calls that change Slurm state, in order.
    mutating: &'static [&'static str],
}

const fn case(name: &'static str, args: &'static [&'static str]) -> Case {
    Case {
        name,
        args,
        input: None,
        exit: 0,
        mutating: &[],
    }
}

const CASES: &[Case] = &[
    case("demo-help", &["-h"]),
    case("demo-queue", &["queue"]),
    case("demo-queue-all", &["q", "--all"]),
    case("demo-why", &["why"]),
    case("demo-why-job", &["why", "70120"]),
    case("demo-why-nodes", &["why", "70120", "--nodes"]),
    case("demo-why-fix", &["why", "70122"]),
    Case {
        input: Some("n\n"),
        ..case("demo-why-fix-confirm", &["why", "70122", "--fix"])
    },
    case("demo-node", &["node", "gpu102"]),
    case("demo-node-shared", &["node", "gpu201"]),
    case("demo-node-cpu", &["node", "cmp201"]),
    case("demo-job-history", &["job", "history"]),
    case("demo-job-output", &["job", "output", "70090"]),
    case("demo-dev-status", &["dev", "status"]),
    Case {
        mutating: &["sbatch --parsable --job-name=ab-dev --partition=gpu"],
        ..case("demo-dev-start", &["dev", "start", "--no-wait"])
    },
    Case {
        mutating: &["scancel 70110"],
        ..case("demo-dev-stop", &["dev", "stop"])
    },
    case("demo-config-show", &["config", "show"]),
    Case {
        exit: 1,
        ..case("demo-error-unknown-job", &["why", "999999"])
    },
    case("plain-info", &["info"]),
    case("plain-free", &["free"]),
    case("plain-free-nodes", &["free", "--nodes"]),
    case("plain-free-cpu", &["free", "--kind", "cpu", "--nodes"]),
    case("plain-free-a5000", &["free", "-G", "a5000", "--nodes"]),
];

const SLURM_COMMANDS: &[&str] = &[
    "scontrol", "squeue", "sinfo", "sacct", "sacctmgr", "sbatch", "scancel",
];

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Runs one case in a fresh scratch dir; returns (output, exit code, Slurm call log).
fn run(case: &Case) -> (String, i32, Vec<String>) {
    let fake = tests_dir().join("fake-slurm");
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("snap-{}", case.name));
    let _ = std::fs::remove_dir_all(&tmp);
    for dir in ["bin", "home", "config", "cache", "site", "demo"] {
        std::fs::create_dir_all(tmp.join(dir)).unwrap();
    }
    for command in SLURM_COMMANDS {
        std::os::unix::fs::symlink(fake.join("slurm"), tmp.join("bin").join(command)).unwrap();
    }
    std::fs::copy(
        fake.join("slurm-70090.out"),
        tmp.join("demo/slurm-70090.out"),
    )
    .unwrap();
    let log = tmp.join("slurm-calls");
    std::fs::write(&log, "").unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_bab"))
        .args(case.args)
        .current_dir(tmp.join("home"))
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", tmp.join("bin").display()),
        )
        .env("HOME", tmp.join("home"))
        .env("USER", "alice")
        .env("LOGNAME", "alice")
        .env("TZ", "UTC")
        .env("LANG", "C.UTF-8")
        .env("COLUMNS", "160")
        .env("XDG_CONFIG_HOME", tmp.join("config"))
        .env("XDG_CACHE_HOME", tmp.join("cache"))
        .env("AB_SITE_CONFIG", tmp.join("site/config.ini"))
        .env("BAB_TEST_NOW", "2026-11-03T10:00:00")
        .env("BAB_TEST_GROUPS", "users")
        .env("BAB_TEST_HOSTNAME", "login1")
        .env(
            "BAB_TEST_INTERACTIVE",
            if case.input.is_some() { "1" } else { "0" },
        )
        .env("FAKE_SLURM_DIR", &fake)
        .env("FAKE_SLURM_SCENARIO", case.name)
        .env("FAKE_SLURM_TMP", &tmp)
        .env("FAKE_SLURM_LOG", &log)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin
        .write_all(case.input.unwrap_or("").as_bytes())
        .unwrap();
    drop(stdin);
    let output = child.wait_with_output().unwrap();

    let text = String::from_utf8([output.stdout, output.stderr].concat()).unwrap();
    let text = text.replace(tmp.to_str().unwrap(), "@TMP@");
    let calls = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    (text, output.status.code().unwrap_or(-1), calls)
}

#[test]
fn snapshots() {
    let update = std::env::var_os("UPDATE_SNAPSHOTS").is_some();
    let mut failed = Vec::new();
    for case in CASES {
        let (output, exit, calls) = run(case);
        let path = tests_dir()
            .join("snapshots")
            .join(format!("{}.txt", case.name));
        if update {
            std::fs::write(&path, &output).unwrap();
        }
        let expected = std::fs::read_to_string(&path).unwrap_or_default();
        let unexpected: Vec<_> = calls
            .iter()
            .filter(|c| c.starts_with("unexpected"))
            .collect();
        let mutating: Vec<_> = calls
            .iter()
            .filter(|c| {
                c.starts_with("sbatch") || c.starts_with("scancel") || c.contains(" update ")
            })
            .collect();
        let mutating_ok = mutating.len() == case.mutating.len()
            && mutating
                .iter()
                .zip(case.mutating)
                .all(|(c, want)| c.starts_with(want));
        if output != expected || exit != case.exit || !unexpected.is_empty() || !mutating_ok {
            failed.push(case.name);
            eprintln!(
                "--- {} (exit {exit}, want {}; Slurm calls {calls:?})\n--- expected:\n{expected}--- got:\n{output}",
                case.name, case.exit
            );
        }
    }
    assert!(failed.is_empty(), "snapshots differ: {failed:?}");
}
