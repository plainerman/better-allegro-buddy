//! `--watch` / `--poll-interval` values click accepts but `time.sleep` cannot take (nan,
//! inf, >= 2**63 ns): Python crashes with a traceback (`dev start` only after cancelling
//! and submitting); bab must reject them up front with a usage error (exit 2) and never
//! touch Slurm state.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const FAKE: &str = r#"#!/bin/sh
echo "$(basename "$0") $*" >> "$CALLS"
case "$(basename "$0") $*" in
  squeue*) echo 111 ;;
  *" partition"*) echo "PartitionName=gpu AllowGroups=ALL MaxTime=UNLIMITED" ;;
  *" show job "*) echo "JobId=111 JobName=ab-dev UserId=alice(1000) JobState=PENDING Reason=Priority Partition=gpu NodeList=(null) SubmitTime=2030-01-01T08:00:00" ;;
  sbatch*) echo 222 ;;
esac
"#;

/// A PATH dir of Slurm commands that log their argv (one existing, pending dev job).
fn fake_slurm(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("bin")).unwrap();
    std::fs::create_dir_all(root.join("config/allegro-buddy")).unwrap();
    std::fs::write(
        root.join("config/allegro-buddy/config.ini"),
        format!(
            "[dev]\nlog_dir = {}\n[cluster]\ngpu_partitions = gpu\n",
            root.join("logs").display()
        ),
    )
    .unwrap();
    for command in [
        "squeue", "scontrol", "sinfo", "sacct", "sbatch", "scancel", "ssh", "srun",
    ] {
        let path = root.join("bin").join(command);
        std::fs::write(&path, FAKE).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    root
}

/// Run bab; returns (exit code, stderr, logged Slurm calls).
fn bab(name: &str, args: &[&str]) -> (i32, String, Vec<String>) {
    let root = fake_slurm(name);
    let output = Command::new(env!("CARGO_BIN_EXE_bab"))
        .args(args)
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", root.join("bin").display()),
        )
        .env("CALLS", root.join("calls"))
        .env("HOME", &root)
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("AB_CLUSTER", "allegro")
        .env("USER", "alice")
        .env("LANG", "C.UTF-8")
        .env("COLUMNS", "200")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let calls = std::fs::read_to_string(root.join("calls")).unwrap_or_default();
    (
        output.status.code().unwrap(),
        String::from_utf8(output.stderr).unwrap(),
        calls.lines().map(str::to_owned).collect(),
    )
}

fn assert_rejected(name: &str, args: &[&str], option: &str, value: &str) {
    let (code, stderr, calls) = bab(name, args);
    assert_eq!(code, 2, "{args:?}: {stderr}");
    assert!(
        stderr.contains(&format!(
            "Invalid value for {option}: {value} is not in the range"
        )),
        "{args:?}: {stderr}"
    );
    assert!(stderr.contains("Try 'bab "), "{args:?}: {stderr}");
    for call in &calls {
        for mutating in ["sbatch", "scancel", "ssh", "srun"] {
            assert!(!call.starts_with(mutating), "{args:?} ran {call}");
        }
    }
    // no tight polling loop: at most the lookups done before the check
    assert!(calls.len() < 10, "{args:?}: {calls:?}");
}

#[test]
fn watch_rejects_unusable_intervals() {
    for (command, value) in [
        ("q", "nan"),
        ("q", "inf"),
        ("free", "nan"),
        ("free", "1e300"),
    ] {
        assert_rejected(
            &format!("sleep-{command}-{value}"),
            &[command, "-w", value],
            "'-w' / '--watch'",
            if value == "1e300" { "1e+300" } else { value },
        );
    }
}

#[test]
fn dev_start_rejects_before_cancel_or_submit() {
    for value in ["nan", "inf"] {
        for extra in [&["--replace"][..], &["--replace", "--login"]] {
            let mut args = vec!["dev", "start", "--poll-interval", value];
            args.extend(extra);
            assert_rejected(
                &format!("sleep-start-{value}-{}", extra.len()),
                &args,
                "'--poll-interval'",
                value,
            );
        }
    }
}

#[test]
fn dev_login_wait_rejects_before_polling() {
    for args in [
        &["dev", "login", "--wait", "--poll-interval", "nan"][..],
        &["l", "--wait", "--poll-interval", "inf"],
    ] {
        assert_rejected(
            &format!("sleep-login-{}", args[0]),
            args,
            "'--poll-interval'",
            args[args.len() - 1],
        );
    }
}

#[test]
fn no_wait_ignores_the_interval_like_python() {
    let (code, stderr, calls) = bab(
        "sleep-no-wait",
        &[
            "dev",
            "start",
            "--replace",
            "--no-wait",
            "--poll-interval",
            "nan",
        ],
    );
    assert_eq!(code, 0, "{stderr}");
    assert!(
        calls.iter().any(|call| call.starts_with("sbatch")),
        "{calls:?}"
    );
}
