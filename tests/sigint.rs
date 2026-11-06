//! Ctrl-C (SIGINT to bab alone, not to its process group) while Slurm hangs: bab must
//! stop waiting at once, kill and reap the child, and exit like Python (`Aborted.`,
//! exit 1; `--watch` swallows the interrupt, exit 0).

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A PATH dir whose Slurm commands log their pid and hang.
fn hanging_slurm(name: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("bin")).unwrap();
    for command in ["squeue", "scontrol", "sinfo", "sacct"] {
        let path = root.join("bin").join(command);
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\necho $$ >> '{}'\nexec sleep 30\n",
                root.join("pids").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    root
}

#[cfg(target_os = "linux")]
fn alive(pid: i32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        stat.rsplit(") ")
            .next()
            .is_some_and(|s| !s.starts_with('Z'))
    })
}

/// No /proc on macOS: ask ps(1) for the state (empty when the pid is gone).
#[cfg(not(target_os = "linux"))]
fn alive(pid: i32) -> bool {
    Command::new("ps")
        .args(["-o", "stat=", "-p", &pid.to_string()])
        .output()
        .is_ok_and(|out| {
            let stat = String::from_utf8_lossy(&out.stdout);
            let stat = stat.trim();
            !stat.is_empty() && !stat.starts_with('Z')
        })
}

fn interrupt(name: &str, args: &[&str]) -> (i32, String, Duration) {
    let root = hanging_slurm(name);
    let child = Command::new(env!("CARGO_BIN_EXE_bab"))
        .args(args)
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", root.join("bin").display()),
        )
        .env("HOME", &root)
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("AB_CLUSTER", "allegro")
        .env("USER", "alice")
        .env("LANG", "C.UTF-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pids = root.join("pids");
    let started = Instant::now();
    while !pids.exists() {
        assert!(started.elapsed() < Duration::from_secs(10), "no Slurm call");
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let interrupted = Instant::now();
    let output = child.wait_with_output().unwrap();
    let took = interrupted.elapsed();
    for pid in std::fs::read_to_string(&pids).unwrap().lines() {
        assert!(
            !alive(pid.parse().unwrap()),
            "Slurm child {pid} left running"
        );
    }
    let stderr = String::from_utf8(output.stderr).unwrap();
    (output.status.code().unwrap(), stderr, took)
}

#[test]
fn ctrl_c_aborts_hanging_slurm_calls() {
    // `why` runs one query, `q` with hints three concurrent ones
    for args in [&["why"][..], &["q"]] {
        let (code, stderr, took) = interrupt(&format!("sigint-{}", args[0]), args);
        assert_eq!((code, stderr.as_str()), (1, "\nAborted.\n"), "{args:?}");
        assert!(took < Duration::from_secs(3), "{args:?} took {took:?}");
    }
}

#[test]
fn ctrl_c_ends_watch_quietly() {
    let (code, stderr, took) = interrupt("sigint-watch", &["q", "-w", "10"]);
    assert_eq!((code, stderr.as_str()), (0, ""));
    assert!(took < Duration::from_secs(3), "took {took:?}");
}
