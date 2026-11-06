//! Running Slurm (and other) commands. Mirrors `core.run`, `core.exec_replace`,
//! `core.run_parsable_rows`. Owned by P2-C.
//!
//! # Contract
//!
//! * [`run`]`(argv, check)` == Python `run(argv, check=check)` with `capture_output=True,
//!   text=True`: stdout/stderr captured and decoded as UTF-8 (lossy; Python would raise on
//!   invalid UTF-8 which never happens with Slurm). `stdin` inherited (Python default).
//!   - command missing (`FileNotFoundError` on exec: ENOENT, incl. not on PATH):
//!     `check` -> `die("required command not found: {argv[0]}")`;
//!     unchecked -> `Output { status: 127, stdout: "", stderr: "command not found: {argv[0]}" }`.
//!     (Other spawn errors, e.g. EACCES, raised a Python traceback; map them to the same
//!     die/127 path; document if a scenario hits it.)
//!   - `check` and non-zero exit: `die("{shlex.join(argv)} failed: {detail}")` with
//!     `detail = stderr.strip() or stdout.strip() or "command failed with exit code {rc}"`.
//!   - a child killed by a signal has Python returncode `-signum`.
//! * [`try_run`]: the same but NEVER prints: returns [`RunError`], so callers inside
//!   [`parallel`] can report errors later in deterministic (Python call) order with
//!   [`RunError::die`].
//! * [`run_inherit`]: `run(argv, capture_output=False)` (scancel): stdout/stderr inherited,
//!   check semantics the same (detail from empty strings -> "command failed with exit code").
//! * [`exec_replace`]: `os.execvp`; on ENOENT `die("command not found: {argv[0]}")`.
//!   Flushes both consoles and restores default signal dispositions first
//!   ([`crate::signals::restore_default`]).
//! * Ctrl-C: if SIGINT arrives while waiting, the wait ends at once (self-pipe of
//!   [`crate::signals`]); like Python's `subprocess.run` the child gets 0.25 s to exit
//!   (the tty sent it SIGINT too), then it is killed and reaped, and the call returns
//!   `Err(AbError::Interrupted)` (Python raises KeyboardInterrupt). After that no command
//!   is started any more, so queued [`parallel`] tasks finish without running anything.
//!
//! # Concurrency
//!
//! [`prefetch`] starts read-only queries a command is about to make on background threads;
//! the later [`run`]/[`try_run`] of the same argv takes the result, so checks and error
//! messages happen in Python's order. A query may go unused when an earlier call dies
//! (fine: read-only); conditional fallbacks that are rarely needed are not prefetched.
//! [`parallel`] runs independent closures on scoped threads (`std::thread::scope`) and
//! returns results in input order. Never mutations (sbatch, scancel, scontrol update,
//! ssh/srun). Closures must not print: use [`try_run`] inside and report errors
//! afterwards in Python's order. Node facts ([`super::nodes::remember_nodes`]) are
//! thread-safe. At most [`MAX_PARALLEL`] queries run at once, so one user never floods
//! slurmctld.
//!
//! # Tests
//!
//! Under `cfg(test)` every command goes to the `fake::FakeSlurm` of the current
//! `crate::hooks::testing::Ctx` (panics without one); release builds spawn directly.

use crate::error::{AbError, Result};
use crate::pyfmt;
use std::io;
use std::os::fd::{AsRawFd, RawFd};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// `subprocess.CompletedProcess` subset.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Output {
    /// Python `returncode` (negative signal number if killed by a signal).
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Output {
    pub fn success(&self) -> bool {
        self.status == 0
    }
}

/// A checked run that failed, not yet reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// Executable not found: message `required command not found: {argv0}`.
    NotFound { argv0: String },
    /// Non-zero exit: message `{shlex.join(argv)} failed: {detail}`.
    Failed {
        joined_argv: String,
        detail: String,
        output: Output,
    },
}

impl RunError {
    /// The message `die()` prints (without `error: `).
    pub fn message(&self) -> String {
        match self {
            RunError::NotFound { argv0 } => format!("required command not found: {argv0}"),
            RunError::Failed {
                joined_argv,
                detail,
                ..
            } => format!("{joined_argv} failed: {detail}"),
        }
    }

    /// Print it like Python's `die` and return the error to propagate.
    #[must_use]
    pub fn die(&self) -> AbError {
        crate::error::die(self.message())
    }
}

/// Most commands [`parallel`] runs at the same time.
pub const MAX_PARALLEL: usize = 8;

/// Spawn `argv`, wait, and capture its output (or inherit stdout/stderr). `Err` only when
/// it could not be started. A captured run that was [`prefetch`]ed takes that result.
fn spawn<S: AsRef<str>>(argv: &[S], capture: bool) -> io::Result<Output> {
    if capture && let Some(result) = take_prefetched(argv) {
        return result;
    }
    spawn_now(argv, capture)
}

fn spawn_now<S: AsRef<str>>(argv: &[S], capture: bool) -> io::Result<Output> {
    // Python's subprocess looks the name up via os.get_exec_path (warnings.catch_warnings)
    crate::cli::help::python_warning_filters_mutated();
    #[cfg(test)]
    {
        let _ = capture;
        let argv: Vec<String> = argv.iter().map(|a| a.as_ref().to_owned()).collect();
        crate::hooks::testing::ctx()
            .slurm
            .as_ref()
            .unwrap_or_else(|| panic!("no FakeSlurm installed for {argv:?}"))
            .call(&argv)
    }
    #[cfg(not(test))]
    {
        let Some((program, args)) = argv.split_first() else {
            return Err(io::ErrorKind::NotFound.into());
        };
        if crate::signals::interrupted() {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let mut command = Command::new(program.as_ref());
        command.args(args.iter().map(AsRef::as_ref));
        run_child(command, capture, crate::signals::wake_fd())
    }
}

/// How long an interrupted child may take to exit before it is killed (Python's
/// `Popen._sigint_wait_secs`).
const SIGINT_GRACE: Duration = Duration::from_millis(250);

/// Spawn `command` (stdin inherited; stdout/stderr captured or inherited) and wait for it.
/// `wake` becoming readable means Ctrl-C: the child is terminated and reaped and the result
/// is `Err(Interrupted)`.
fn run_child(mut command: Command, capture: bool, wake: Option<RawFd>) -> io::Result<Output> {
    let stdio = || {
        if capture {
            Stdio::piped()
        } else {
            Stdio::inherit()
        }
    };
    command
        .stdin(Stdio::inherit())
        .stdout(stdio())
        .stderr(stdio());
    let mut child = command.spawn()?;
    let (stdout, stderr) = if capture {
        match read_pipes(&mut child, wake) {
            Ok(Some(streams)) => streams,
            Ok(None) => return Err(stop_child(child)),
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(err);
            }
        }
    } else {
        (Vec::new(), Vec::new())
    };
    match wait_child(&mut child, wake)? {
        Some(status) => Ok(Output {
            status: exit_status(status),
            stdout: pyfmt::decode_text(stdout),
            stderr: pyfmt::decode_text(stderr),
        }),
        None => Err(stop_child(child)),
    }
}

fn pollfd(fd: RawFd) -> libc::pollfd {
    libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    }
}

/// Read the child's stdout and stderr to EOF; `None` when `wake` fired first.
fn read_pipes(child: &mut Child, wake: Option<RawFd>) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    use std::io::Read;
    let mut pipes: [Option<Box<dyn ReadFd>>; 2] = [
        child.stdout.take().map(|p| Box::new(p) as Box<dyn ReadFd>),
        child.stderr.take().map(|p| Box::new(p) as Box<dyn ReadFd>),
    ];
    let mut buffers = [Vec::new(), Vec::new()];
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let open: Vec<usize> = (0..2).filter(|&i| pipes[i].is_some()).collect();
        if open.is_empty() {
            let [out, err] = buffers;
            return Ok(Some((out, err)));
        }
        let mut polled: Vec<libc::pollfd> = open
            .iter()
            .map(|&i| pollfd(pipes[i].as_ref().expect("open pipe").as_raw_fd()))
            .collect();
        polled.extend(wake.map(pollfd));
        crate::signals::poll(&mut polled, None)?;
        if wake.is_some() && polled[open.len()].revents != 0 {
            return Ok(None);
        }
        for (slot, &i) in open.iter().enumerate() {
            if polled[slot].revents == 0 {
                continue;
            }
            let pipe = pipes[i].as_mut().expect("open pipe");
            match pipe.read(&mut chunk) {
                Ok(0) => pipes[i] = None,
                Ok(n) => buffers[i].extend_from_slice(&chunk[..n]),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(err),
            }
        }
    }
}

/// A child's stdout or stderr pipe.
trait ReadFd: std::io::Read + AsRawFd {}
impl<T: std::io::Read + AsRawFd> ReadFd for T {}

/// `pidfd_open(2)` (Linux 5.3+): becomes readable when the process exits.
#[cfg(target_os = "linux")]
fn pidfd_open(child: &Child) -> Option<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    // Raw syscall: glibc's wrapper is 2.36+, wheels target manylinux2014 (glibc 2.17).
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, child.id() as libc::pid_t, 0) };
    (fd >= 0).then(|| unsafe { std::os::fd::OwnedFd::from_raw_fd(fd as RawFd) })
}

/// No pidfds outside Linux: [`stop_child`] polls `try_wait` instead.
#[cfg(not(target_os = "linux"))]
fn pidfd_open(_: &Child) -> Option<std::os::fd::OwnedFd> {
    None
}

/// Wait for the child to exit; `None` when `wake` fired first. Waits on a pidfd so any
/// thread's Ctrl-C ends the wait; without pidfds (old kernels) a blocking wait (the child
/// got the terminal's SIGINT as well).
#[cfg(target_os = "linux")]
fn wait_child(
    child: &mut Child,
    wake: Option<RawFd>,
) -> io::Result<Option<std::process::ExitStatus>> {
    if let (Some(wake), Some(pidfd)) = (wake, pidfd_open(child)) {
        let mut polled = [pollfd(pidfd.as_raw_fd()), pollfd(wake)];
        crate::signals::poll(&mut polled, None)?;
        if polled[0].revents == 0 {
            return Ok(None);
        }
    }
    child.wait().map(Some)
}

/// [`wait_child`] without pidfds (macOS): poll `wake` and a kqueue that reports the
/// child's exit (`EVFILT_PROC`/`NOTE_EXIT`), re-checking `try_wait` at least every
/// [`EXIT_RECHECK`] in case the exit happened before the kqueue was armed.
#[cfg(not(target_os = "linux"))]
fn wait_child(
    child: &mut Child,
    wake: Option<RawFd>,
) -> io::Result<Option<std::process::ExitStatus>> {
    const EXIT_RECHECK: Duration = Duration::from_millis(50);
    let Some(wake) = wake else {
        return child.wait().map(Some);
    };
    let exit = exit_kqueue(child);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        // poll(2) ignores negative fds
        let exit_fd = exit.as_ref().map_or(-1, |fd| fd.as_raw_fd());
        let mut polled = [pollfd(wake), pollfd(exit_fd)];
        crate::signals::poll(&mut polled, Some(EXIT_RECHECK))?;
        if polled[0].revents != 0 {
            return Ok(None);
        }
    }
}

/// A kqueue that becomes readable when `child` exits (not inherited across fork).
#[cfg(any(
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
fn exit_kqueue(child: &Child) -> Option<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    let kq = unsafe { libc::kqueue() };
    if kq < 0 {
        return None;
    }
    let kq = unsafe { std::os::fd::OwnedFd::from_raw_fd(kq) };
    let mut change: libc::kevent = unsafe { std::mem::zeroed() };
    change.ident = child.id() as _;
    change.filter = libc::EVFILT_PROC;
    change.flags = libc::EV_ADD;
    change.fflags = libc::NOTE_EXIT;
    let rc = unsafe {
        libc::kevent(
            kq.as_raw_fd(),
            &change,
            1,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
        )
    };
    (rc == 0).then_some(kq)
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
fn exit_kqueue(_: &Child) -> Option<std::os::fd::OwnedFd> {
    None
}

/// Ctrl-C while waiting: give the child [`SIGINT_GRACE`] to exit, then kill it; always
/// reap it. Returns the `Interrupted` error to propagate.
fn stop_child(mut child: Child) -> io::Error {
    drop(child.stdout.take());
    drop(child.stderr.take());
    let exited = match pidfd_open(&child) {
        Some(pidfd) => {
            let mut polled = [pollfd(pidfd.as_raw_fd())];
            crate::signals::poll(&mut polled, Some(SIGINT_GRACE)).unwrap_or(false)
        }
        None => {
            let deadline = std::time::Instant::now() + SIGINT_GRACE;
            loop {
                match child.try_wait() {
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(10))
                    }
                    Ok(Some(_)) => break true,
                    _ => break false,
                }
            }
        }
    };
    if !exited {
        let _ = child.kill();
    }
    let _ = child.wait();
    io::ErrorKind::Interrupted.into()
}

/// Python's `returncode`: the exit code, or `-signum` when killed by a signal.
fn exit_status(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| -status.signal().unwrap_or(0))
}

/// Python only raised `FileNotFoundError` for a missing command; other spawn errors
/// (EACCES, ...) end up on the same path here instead of a traceback.
fn not_found<S: AsRef<str>>(argv: &[S]) -> RunError {
    RunError::NotFound {
        argv0: argv
            .first()
            .map(|a| a.as_ref().to_owned())
            .unwrap_or_default(),
    }
}

fn check_output<S: AsRef<str>>(
    argv: &[S],
    output: Output,
) -> std::result::Result<Output, RunError> {
    if output.success() {
        return Ok(output);
    }
    let stderr = pyfmt::strip(&output.stderr);
    let stdout = pyfmt::strip(&output.stdout);
    let detail = if !stderr.is_empty() {
        stderr.to_owned()
    } else if !stdout.is_empty() {
        stdout.to_owned()
    } else {
        format!("command failed with exit code {}", output.status)
    };
    Err(RunError::Failed {
        joined_argv: pyfmt::shlex_join(argv),
        detail,
        output,
    })
}

fn interrupted() -> bool {
    #[cfg(test)]
    return false;
    #[cfg(not(test))]
    crate::signals::interrupted()
}

/// `Err(AbError::Interrupted)` once Ctrl-C arrived (after joining [`parallel`] work that
/// used [`try_run`]).
pub fn check_interrupted() -> Result<()> {
    if interrupted() {
        return Err(AbError::Interrupted);
    }
    Ok(())
}

/// `core.run(argv, check=check)`. See module docs.
pub fn run<S: AsRef<str>>(argv: &[S], check: bool) -> Result<Output> {
    let result = try_run(argv, check);
    if interrupted() {
        return Err(AbError::Interrupted);
    }
    result.map_err(|err| err.die())
}

/// [`run`] without printing: `Err` only for checked failures (`check=false` always gives
/// `Ok`, with the 127 output for a missing command). Ctrl-C is not reported here; callers
/// check [`crate::signals::check`] after joining.
pub fn try_run<S: AsRef<str>>(argv: &[S], check: bool) -> std::result::Result<Output, RunError> {
    match spawn(argv, true) {
        Ok(output) if check => check_output(argv, output),
        Ok(output) => Ok(output),
        Err(_) if check => Err(not_found(argv)),
        Err(_) => {
            let argv0 = argv.first().map(AsRef::as_ref).unwrap_or_default();
            Ok(Output {
                status: 127,
                stdout: String::new(),
                stderr: format!("command not found: {argv0}"),
            })
        }
    }
}

/// `run(argv, capture_output=False)` (only `scancel` in Python), checked.
pub fn run_inherit<S: AsRef<str>>(argv: &[S]) -> Result<Output> {
    let result = match spawn(argv, false) {
        Ok(output) => check_output(argv, output),
        Err(_) => Err(not_found(argv)),
    };
    if interrupted() {
        return Err(AbError::Interrupted);
    }
    result.map_err(|err| err.die())
}

/// `core.exec_replace(argv)`: replaces the process; only returns on failure, after
/// printing `error: command not found: {argv0}` (Python's message for ENOENT; other exec
/// errors, a Python traceback, take the same path). The consoles write through on every
/// print, so there is nothing to flush. Under `cfg(test)` the argv goes to the fake and a
/// successful "exec" returns `AbError::Exit(0)`.
pub fn exec_replace<S: AsRef<str>>(argv: &[S]) -> AbError {
    let argv0 = argv.first().map(AsRef::as_ref).unwrap_or_default();
    #[cfg(test)]
    {
        let argv: Vec<String> = argv.iter().map(|a| a.as_ref().to_owned()).collect();
        let ctx = crate::hooks::testing::ctx();
        let slurm = ctx.slurm.as_ref().expect("no FakeSlurm installed");
        if slurm.call(&argv).is_ok() {
            return AbError::Exit(0);
        }
    }
    #[cfg(not(test))]
    {
        use std::os::unix::process::CommandExt;
        crate::signals::restore_default();
        let _ = Command::new(argv0)
            .args(argv.iter().skip(1).map(AsRef::as_ref))
            .exec();
    }
    crate::error::die(format!("command not found: {argv0}"))
}

/// `run_parsable_rows(argv, headers, check=check)` = `parse_pipe_rows(run(...).stdout, headers)`.
pub fn run_parsable_rows<S: AsRef<str>>(
    argv: &[S],
    headers: &[&str],
    check: bool,
) -> Result<Vec<super::Record>> {
    let output = run(argv, check)?;
    Ok(super::parse::parse_pipe_rows(&output.stdout, headers))
}

#[cfg(test)]
pub type PrefetchedForTests = Prefetched;
type Prefetched = Vec<(Vec<String>, std::thread::JoinHandle<io::Result<Output>>)>;

fn with_prefetched<T>(f: impl FnOnce(&mut Prefetched) -> T) -> T {
    #[cfg(test)]
    {
        f(&mut crate::hooks::testing::ctx()
            .prefetched
            .lock()
            .expect("prefetched"))
    }
    #[cfg(not(test))]
    {
        static PREFETCHED: std::sync::Mutex<Prefetched> = std::sync::Mutex::new(Vec::new());
        f(&mut PREFETCHED.lock().expect("prefetched"))
    }
}

/// The result of a [`prefetch`]ed run of exactly `argv`, waiting for it if needed.
fn take_prefetched<S: AsRef<str>>(argv: &[S]) -> Option<io::Result<Output>> {
    let handle = with_prefetched(|pending| {
        let index = pending.iter().position(|(key, _)| {
            key.len() == argv.len() && key.iter().zip(argv).all(|(k, a)| k == a.as_ref())
        })?;
        Some(pending.swap_remove(index).1)
    })?;
    Some(
        handle
            .join()
            .unwrap_or_else(|_| Err(io::ErrorKind::Other.into())),
    )
}

/// Read-only queries started early by [`prefetch`]; dropping it forgets the ones nobody
/// asked for (they finish on their own), so a `--watch` round never sees stale answers.
#[must_use = "dropping the guard forgets the prefetched queries"]
pub struct Prefetch(Vec<Vec<String>>);

impl Drop for Prefetch {
    fn drop(&mut self) {
        let unused: Prefetched = with_prefetched(|pending| {
            let (unused, kept) = std::mem::take(pending)
                .into_iter()
                .partition(|(key, _)| self.0.contains(key));
            *pending = kept;
            unused
        });
        // Unit tests wait for them, so their call logs are complete.
        #[cfg(test)]
        for (_, handle) in unused {
            let _ = handle.join();
        }
        #[cfg(not(test))]
        drop(unused);
    }
}

/// Start read-only queries now, each on its own thread; a later captured [`run`] /
/// [`try_run`] of the same argv takes the result instead of starting the command again.
/// Checks and error reporting happen there, in the caller's (Python's) order. Only for
/// read-only Slurm queries the command is about to make (an earlier failure may leave
/// one unused); never mutations. At most [`MAX_PARALLEL`] pending at once; an argv that
/// is already pending is not started twice.
pub fn prefetch(argvs: Vec<Vec<String>>) -> Prefetch {
    let argvs = with_prefetched(|pending| {
        let mut started: Vec<Vec<String>> = Vec::new();
        for argv in argvs {
            // already on its way (an outer prefetch), or too many at once
            if pending.len() >= MAX_PARALLEL || pending.iter().any(|(key, _)| *key == argv) {
                continue;
            }
            #[cfg(test)]
            let ctx = crate::hooks::testing::ctx();
            let owned = argv.clone();
            let run = move || spawn_now(&owned, true);
            #[cfg(test)]
            let run = move || crate::hooks::testing::with_ctx(ctx, run);
            pending.push((argv.clone(), std::thread::spawn(run)));
            started.push(argv);
        }
        started
    });
    Prefetch(argvs)
}

/// `["a", "b"]` as an owned argv.
pub fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

/// A boxed task for [`parallel`].
pub type Task<'a, T> = Box<dyn FnOnce() -> T + Send + 'a>;

/// Run `tasks` concurrently on scoped threads (at most [`MAX_PARALLEL`] at a time);
/// results in input order. With one task, runs inline. See module docs for what may run
/// here.
pub fn parallel<'a, T: Send>(tasks: Vec<Task<'a, T>>) -> Vec<T> {
    if tasks.len() <= 1 {
        return tasks.into_iter().map(|task| task()).collect();
    }
    let count = tasks.len();
    let queue: std::sync::Mutex<std::vec::IntoIter<(usize, Task<'a, T>)>> = std::sync::Mutex::new(
        tasks
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_iter(),
    );
    let results: std::sync::Mutex<Vec<Option<T>>> =
        std::sync::Mutex::new((0..count).map(|_| None).collect());
    #[cfg(test)]
    let ctx = crate::hooks::testing::ctx();
    std::thread::scope(|scope| {
        for _ in 0..count.min(MAX_PARALLEL) {
            #[cfg(test)]
            let ctx = ctx.clone();
            let worker = || loop {
                let next = queue.lock().expect("task queue").next();
                let Some((index, task)) = next else { break };
                let value = task();
                results.lock().expect("results")[index] = Some(value);
            };
            #[cfg(test)]
            scope.spawn(move || crate::hooks::testing::with_ctx(ctx, worker));
            #[cfg(not(test))]
            scope.spawn(worker);
        }
    });
    results
        .into_inner()
        .expect("results")
        .into_iter()
        .map(|value| value.expect("parallel task panicked"))
        .collect()
}

/// Two different-typed tasks concurrently (convenience over [`parallel`]).
pub fn join2<A: Send, B: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(test)]
    let ctx = crate::hooks::testing::ctx();
    std::thread::scope(|scope| {
        #[cfg(test)]
        let hb = scope.spawn(move || crate::hooks::testing::with_ctx(ctx, b));
        #[cfg(not(test))]
        let hb = scope.spawn(b);
        let ra = a();
        (ra, hb.join().expect("parallel task panicked"))
    })
}

/// Three different-typed tasks concurrently.
pub fn join3<A: Send, B: Send, C: Send>(
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
    c: impl FnOnce() -> C + Send,
) -> (A, B, C) {
    let (ra, (rb, rc)) = join2(a, || join2(b, c));
    (ra, rb, rc)
}

/// The Slurm stand-in for unit tests (Python's `patch('subprocess.run', ...)`).
#[cfg(test)]
pub mod fake {
    use super::Output;
    use std::io;
    use std::sync::{Arc, Mutex};

    type Handler = dyn Fn(&[String]) -> Option<Output> + Send + Sync;

    /// Answers every command with a closure (`None` = command not found) and records argv.
    pub struct FakeSlurm {
        handler: Box<Handler>,
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl FakeSlurm {
        pub fn new(
            handler: impl Fn(&[String]) -> Option<Output> + Send + Sync + 'static,
        ) -> Arc<FakeSlurm> {
            Arc::new(FakeSlurm {
                handler: Box::new(handler),
                calls: Mutex::default(),
            })
        }

        /// Every command gets the same answer (`patch('subprocess.run', return_value=...)`).
        pub fn always(output: Output) -> Arc<FakeSlurm> {
            FakeSlurm::new(move |_| Some(output.clone()))
        }

        pub fn call(&self, argv: &[String]) -> io::Result<Output> {
            self.calls.lock().expect("calls").push(argv.to_vec());
            (self.handler)(argv).ok_or_else(|| io::ErrorKind::NotFound.into())
        }

        /// Every argv so far, in call order.
        pub fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().expect("calls").clone()
        }
    }

    /// `_proc(stdout=...)`.
    pub fn ok(stdout: &str) -> Output {
        Output {
            status: 0,
            stdout: stdout.to_owned(),
            stderr: String::new(),
        }
    }

    /// `_proc(stderr=..., returncode=...)`.
    pub fn fail(status: i32, stderr: &str) -> Output {
        Output {
            status,
            stdout: String::new(),
            stderr: stderr.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{FakeSlurm, fail, ok};
    use super::*;
    use crate::hooks::testing::Ctx;

    #[test]
    fn nonzero_returncode_fails_with_the_python_message() {
        let _ctx = Ctx::default()
            .slurm(FakeSlurm::always(fail(1, "bad\n")))
            .install();
        let err = try_run(&["scontrol", "show", "bad thing"], true).unwrap_err();
        assert_eq!(err.message(), "scontrol show 'bad thing' failed: bad");
        let quiet = FakeSlurm::always(Output {
            status: 3,
            ..ok(" out ")
        });
        let _ctx = Ctx::default().slurm(quiet).install();
        assert_eq!(
            try_run(&["x"], true).unwrap_err().message(),
            "x failed: out"
        );
        let _ctx = Ctx::default()
            .slurm(FakeSlurm::always(fail(-9, "")))
            .install();
        assert_eq!(
            try_run(&["x"], true).unwrap_err().message(),
            "x failed: command failed with exit code -9"
        );
    }

    #[test]
    fn file_not_found() {
        let _ctx = Ctx::default().slurm(FakeSlurm::new(|_| None)).install();
        assert_eq!(
            try_run(&["nonexistent_command"], true)
                .unwrap_err()
                .message(),
            "required command not found: nonexistent_command"
        );
        assert_eq!(
            run(&["nonexistent_command"], false).unwrap(),
            Output {
                status: 127,
                stdout: String::new(),
                stderr: "command not found: nonexistent_command".to_owned()
            }
        );
    }

    #[test]
    fn parallel_keeps_order_and_context() {
        let fake = FakeSlurm::new(|argv| Some(ok(&argv[1])));
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        let tasks: Vec<Task<'_, String>> = (0..20)
            .map(|i| -> Task<'_, String> {
                Box::new(move || {
                    run(&["echo".to_owned(), i.to_string()], true)
                        .unwrap()
                        .stdout
                })
            })
            .collect();
        let expected: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        assert_eq!(parallel(tasks), expected);
        assert_eq!(fake.calls().len(), 20);
        let (a, b, c) = join3(|| 1, crate::hooks::user, || "c");
        assert_eq!((a, b, c), (1, "alice", "c"));
    }

    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    /// A private self-pipe: (read end, write end).
    fn wake_pipe() -> (std::os::fd::OwnedFd, std::os::fd::OwnedFd) {
        let (read, write) = std::io::pipe().unwrap();
        (read.into(), write.into())
    }

    #[test]
    fn real_child_output_and_status() {
        let (wake, _keep) = wake_pipe();
        let output = run_child(
            sh("head -c 200000 /dev/zero; printf 'e\\r\\n' >&2; head -c 100000 /dev/zero >&2; exit 3"),
            true,
            Some(wake.as_raw_fd()),
        )
        .unwrap();
        assert_eq!(output.status, 3);
        assert_eq!(output.stdout.len(), 200000);
        assert_eq!(output.stderr.len(), 100002);
        assert!(output.stderr.starts_with("e\n"));
        let output = run_child(sh("kill -TERM $$"), true, None).unwrap();
        assert_eq!(output.status, -libc::SIGTERM);
        let output = run_child(sh("exit 4"), false, Some(wake.as_raw_fd())).unwrap();
        assert_eq!(output.status, 4);
    }

    #[test]
    fn ctrl_c_ends_the_wait_and_kills_the_child() {
        use std::io::Write;
        // `trap '' INT`: the child survives SIGINT, so the grace period ends with a kill
        for (capture, script) in [
            (true, "sleep 30"),
            (false, "sleep 30"),
            (true, "trap '' INT; exec sleep 30"),
        ] {
            let (wake, write_end) = wake_pipe();
            let started = std::time::Instant::now();
            let waker = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                std::fs::File::from(write_end).write_all(b"!").unwrap();
            });
            let err = run_child(sh(script), capture, Some(wake.as_raw_fd())).unwrap_err();
            waker.join().unwrap();
            assert_eq!(err.kind(), io::ErrorKind::Interrupted, "{script}");
            assert!(started.elapsed() < Duration::from_secs(5), "{script}");
        }
    }
}
