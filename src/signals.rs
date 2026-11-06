//! Ctrl-C handling (Python `KeyboardInterrupt`). Owned by P2-B.
//!
//! Python turns SIGINT into `KeyboardInterrupt` wherever the interpreter happens to be.
//! bab installs a SIGINT handler at startup ([`install`], called by `cli::main`) that only
//! sets a flag; the places where Python can realistically be interrupted check it:
//!
//! * [`sleep`] (dev wait polling, `--watch` interval) returns `Err(AbError::Interrupted)`.
//! * Slurm subprocess waits ([`crate::core::slurm`]) return `Err(AbError::Interrupted)`
//!   at once: like Python's `subprocess.run` they give the child 0.25 s to exit (it got
//!   SIGINT from the tty too), then kill and reap it; no new command starts afterwards.
//!
//! Waiting threads wake up through a self-pipe ([`wake_fd`]): the handler writes a byte
//! that nobody reads until [`clear`], so the read end stays readable and every `poll(2)`
//! on it returns, whichever thread the signal was delivered to. No polling loops.
//! * Prompts ([`crate::cli::prompt`]) map Ctrl-C/EOF to `AbError::Abort`.
//!
//! [`restore_default`] resets SIGINT (and SIGPIPE) to default before `execvp`
//! (`core::slurm::exec_replace`), so ssh/srun behave as when started by Python.
//! SIGPIPE: Python ignores it and rich-click exits 1 on `BrokenPipeError`; bab exits 1
//! from a SIGPIPE handler (quietly, like `bab q | head` in Python).

use crate::error::{AbError, Result};
use std::os::fd::RawFd;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INSTALL: Once = Once::new();
/// The self-pipe (`O_NONBLOCK | O_CLOEXEC`); -1 before [`install`].
static WAKE_READ: AtomicI32 = AtomicI32::new(-1);
static WAKE_WRITE: AtomicI32 = AtomicI32::new(-1);

extern "C" fn on_sigint(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
    let fd = WAKE_WRITE.load(Ordering::SeqCst);
    if fd >= 0 {
        // write(2) is async-signal-safe; keep errno for the interrupted code.
        unsafe {
            let errno = *errno_location();
            libc::write(fd, b"!".as_ptr().cast(), 1);
            *errno_location() = errno;
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__errno_location() }
}

#[cfg(not(target_os = "linux"))]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__error() }
}

/// `pipe2(O_NONBLOCK | O_CLOEXEC)`.
#[cfg(target_os = "linux")]
fn wake_pipe() -> Option<[RawFd; 2]> {
    let mut fds = [-1; 2];
    (unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_NONBLOCK | libc::O_CLOEXEC) } == 0)
        .then_some(fds)
}

/// macOS has no `pipe2`: `pipe` + `fcntl` (at startup, before any thread can fork).
#[cfg(not(target_os = "linux"))]
fn wake_pipe() -> Option<[RawFd; 2]> {
    let mut fds = [-1; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return None;
    }
    for fd in fds {
        unsafe {
            libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
            let flags = libc::fcntl(fd, libc::F_GETFL);
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }
    Some(fds)
}

/// Python ignores SIGPIPE and exits 1 on `BrokenPipeError` (rich-click's main); exiting
/// from the handler gives the same status without a message.
extern "C" fn on_sigpipe(_: libc::c_int) {
    unsafe { libc::_exit(1) }
}

fn set_handler(signal: libc::c_int, handler: libc::sighandler_t) {
    // No SA_RESTART: blocking reads (prompts) return EINTR on Ctrl-C.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler;
        libc::sigemptyset(&mut action.sa_mask);
        action.sa_flags = 0;
        libc::sigaction(signal, &action, std::ptr::null_mut());
    }
}

/// Install the flag-setting SIGINT handler and the SIGPIPE handler (idempotent).
pub fn install() {
    INSTALL.call_once(|| {
        if let Some(fds) = wake_pipe() {
            WAKE_READ.store(fds[0], Ordering::SeqCst);
            WAKE_WRITE.store(fds[1], Ordering::SeqCst);
        }
        set_handler(
            libc::SIGINT,
            on_sigint as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
        set_handler(
            libc::SIGPIPE,
            on_sigpipe as extern "C" fn(libc::c_int) as libc::sighandler_t,
        );
    });
}

/// True once SIGINT arrived (since [`install`]).
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// Forget a SIGINT that was handled (Python caught the `KeyboardInterrupt`, e.g.
/// `--watch` ending on Ctrl-C), so `cli::main`'s final check does not report it again.
pub fn clear() {
    INTERRUPTED.store(false, Ordering::SeqCst);
    let fd = WAKE_READ.load(Ordering::SeqCst);
    if fd >= 0 {
        let mut buf = [0u8; 64];
        while unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
    }
}

/// Read end of the self-pipe: readable once SIGINT arrived (until [`clear`]). `None`
/// before [`install`] (unit tests).
pub fn wake_fd() -> Option<RawFd> {
    let fd = WAKE_READ.load(Ordering::SeqCst);
    (fd >= 0).then_some(fd)
}

/// `poll(2)` on `fds` (retried on EINTR), `None` = no timeout. Returns false on timeout.
pub(crate) fn poll(fds: &mut [libc::pollfd], timeout: Option<Duration>) -> std::io::Result<bool> {
    let deadline = timeout.and_then(|t| Instant::now().checked_add(t));
    loop {
        let wait_ms = match (timeout, deadline) {
            (None, _) | (Some(_), None) => -1,
            (Some(_), Some(deadline)) => {
                let left = deadline.saturating_duration_since(Instant::now());
                // round up so the deadline has passed when poll returns 0
                left.as_nanos().div_ceil(1_000_000).min(i32::MAX as u128) as libc::c_int
            }
        };
        let rc = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, wait_ms) };
        if rc > 0 {
            return Ok(true);
        }
        if rc == 0 {
            if deadline.is_none_or(|d| Instant::now() >= d) {
                return Ok(false);
            }
            continue;
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

/// Pretend SIGINT arrived (unit tests).
#[cfg(test)]
pub(crate) fn simulate() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

/// Return `Err(AbError::Interrupted)` if SIGINT arrived.
pub fn check() -> Result<()> {
    if interrupted() {
        Err(AbError::Interrupted)
    } else {
        Ok(())
    }
}

/// Interruptible `time.sleep(duration)`: wakes early with `Err(Interrupted)` on Ctrl-C.
pub fn sleep(duration: Duration) -> Result<()> {
    check()?;
    if let Some(fd) = wake_fd() {
        let mut fds = [libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        }];
        // a duration too long for an Instant: sleep until Ctrl-C
        let timeout = Instant::now().checked_add(duration).map(|_| duration);
        if poll(&mut fds, timeout).is_ok() {
            return check();
        }
    }
    let deadline = Instant::now().checked_add(duration);
    loop {
        check()?;
        let left = match deadline {
            Some(deadline) => deadline.saturating_duration_since(Instant::now()),
            None => Duration::from_secs(3600),
        };
        if left.is_zero() {
            return Ok(());
        }
        std::thread::sleep(left.min(Duration::from_millis(50)));
    }
}

/// SIGINT and SIGPIPE back to `SIG_DFL` (before exec).
pub fn restore_default() {
    set_handler(libc::SIGINT, libc::SIG_DFL);
    set_handler(libc::SIGPIPE, libc::SIG_DFL);
}
