//! Error contract (binding for all modules).
//!
//! Python ends a command in four ways, each mapped to an [`AbError`] variant that bubbles
//! up to [`crate::cli::main`], which turns it into an exit code (and, for click errors,
//! prints them). Nothing below `main.rs` calls `std::process::exit`.
//!
//! | Python                                   | Rust                          | exit |
//! |------------------------------------------|-------------------------------|------|
//! | `die(msg, code)` -> `SystemExit(code)`   | `Err(die(msg))` -> `Exit(code)` (already printed) | code |
//! | bare `raise SystemExit(1)`               | `Err(AbError::Exit(1))`       | 1    |
//! | `click.UsageError` / `BadParameter` ...  | `Err(AbError::Usage(..))`, printed by cli in rich-click style | 2 |
//! | `click.ClickException(msg)`              | `Err(AbError::Click(msg))`, printed by cli (rich-click "Error: msg" panel) | 1 |
//! | `click.Abort` (prompt EOF/Ctrl-C)        | `Err(AbError::Abort)`, cli prints rich-click's "Aborted." (red) | 1 |
//! | `KeyboardInterrupt` in a handler         | `Err(AbError::Interrupted)`; `run_watched` turns it into `Ok(())`, `dev start` prints "Detached; ..." and re-raises; cli main treats it like rich-click (newline + "Aborted.", exit 1) | 1 |
//!
//! `--watch` (cli::watch::run_watched) suppresses `Exit(_)` (Python suppresses
//! `SystemExit`) after the message has been captured, and retries next round.

use crate::render::text::Text;

/// The Result type used everywhere in bab.
pub type Result<T, E = AbError> = std::result::Result<T, E>;

/// Why a command stopped early. See the module docs for the mapping to Python.
#[derive(Debug, Clone, PartialEq)]
pub enum AbError {
    /// `SystemExit(code)`: the message (if any) has ALREADY been printed (e.g. by [`die`]).
    Exit(i32),
    /// A click usage error (exit 2). The cli layer renders it with the usage line of the
    /// command in `command_path` and the rich-click error panel.
    Usage(UsageError),
    /// `click.ClickException(message)` (exit 1), e.g. a config error raised while building
    /// the default map. Rendered by the cli layer.
    Click(String),
    /// `click.Abort`: prompt aborted (EOF / Ctrl-C). cli prints `Aborted.` to stderr, exit 1.
    Abort,
    /// Ctrl-C (SIGINT) noticed by an interruptible wait ([`crate::signals`]).
    Interrupted,
}

/// A click `UsageError`/`BadParameter`/`NoSuchOption`... already formatted the click way.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageError {
    /// Full message as click formats it, e.g. `No such option: --foo` or
    /// `Invalid value for '-w' / '--watch': 5.0 is not in the range x>=10.`
    pub message: String,
    /// Command path whose usage line is shown, e.g. `["bab", "dev", "start"]`. Empty with
    /// `show_help_hint` means "the invoked command" (filled in by `cli::run` for errors a
    /// handler returns); empty without it means click had no context (no usage line and
    /// no hint, e.g. `Option '--user' requires an argument.`).
    pub command_path: Vec<String>,
    /// Whether rich-click prints the usage line and `Try 'bab ... -h' for help` (click
    /// attached a context to the error).
    pub show_help_hint: bool,
}

impl AbError {
    /// Exit status `cli::main` returns for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            AbError::Exit(code) => *code,
            AbError::Usage(_) => 2,
            AbError::Click(_) | AbError::Abort | AbError::Interrupted => 1,
        }
    }
}

impl std::fmt::Display for AbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbError::Exit(code) => write!(f, "exit {code}"),
            AbError::Usage(u) => write!(f, "usage error: {}", u.message),
            AbError::Click(msg) => write!(f, "{msg}"),
            AbError::Abort => write!(f, "Aborted."),
            AbError::Interrupted => write!(f, "interrupted"),
        }
    }
}

impl std::error::Error for AbError {}

/// Mirrors `core.die(message, exit_code=1)` minus the raise: prints
/// `Text.assemble(("error: ", "bold red"), message)` on the stderr console
/// (`ERROR_CONSOLE.print`, so it wraps at the stderr console width and is captured under
/// `--watch`) and returns `AbError::Exit(1)` for the caller to propagate:
///
/// ```ignore
/// return Err(die(format!("unknown node: {node_name}")));
/// ```
///
/// `message` may be a `&str`/`String` (plain, no markup, like a str inside Text.assemble)
/// or a styled [`Text`].
#[must_use = "die() only prints; return the error it gives you"]
pub fn die(message: impl Into<Text>) -> AbError {
    die_with_code(message, 1)
}

/// `die(message, exit_code=code)`; Python uses code 2 once (`load_job_records`).
#[must_use = "die() only prints; return the error it gives you"]
pub fn die_with_code(message: impl Into<Text>, code: i32) -> AbError {
    // After Ctrl-C Python is already unwinding with KeyboardInterrupt: a failure that only
    // follows from the interrupted (or never started) command is not reported.
    #[cfg(not(test))]
    if crate::signals::interrupted() {
        return AbError::Interrupted;
    }
    let line = Text::assemble([("error: ", "bold red").into(), message.into().into()]);
    crate::render::console::stderr().print(&line);
    AbError::Exit(code)
}
