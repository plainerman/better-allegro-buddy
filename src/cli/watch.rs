//! `--watch`: Python `run_watched`. Owned by P2-B.
//!
//! ```text
//! interval None -> render() once, errors propagate.
//! loop:
//!   capture stdout console AND stderr console; run render(); an Err(AbError::Exit(_))
//!     (die / SystemExit) is swallowed after its message was captured;
//!     Err(Interrupted) ends the loop -> Ok(()) (exit 0); other errors propagate.
//!   stdout console .clear()                       (ESC[2J ESC[H on a terminal only)
//!   stdout console .print_str(f"every {interval:g}s · {time.strftime('%H:%M:%S')} · Ctrl-C to quit", style="dim")
//!                                                 (a plain str: markup + highlighting!)
//!   write captured stderr to stderr, flush; write captured stdout to stdout, flush
//!   signals::sleep(interval)  (Ctrl-C -> Ok(()))
//! ```
//! bab-only: an interval `time.sleep` cannot take (nan, inf, >= 2**63 ns) is rejected
//! up front as a usage error, where Python renders once and crashes with a traceback.
//! The clock comes from `hooks::now()`; `{interval:g}` from `pyfmt::format_g`.

use crate::error::{AbError, Result};
use crate::render::console;

/// `run_watched(render, interval)`.
pub fn run_watched(render: &mut dyn FnMut() -> Result<()>, interval: Option<f64>) -> Result<()> {
    let Some(interval) = interval else {
        return render();
    };
    let pause = super::parser::sleep_duration(&super::commands::WATCH, interval)?;
    let (out, err) = (console::stdout(), console::stderr());
    loop {
        out.begin_capture();
        err.begin_capture();
        let result = render();
        let errors = err.end_capture();
        let captured = out.end_capture();
        match result {
            Ok(()) | Err(AbError::Exit(_)) => {}
            Err(AbError::Interrupted) => return stopped(),
            Err(other) => return Err(other),
        }
        if crate::signals::check().is_err() {
            return stopped();
        }
        out.clear();
        let now = crate::hooks::now().format("%H:%M:%S");
        out.print_str(
            &format!(
                "every {}s · {now} · Ctrl-C to quit",
                crate::pyfmt::format_g(interval)
            ),
            Some("dim"),
        );
        err.write_raw(&errors);
        out.write_raw(&captured);
        match crate::signals::sleep(pause) {
            Ok(()) => {}
            Err(_) => return stopped(),
        }
    }
}

/// Ctrl-C ended the loop: Python's `except KeyboardInterrupt: return`, so the interrupt
/// is handled and must not make `cli::main` report `Aborted.` (exit 1).
fn stopped() -> Result<()> {
    crate::signals::clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_interval_errors_propagate() {
        let mut render = || Err(AbError::Exit(3));
        assert_eq!(run_watched(&mut render, None), Err(AbError::Exit(3)));
    }

    #[test]
    fn rerenders_survives_die_and_stops_on_ctrl_c() {
        let mut calls = 0;
        let mut render = || {
            calls += 1;
            match calls {
                1 => Err(AbError::Exit(1)),
                2 => Ok(()),
                _ => {
                    crate::signals::simulate();
                    Err(AbError::Interrupted)
                }
            }
        };
        assert_eq!(run_watched(&mut render, Some(0.001)), Ok(()));
        assert_eq!(calls, 3);
        // handled: cli::main's final check must not turn it into "Aborted." (exit 1)
        assert!(!crate::signals::interrupted());
    }

    #[test]
    fn other_errors_end_the_loop() {
        let mut render = || Err(AbError::Click("boom".into()));
        assert_eq!(
            run_watched(&mut render, Some(0.001)),
            Err(AbError::Click("boom".into()))
        );
    }
}
