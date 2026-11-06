//! `click.confirm`. Owned by P2-B.
//!
//! `click.confirm('Apply?', default=False)` echoes `Apply? [y/N]:` to stdout (no newline),
//! then Python's `input(" ")` writes the last prompt character and reads a line from
//! stdin: `y`/`yes` -> true, `n`/`no` -> false, empty -> default, anything else ->
//! `Error: invalid input` and ask again; EOF or Ctrl-C -> `click.Abort` ->
//! `Err(AbError::Abort)` (rich-click then prints `Aborted.`).

use crate::error::{AbError, Result};
use std::io::Write;

/// `click.confirm(text, default=default)`.
pub fn confirm(text: &str, default: bool) -> Result<bool> {
    let prompt = format!("{text} [{}]: ", if default { "Y/n" } else { "y/N" });
    let (head, last) = prompt.split_at(prompt.len() - 1);
    loop {
        write_stdout(head);
        write_stdout(last);
        let line = read_line()?.ok_or(AbError::Abort)?;
        let value = line.to_lowercase();
        match value.trim() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            "" => return Ok(default),
            _ => write_stdout("Error: invalid input\n"),
        }
    }
}

fn write_stdout(text: &str) {
    let mut out = std::io::stdout();
    let _ = out.write_all(text.as_bytes());
    let _ = out.flush();
}

/// One line from stdin without the newline; `None` on EOF before any input. Ctrl-C ->
/// `Err(Abort)`.
fn read_line() -> Result<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        crate::signals::check().map_err(|_| AbError::Abort)?;
        let mut byte = 0u8;
        let n = unsafe { libc::read(0, (&raw mut byte).cast(), 1) };
        if n < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(AbError::Abort);
        }
        if n == 0 {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }
        if byte == b'\n' {
            break;
        }
        bytes.push(byte);
    }
    Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
}
