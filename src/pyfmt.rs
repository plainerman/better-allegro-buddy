//! Python formatting and string helpers whose exact output shows up in user-facing
//! messages or decides how Slurm output is split.
//!
//! Owned by P2-C. All of these must match CPython 3.12 byte for byte.

use std::fmt::Write as _;

/// `str.isspace()` for one character (Python counts `\x1c`..`\x1f` as whitespace too).
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// `s.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s.split()` (no argument): runs of whitespace separate, no empty items.
pub fn split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|part| !part.is_empty())
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()`: every Python line boundary (`\r\n` counts once), no trailing empty line.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if is_line_break(c) {
            lines.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        lines.push(&s[start..]);
    }
    lines
}

/// Text-mode decoding of subprocess/file output (`text=True`, universal newlines):
/// UTF-8 (invalid bytes replaced) with `\r\n` and `\r` turned into `\n`.
pub fn decode_text(bytes: Vec<u8>) -> String {
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(err) => String::from_utf8_lossy(err.as_bytes()).into_owned(),
    };
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text
    }
}

/// `int(s)` for a `str`: surrounding whitespace, a sign and `_` between digits allowed.
pub fn parse_int(s: &str) -> Option<i64> {
    let s = strip(s);
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let cleaned = without_digit_underscores(digits)?;
    let magnitude: i128 = cleaned.parse().ok()?;
    let value = if negative { -magnitude } else { magnitude };
    i64::try_from(value).ok()
}

/// Remove `_` digit separators; None when one is not between two digits.
fn without_digit_underscores(s: &str) -> Option<std::borrow::Cow<'_, str>> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return None;
    }
    if !s.contains('_') {
        return Some(s.into());
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_'
            && (i == 0
                || i + 1 == bytes.len()
                || !bytes[i - 1].is_ascii_digit()
                || !bytes[i + 1].is_ascii_digit())
        {
            return None;
        }
    }
    Some(s.replace('_', "").into())
}

/// `float(s)` for a `str` (whitespace, sign, `_` between digits, `inf`/`nan`, exponents).
pub fn parse_float(s: &str) -> Option<f64> {
    let s = strip(s);
    if s.is_empty() {
        return None;
    }
    let cleaned: std::borrow::Cow<'_, str> = if s.contains('_') {
        let bytes = s.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'_'
                && (i == 0
                    || i + 1 == bytes.len()
                    || !bytes[i - 1].is_ascii_digit()
                    || !bytes[i + 1].is_ascii_digit())
            {
                return None;
            }
        }
        s.replace('_', "").into()
    } else {
        s.into()
    };
    let body = cleaned.trim_start_matches(['+', '-']);
    if cleaned.len() - body.len() > 1 {
        return None;
    }
    let lower = body.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity" | "nan") {
        return cleaned.parse().ok();
    }
    if !body
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return None;
    }
    cleaned.parse().ok()
}

/// Python `round(numerator / denominator)` (half to even) for an exact quotient.
pub fn round_div(numerator: i64, denominator: i64) -> i64 {
    let quotient = numerator.div_euclid(denominator);
    let twice_rest = 2 * numerator.rem_euclid(denominator);
    match twice_rest.cmp(&denominator) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal => quotient + quotient.rem_euclid(2),
    }
}

/// `math.ceil(numerator / denominator)` for a positive denominator.
pub fn ceil_div(numerator: i64, denominator: i64) -> i64 {
    -(-numerator).div_euclid(denominator)
}

/// `str.zfill(width)` (sign stays in front, width in code points).
pub fn zfill(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        return s.to_owned();
    }
    let zeros = "0".repeat(width - len);
    match s.chars().next() {
        Some(sign @ ('+' | '-')) => format!("{sign}{zeros}{}", &s[1..]),
        _ => format!("{zeros}{s}"),
    }
}

/// Best-effort `not str.isprintable()` for one character: control, format, separator
/// (except space), private-use and noncharacter code points. Unassigned code points
/// (Python's `Cn`) are not tracked.
fn is_unprintable(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x00..=0x1F | 0x7F..=0xA0 | 0xAD
        | 0x600..=0x605 | 0x61C | 0x6DD | 0x70F | 0x890..=0x891 | 0x8E2
        | 0x1680 | 0x180E | 0x2000..=0x200F | 0x2028..=0x202F | 0x205F..=0x206F
        | 0x3000 | 0xD800..=0xF8FF | 0xFDD0..=0xFDEF | 0xFEFF | 0xFFF9..=0xFFFB
        | 0x110BD | 0x110CD | 0x13430..=0x1343F | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A
        | 0xE0001 | 0xE0020..=0xE007F | 0xF0000..=0x10FFFF)
        || (u & 0xFFFE) == 0xFFFE
}

/// `repr(s)` for a `str`: single quotes unless the string contains `'` and no `"`;
/// escapes `\\`, the chosen quote, `\n`, `\r`, `\t`, other non-printable characters as
/// `\xNN` / `\uNNNN` / `\UNNNNNNNN`. Used by `{x!r}` in messages, e.g.
/// `unknown --gpu-type 'foo'`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_unprintable(c) => {
                let u = c as u32;
                let _ = if u < 0x100 {
                    write!(out, "\\x{u:02x}")
                } else if u < 0x10000 {
                    write!(out, "\\u{u:04x}")
                } else {
                    write!(out, "\\U{u:08x}")
                };
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(list_of_str)`: `['a', 'b']` (empty: `[]`). Used for `sorted(SECTIONS)` etc.
pub fn repr_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| repr_str(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// `shlex.quote(s)`: `''` for empty, `s` unchanged when it only has `[A-Za-z0-9_@%+=:,./-]`,
/// else `'...'` with `'` -> `'"'"'`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_owned();
    }
    let safe = s
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&b));
    if safe {
        return s.to_owned();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// `shlex.join(argv)`: `shlex_quote` each element, joined by single spaces.
pub fn shlex_join<S: AsRef<str>>(argv: &[S]) -> String {
    let quoted: Vec<String> = argv.iter().map(|a| shlex_quote(a.as_ref())).collect();
    quoted.join(" ")
}

/// Digits and decimal exponent of `value` (finite, non-zero, positive) in Rust's
/// shortest round-trip form or rounded to `precision` significant digits.
fn decimal_digits(value: f64, precision: Option<usize>) -> (String, i32) {
    let formatted = match precision {
        Some(p) => format!("{:.*e}", p.saturating_sub(1), value),
        None => format!("{value:e}"),
    };
    let (mantissa, exponent) = formatted.split_once('e').expect("exponent form");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    (digits, exponent.parse().expect("exponent"))
}

fn special_float(value: f64) -> Option<String> {
    if value.is_nan() {
        Some("nan".to_owned())
    } else if value.is_infinite() {
        Some(if value > 0.0 { "inf" } else { "-inf" }.to_owned())
    } else {
        None
    }
}

/// Lay out `digits` (no leading zeros) with decimal exponent `exp` the way Python's
/// float formatting does: fixed when `low <= exp < high`, else `d.ddde+XX`.
fn layout(negative: bool, digits: &str, exp: i32, low: i32, high: i32, keep_point: bool) -> String {
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (low..high).contains(&exp) {
        if exp < 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                if keep_point {
                    out.push_str(".0");
                }
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    out
}

/// `format(value, 'g')` / `f'{value:g}'`: 6 significant digits, no trailing zeros,
/// exponent form outside 1e-4..1e6 (`10.0` -> `10`, `2.5` -> `2.5`, `1.23457e+06`).
pub fn format_g(value: f64) -> String {
    if let Some(special) = special_float(value) {
        return special;
    }
    let negative = value.is_sign_negative();
    if value == 0.0 {
        return if negative { "-0" } else { "0" }.to_owned();
    }
    let (digits, exp) = decimal_digits(value.abs(), Some(6));
    let digits = digits.trim_end_matches('0');
    layout(negative, digits, exp, -4, 6, false)
}

/// `repr(value)` / `str(value)` for a float: shortest round-trip digits, fixed notation
/// for exponents in -4..16 (`5.0`, `0.0001`, `1e+16`, `1e-05`).
pub fn float_repr(value: f64) -> String {
    if let Some(special) = special_float(value) {
        return special;
    }
    let negative = value.is_sign_negative();
    if value == 0.0 {
        return if negative { "-0.0" } else { "0.0" }.to_owned();
    }
    let (digits, exp) = decimal_digits(value.abs(), None);
    let digits = digits.trim_end_matches('0');
    layout(negative, digits, exp, -4, 16, true)
}

/// `str(OSError)` for an error about `path`: `[Errno 13] Permission denied: '/x'`.
pub fn os_error_str(err: &std::io::Error, path: &std::path::Path) -> String {
    let filename = repr_str(&path.to_string_lossy());
    match err.raw_os_error() {
        Some(code) => {
            let full = err.to_string();
            let suffix = format!(" (os error {code})");
            let message = full.strip_suffix(&suffix).unwrap_or(&full);
            format!("[Errno {code}] {message}: {filename}")
        }
        None => format!("{err}: {filename}"),
    }
}

/// `str(value)` of a value `json.loads` produced (`None`, `True`, `1.5`, `['a']`, ...).
pub fn json_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => json_repr(other),
    }
}

/// `repr(value)` of a value `json.loads` produced.
pub fn json_repr(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                float_repr(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(json_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), json_repr(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Python truthiness of a `json.loads` value.
pub fn json_truthy(value: &serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_quotes_like_python() {
        assert_eq!(repr_str("foo"), "'foo'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(repr_str("gpus\n"), "'gpus\\n'");
        assert_eq!(
            repr_str("t\tb\\\x07\u{a0}é\u{200b}"),
            "'t\\tb\\\\\\x07\\xa0é\\u200b'"
        );
        assert_eq!(repr_list(&["cluster", "dev"]), "['cluster', 'dev']");
        assert_eq!(repr_list::<&str>(&[]), "[]");
    }

    #[test]
    fn shlex_like_python() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("--user=alice"), "--user=alice");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(
            shlex_join(&["sacct", "--format=A,B", "x y"]),
            "sacct --format=A,B 'x y'"
        );
    }

    #[test]
    fn float_formats_like_python() {
        let g = [
            (10.0, "10"),
            (2.5, "2.5"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (-0.0, "-0"),
            (0.5, "0.5"),
            (999999.5, "1e+06"),
        ];
        for (value, text) in g {
            assert_eq!(format_g(value), text, "{value}");
        }
        let r = [
            (5.0, "5.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456789012345678.0, "1.2345678901234568e+17"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (10.5, "10.5"),
        ];
        for (value, text) in r {
            assert_eq!(float_repr(value), text, "{value}");
        }
    }

    #[test]
    fn int_and_float_parsing_like_python() {
        assert_eq!(parse_int(" 12 "), Some(12));
        assert_eq!(parse_int("+1_000"), Some(1000));
        assert_eq!(parse_int("-5"), Some(-5));
        for bad in ["", "1_", "_1", "1__0", "1.0", "abc", "- 1"] {
            assert_eq!(parse_int(bad), None, "{bad:?}");
        }
        assert_eq!(parse_float("3.7"), Some(3.7));
        assert_eq!(parse_float(" 1e3 "), Some(1000.0));
        assert_eq!(parse_float("1_0.5"), Some(10.5));
        assert!(parse_float("nan").unwrap().is_nan());
        assert_eq!(parse_float("-inf"), Some(f64::NEG_INFINITY));
        for bad in ["", "abc", "1,5", "--1", "0x10", "1e"] {
            assert_eq!(parse_float(bad), None, "{bad:?}");
        }
    }
}
