//! Time parsing/formatting from `core.py`. Owned by P2-C. "now" always comes from
//! [`crate::hooks::now`]; functions take it as a parameter where Python has `now=`.
//!
//! Python datetimes here are naive local times; `datetime.fromisoformat` accepts
//! `YYYY-MM-DDTHH:MM:SS` (Slurm's format) and more; [`iso_to_dt`] implements the subset
//! of Python 3.12's grammar listed there. Strings with a UTC offset (`Z`, `+01:00`) give
//! None here: Python would make an aware datetime that then breaks `end - now` with a
//! TypeError; Slurm never sends one. `strftime("%a"/"%b")` uses the C locale (English
//! names) since Python doesn't call setlocale.

use crate::pyfmt;
use chrono::{Datelike, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike};

/// `parse_slurm_time("1-02:03:04")` -> seconds; `UNLIMITED`/`INFINITE`/`INFINITY`/`N/A`
/// -> None; unparseable -> None. `MM:SS` without a day part, `D-HH:MM` with one; a single
/// number is minutes (`D-HH` with a day part). Python `int()` accepts surrounding
/// whitespace, `+`/`-` signs and `_` digit separators.
pub fn parse_slurm_time(value: &str) -> Option<i64> {
    let value = pyfmt::strip(value);
    if matches!(
        value.to_uppercase().as_str(),
        "UNLIMITED" | "INFINITE" | "INFINITY" | "N/A"
    ) {
        return None;
    }
    let with_days = value.contains('-');
    let (days, rest) = match value.split_once('-') {
        Some((day_part, rest)) => (pyfmt::parse_int(day_part)?, rest),
        None => (0, value),
    };
    let parts: Vec<&str> = rest.split(':').collect();
    let ints = || -> Option<Vec<i64>> { parts.iter().map(|p| pyfmt::parse_int(p)).collect() };
    let (hours, minutes, seconds) = match parts.len() {
        3 => {
            let v = ints()?;
            (v[0], v[1], v[2])
        }
        2 => {
            let v = ints()?;
            if with_days {
                (v[0], v[1], 0)
            } else {
                (0, v[0], v[1])
            }
        }
        1 => {
            let v = ints()?;
            if with_days {
                (v[0], 0, 0)
            } else {
                (0, v[0], 0)
            }
        }
        _ => return None,
    };
    days.checked_mul(86400)?
        .checked_add(hours.checked_mul(3600)?)?
        .checked_add(minutes.checked_mul(60)?)?
        .checked_add(seconds)
}

/// `format_seconds(value)`: None -> `unlimited`; `45s`, `40m`, `6h35m`, `2d2h`
/// (negatives clamp to 0 -> `0s`).
pub fn format_seconds(value: Option<i64>) -> String {
    let Some(value) = value else {
        return "unlimited".to_owned();
    };
    let value = value.max(0);
    if value < 60 {
        return format!("{value}s");
    }
    let (days, rem) = (value / 86400, value % 86400);
    let (hours, rem) = (rem / 3600, rem % 3600);
    let mut out = String::new();
    for (n, unit) in [(days, 'd'), (hours, 'h'), (rem / 60, 'm')] {
        if n != 0 {
            out.push_str(&n.to_string());
            out.push(unit);
        }
    }
    out
}

fn digits(s: &str) -> Option<u32> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// `HH[:MM[:SS[.f+]]]` or `HH[MM[SS[.f+]]]` (`,` works as decimal mark too).
fn parse_iso_time(s: &str) -> Option<NaiveTime> {
    let (clock, fraction) = match s.find(['.', ',']) {
        Some(at) => (&s[..at], Some(&s[at + 1..])),
        None => (s, None),
    };
    let fields: Vec<&str> = if clock.contains(':') {
        clock.split(':').collect()
    } else {
        if clock.len() % 2 != 0 || !clock.is_ascii() {
            return None;
        }
        (0..clock.len())
            .step_by(2)
            .map(|i| &clock[i..i + 2])
            .collect()
    };
    if fields.is_empty() || fields.len() > 3 || fields.iter().any(|f| f.len() != 2) {
        return None;
    }
    let mut values = [0u32; 3];
    for (slot, field) in values.iter_mut().zip(&fields) {
        *slot = digits(field)?;
    }
    let micro = match fraction {
        None => 0,
        Some(f) => {
            if fields.len() != 3 || f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let six: String = f.chars().chain(std::iter::repeat('0')).take(6).collect();
            six.parse().ok()?
        }
    };
    NaiveTime::from_hms_micro_opt(values[0], values[1], values[2], micro)
}

/// Python 3.12 `datetime.fromisoformat` for naive values: `YYYY-MM-DD` or `YYYYMMDD`,
/// optionally followed by any one separator character and a time.
fn from_isoformat(s: &str) -> Option<NaiveDateTime> {
    let bytes = s.as_bytes();
    // `s.get`: a slice that is not on char boundaries holds a non-ASCII digit -> None
    let field = |range: std::ops::Range<usize>| s.get(range).and_then(digits);
    let (date, date_len) = if bytes.len() >= 10 && bytes[4] == b'-' && bytes[7] == b'-' {
        let date = NaiveDate::from_ymd_opt(field(0..4)? as i32, field(5..7)?, field(8..10)?)?;
        (date, 10)
    } else if bytes.len() >= 8 {
        let date = NaiveDate::from_ymd_opt(field(0..4)? as i32, field(4..6)?, field(6..8)?)?;
        (date, 8)
    } else {
        return None;
    };
    let rest = &s[date_len..];
    let mut chars = rest.chars();
    let time = match chars.next() {
        None => NaiveTime::MIN,
        Some(_) => parse_iso_time(chars.as_str())?,
    };
    if date.year() < 1 {
        return None;
    }
    Some(date.and_time(time))
}

/// `iso_to_dt(value)`: None for empty, `Unknown`, `N/A`, `None` or unparseable.
pub fn iso_to_dt(value: &str) -> Option<NaiveDateTime> {
    if matches!(value, "" | "Unknown" | "N/A" | "None") {
        return None;
    }
    from_isoformat(value)
}

/// `dt.isoformat()` of a naive datetime (`2026-10-07T12:26:02`, `.ffffff` when set).
pub fn isoformat(dt: NaiveDateTime) -> String {
    let micro = dt.nanosecond() / 1000;
    if micro == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        format!("{}.{micro:06}", dt.format("%Y-%m-%dT%H:%M:%S"))
    }
}

/// `(a - b)` in microseconds.
pub fn delta_micros(a: NaiveDateTime, b: NaiveDateTime) -> i64 {
    let delta = a - b;
    delta
        .num_microseconds()
        .unwrap_or(if delta > chrono::TimeDelta::zero() {
            i64::MAX
        } else {
            i64::MIN
        })
}

/// `(a - b).total_seconds()`.
pub fn total_seconds(a: NaiveDateTime, b: NaiveDateTime) -> f64 {
    delta_micros(a, b) as f64 / 1e6
}

/// `int((a - b).total_seconds())` (truncation toward zero, microseconds included).
pub fn delta_seconds(a: NaiveDateTime, b: NaiveDateTime) -> i64 {
    total_seconds(a, b).trunc() as i64
}

/// `format_eta(end_time, now=now)`: `unknown` or `format_seconds(end - now)`.
pub fn format_eta(end_time: &str, now: NaiveDateTime) -> String {
    match iso_to_dt(end_time) {
        None => "unknown".to_owned(),
        Some(end) => format_seconds(Some(delta_seconds(end, now))),
    }
}

/// `format_eta(end_time)` with `now = hooks::now()`.
pub fn format_eta_now(end_time: &str) -> String {
    format_eta(end_time, crate::hooks::now())
}

/// `format_pending_start(start_time, now=now)`: `in 2h10m (Tue 16:40)`,
/// `due (Tue 16:40)`, or None; dates `< 7` calendar days ahead show the weekday, later
/// ones `%b %d %H:%M`.
pub fn format_pending_start(start_time: &str, now: NaiveDateTime) -> Option<String> {
    let start = iso_to_dt(start_time)?;
    let days_ahead = (start.date() - now.date()).num_days();
    let clock = if days_ahead < 7 {
        start.format("%a %H:%M").to_string()
    } else {
        start.format("%b %d %H:%M").to_string()
    };
    if start <= now {
        return Some(format!("due ({clock})"));
    }
    Some(format!(
        "in {} ({clock})",
        format_seconds(Some(delta_seconds(start, now)))
    ))
}

/// `format_history_end(end_time, now=now)`: `today 08:41`, `Mon 23:17` (1..6 days ago),
/// `%b %d %H:%M` same year, else `%Y-%m-%d`; unparseable: `-` if nullish else verbatim.
pub fn format_history_end(end_time: &str, now: NaiveDateTime) -> String {
    let Some(end) = iso_to_dt(end_time) else {
        return if super::parse::is_nullish(Some(end_time)) {
            "-".to_owned()
        } else {
            end_time.to_owned()
        };
    };
    let days = (now.date() - end.date()).num_days();
    let pattern = if days == 0 {
        "today %H:%M"
    } else if 0 < days && days < 7 {
        "%a %H:%M"
    } else if end.year() == now.year() {
        "%b %d %H:%M"
    } else {
        "%Y-%m-%d"
    };
    end.format(pattern).to_string()
}

/// `datetime.fromtimestamp(epoch).isoformat()` in local time (TZ honoured), used by
/// `why._start_text`.
pub fn local_iso_from_epoch(epoch: i64) -> String {
    match Local.timestamp_opt(epoch, 0).earliest() {
        Some(dt) => isoformat(dt.naive_local()),
        None => epoch.to_string(),
    }
}

/// `dt.timestamp()` of a naive local datetime (sorting keys in history/dev jobs). Like
/// Python (`fold=0`): ambiguous times take the earlier offset, times in a DST gap the
/// offset from before the gap.
pub fn local_timestamp(dt: NaiveDateTime) -> f64 {
    let micros = f64::from(dt.nanosecond() / 1000) / 1e6;
    let resolved = match Local.from_local_datetime(&dt) {
        chrono::LocalResult::Single(t) => Some(t),
        chrono::LocalResult::Ambiguous(earlier, _) => Some(earlier),
        chrono::LocalResult::None => None,
    };
    let seconds = match resolved {
        Some(t) => t.timestamp(),
        None => {
            let before = dt - chrono::TimeDelta::hours(24);
            let offset = Local
                .from_local_datetime(&before)
                .earliest()
                .map_or(0, |t| i64::from(t.offset().local_minus_utc()));
            dt.and_utc().timestamp() - offset
        }
    };
    seconds as f64 + micros
}

#[cfg(test)]
mod tests {
    //! Port of the time parts of `tests/test_parsing.py` and `test_queue.py`.
    use super::*;

    fn dt(s: &str) -> NaiveDateTime {
        iso_to_dt(s).unwrap()
    }

    #[test]
    fn slurm_times() {
        assert_eq!(parse_slurm_time("1:30:00"), Some(5400));
        assert_eq!(parse_slurm_time("1:30"), Some(90));
        assert_eq!(parse_slurm_time("30:00"), Some(30 * 60));
        assert_eq!(parse_slurm_time("40:00"), Some(40 * 60));
        assert_eq!(parse_slurm_time("1-1:30"), Some(86400 + 3600 + 30 * 60));
        assert_eq!(parse_slurm_time("1-00:00:00"), Some(86400));
        assert_eq!(
            parse_slurm_time("2-12:30:00"),
            Some(2 * 86400 + 12 * 3600 + 30 * 60)
        );
        assert_eq!(parse_slurm_time("0-5"), Some(5 * 3600));
        for v in ["UNLIMITED", "unlimited", "INFINITE", "INFINITY", "N/A"] {
            assert_eq!(parse_slurm_time(v), None, "{v}");
        }
        assert_eq!(parse_slurm_time("30"), Some(30 * 60));
        assert_eq!(parse_slurm_time("0:00:00"), Some(0));
        assert_eq!(parse_slurm_time("  1:00:00  "), Some(3600));
        assert_eq!(parse_slurm_time("08:00:00"), Some(8 * 3600));
        assert_eq!(parse_slurm_time("7-00:00:00"), Some(7 * 86400));
        assert_eq!(parse_slurm_time("6:57:57"), Some(6 * 3600 + 57 * 60 + 57));
        for v in [
            "not-a-time",
            "1:2:3:4",
            "",
            "abc:def:ghi",
            "1-abc:00:00",
            "foo-bar",
            "::",
            "1:2:foo",
        ] {
            assert_eq!(parse_slurm_time(v), None, "{v:?}");
        }
    }

    #[test]
    fn seconds() {
        assert_eq!(format_seconds(None), "unlimited");
        assert_eq!(format_seconds(Some(45)), "45s");
        assert_eq!(format_seconds(Some(90)), "1m");
        assert_eq!(format_seconds(Some(3600 + 30 * 60)), "1h30m");
        assert_eq!(format_seconds(Some(86400)), "1d");
        assert_eq!(
            format_seconds(Some(2 * 86400 + 3 * 3600 + 15 * 60)),
            "2d3h15m"
        );
        assert_eq!(format_seconds(Some(-100)), "0s");
        assert_eq!(format_seconds(Some(0)), "0s");
        assert_eq!(format_seconds(Some(60)), "1m");
        assert_eq!(format_seconds(Some(3600)), "1h");
        assert_eq!(format_seconds(parse_slurm_time("6:57:57")), "6h57m");
        assert_eq!(format_seconds(parse_slurm_time("8:00:00")), "8h");
        assert_eq!(format_seconds(parse_slurm_time("1:02:03")), "1h2m");
    }

    #[test]
    fn iso_parsing() {
        let d = dt("2024-01-15T10:30:00");
        assert_eq!((d.year(), d.month(), d.day(), d.hour()), (2024, 1, 15, 10));
        for v in ["Unknown", "None", "N/A", "", "not-a-date", "2024/01/15"] {
            assert_eq!(iso_to_dt(v), None, "{v}");
        }
        // Same answers as Python 3.12's datetime.fromisoformat.
        let ok = [
            ("2024-01-15", "2024-01-15T00:00:00"),
            ("2024-01-15 10:30", "2024-01-15T10:30:00"),
            ("2024-01-15T10", "2024-01-15T10:00:00"),
            ("2024-01-15T10:30:00.5", "2024-01-15T10:30:00.500000"),
            ("2024-01-15T10:30:00.1234567", "2024-01-15T10:30:00.123456"),
            ("20240115T103000", "2024-01-15T10:30:00"),
            ("2024-01-15X10:30:00", "2024-01-15T10:30:00"),
            ("2024-01-15T10:30:00,5", "2024-01-15T10:30:00.500000"),
            ("2024-01-15T1030", "2024-01-15T10:30:00"),
        ];
        for (input, iso) in ok {
            assert_eq!(
                iso_to_dt(input).map(isoformat).as_deref(),
                Some(iso),
                "{input}"
            );
        }
        for bad in [
            "2024-01-15T24:00:00",
            " 2024-01-15",
            "2024-1-15",
            "2024-01-15T10:30:00.",
            "2024-02-30",
            // non-ASCII where a digit belongs (used to panic slicing inside a char)
            "2026-01-\u{2603}",
            "2026\u{2603}x",
            "20\u{e9}6-01-01",
            "2026-0\u{e9}-01",
            "\u{2603}\u{2603}\u{2603}",
            "2026-01-01T1\u{2603}",
            "2026-01-01T10:3\u{e9}",
            "2026-01-01T10:30:00.\u{2603}",
        ] {
            assert_eq!(iso_to_dt(bad), None, "{bad}");
        }
    }

    #[test]
    fn history_end() {
        let now = dt("2024-03-15T18:00:00");
        for (end, expected) in [
            ("2024-03-15T14:30:00", "today 14:30"),
            ("2024-03-13T09:05:00", "Wed 09:05"),
            ("2024-01-02T08:00:00", "Jan 02 08:00"),
            ("2023-12-31T23:59:00", "2023-12-31"),
        ] {
            assert_eq!(format_history_end(end, now), expected);
        }
        assert_eq!(format_history_end("Unknown", now), "-");
        assert_eq!(format_history_end("None", now), "-");
        assert_eq!(format_history_end("", now), "-");
        assert_eq!(
            format_history_end("some-other-format", now),
            "some-other-format"
        );
    }

    #[test]
    fn pending_start() {
        let now = dt("2026-10-06T14:30:00");
        let f = |s: &str| format_pending_start(s, now);
        assert_eq!(
            f("2026-10-06T16:40:00").as_deref(),
            Some("in 2h10m (Tue 16:40)")
        );
        assert_eq!(
            f("2026-10-12T09:15:00").as_deref(),
            Some("in 5d18h45m (Mon 09:15)")
        );
        assert_eq!(
            f("2026-10-13T09:00:00").as_deref(),
            Some("in 6d18h30m (Oct 13 09:00)")
        );
        assert_eq!(
            f("2026-11-03T08:56:51").as_deref(),
            Some("in 27d18h26m (Nov 03 08:56)")
        );
        assert_eq!(f("2026-10-06T14:05:00").as_deref(), Some("due (Tue 14:05)"));
        for v in ["N/A", "Unknown", "", "None"] {
            assert_eq!(f(v), None);
        }
    }

    #[test]
    fn eta_and_deltas() {
        let now = dt("2026-10-06T14:30:00.900000");
        assert_eq!(format_eta("2026-10-06T16:40:00", now), "2h9m");
        assert_eq!(format_eta("N/A", now), "unknown");
        assert_eq!(delta_seconds(dt("2026-10-06T14:30:00"), now), 0);
        assert_eq!(delta_seconds(dt("2026-10-06T14:29:59"), now), -1);
        assert_eq!(total_seconds(dt("2026-10-06T14:30:01"), now), 0.1);
    }
}
