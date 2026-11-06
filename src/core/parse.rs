//! Text parsing helpers from `core.py`. Owned by P2-C. All pure.

use super::{NON_TERMINAL_JOB_STATES, NULLISH_VALUES, Record};
use crate::pyfmt;
use std::collections::BTreeSet;

/// Does `\s+\S+=` match at byte `k` of `chars` (a whitespace run, then a token with an
/// `=` after its first character)?
fn next_key_starts(chars: &[(usize, char)], k: usize) -> bool {
    let mut i = k;
    if i >= chars.len() || !pyfmt::is_space(chars[i].1) {
        return false;
    }
    while i < chars.len() && pyfmt::is_space(chars[i].1) {
        i += 1;
    }
    let token_start = i;
    while i < chars.len() && !pyfmt::is_space(chars[i].1) {
        if chars[i].1 == '=' && i > token_start {
            return true;
        }
        i += 1;
    }
    false
}

/// `parse_prop_line(line)`: `Key=Value` pairs of an `scontrol --oneliner` line using the
/// regex `(\S+?)=(.*?)(?=\s+\S+=|$)` on `line.strip()`; later duplicate keys win.
/// (Values may contain spaces, e.g. `Reason=Not responding [slurm@...]`.)
pub fn parse_prop_line(line: &str) -> Record {
    let s = pyfmt::strip(line);
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let byte_at = |i: usize| chars.get(i).map_or(s.len(), |(b, _)| *b);
    let mut record = Record::new();
    let mut i = 0;
    while i < chars.len() {
        if pyfmt::is_space(chars[i].1) {
            i += 1;
            continue;
        }
        // `\S+?=`: the first `=` after at least one character, without whitespace before it.
        let mut j = i + 1;
        while j < chars.len() && !pyfmt::is_space(chars[j].1) && chars[j].1 != '=' {
            j += 1;
        }
        if j >= chars.len() || chars[j].1 != '=' {
            i = j;
            continue;
        }
        // `(.*?)(?=\s+\S+=|$)`: up to the next `key=` or the end; `.` stops at `\n`.
        let mut k = j + 1;
        while k < chars.len() && chars[k].1 != '\n' && !next_key_starts(&chars, k) {
            k += 1;
        }
        if k < chars.len() && chars[k].1 == '\n' && !next_key_starts(&chars, k) {
            // `$` only matches before a final newline; elsewhere the match fails here.
            if k + 1 != chars.len() {
                i += 1;
                continue;
            }
        }
        record.insert(
            s[byte_at(i)..byte_at(j)].to_owned(),
            s[byte_at(j + 1)..byte_at(k)].to_owned(),
        );
        i = k;
    }
    record
}

/// `parse_pipe_rows(output, headers)`: one record per non-blank line, `|`-split, missing
/// trailing fields `""`, extra fields dropped.
pub fn parse_pipe_rows(output: &str, headers: &[&str]) -> Vec<Record> {
    pyfmt::splitlines(output)
        .into_iter()
        .filter(|line| !pyfmt::strip(line).is_empty())
        .map(|line| {
            let mut parts = line.split('|');
            headers
                .iter()
                .map(|header| (header.to_string(), parts.next().unwrap_or("").to_owned()))
                .collect()
        })
        .collect()
}

/// `is_nullish(value)`: None or `strip().lower()` in [`super::NULLISH_VALUES`].
pub fn is_nullish(value: Option<&str>) -> bool {
    match value {
        None => true,
        Some(v) => NULLISH_VALUES.contains(&pyfmt::strip(v).to_lowercase().as_str()),
    }
}

/// `parse_int(value, default)`: `int(float(value))`; empty, `(null)`, `N/A` or unparseable
/// -> `default`. `nan` and `inf` (a ValueError / an OverflowError traceback in Python)
/// give `default`; huge values saturate.
pub fn parse_int(value: Option<&str>, default: i64) -> i64 {
    match value {
        None | Some("" | "(null)" | "N/A") => default,
        Some(v) => match pyfmt::parse_float(v) {
            Some(f) if f.is_finite() => f.trunc() as i64,
            _ => default,
        },
    }
}

/// `parse_mb_to_gb(value)`: `round(parse_int(value) / 1024)` with Python's banker's
/// rounding (round half to even).
pub fn parse_mb_to_gb(value: Option<&str>) -> i64 {
    py_round_div(parse_int(value, 0), 1024)
}

/// Python `round(x)` for `int / 1024` style divisions (half to even). Shared helper.
pub fn py_round_div(numerator: i64, denominator: i64) -> i64 {
    pyfmt::round_div(numerator, denominator)
}

/// `format_mem_value(value)`: `-` for empty/`(null)`/`N/A`; all digits -> `"{gb}G"`;
/// else the value unchanged.
pub fn format_mem_value(value: Option<&str>) -> String {
    match value {
        None | Some("" | "(null)" | "N/A") => "-".to_owned(),
        Some(v) if v.chars().all(|c| c.is_ascii_digit()) => {
            format!("{}G", parse_mb_to_gb(Some(v)))
        }
        Some(v) => v.to_owned(),
    }
}

/// `truncate(value, width)`: cut to `width-1` code points + `…` when longer (width<=0: as is).
pub fn truncate(value: &str, width: usize) -> String {
    if width == 0 || value.chars().count() <= width {
        return value.to_owned();
    }
    let mut cut: String = value.chars().take(width - 1).collect();
    cut.push('…');
    cut
}

/// `parse_index_blob("0-2,5")` -> `[0, 1, 2, 5]` (bad items skipped).
pub fn parse_index_blob(blob: &str) -> Vec<i64> {
    let mut ids = Vec::new();
    for part in blob.split(',') {
        let item = pyfmt::strip(part);
        if item.is_empty() {
            continue;
        }
        if let Some((start, end)) = item.split_once('-') {
            if let (Some(start), Some(end)) = (pyfmt::parse_int(start), pyfmt::parse_int(end)) {
                ids.extend(start..=end);
            }
            continue;
        }
        if let Some(id) = pyfmt::parse_int(item) {
            ids.push(id);
        }
    }
    ids
}

/// `_split_top_level(spec)`: split at commas outside brackets; parts stripped, empties dropped.
fn split_top_level(spec: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i64;
    let mut start = 0;
    for (i, c) in spec.char_indices() {
        match c {
            ',' if depth == 0 => {
                parts.push(&spec[start..i]);
                start = i + 1;
                continue;
            }
            '[' => depth += 1,
            ']' => depth -= 1,
            _ => {}
        }
    }
    parts.push(&spec[start..]);
    parts
        .into_iter()
        .map(pyfmt::strip)
        .filter(|p| !p.is_empty())
        .collect()
}

/// `_HOSTLIST_PART`: `^(?P<prefix>[^\[\]]*)\[(?P<ranges>[^\]]+)\](?P<suffix>[^\[\]]*)$`.
fn hostlist_part(part: &str) -> Option<(&str, &str, &str)> {
    let open = part.find('[')?;
    let prefix = &part[..open];
    if prefix.contains(']') {
        return None;
    }
    let rest = &part[open + 1..];
    let close = rest.find(']')?;
    let ranges = &rest[..close];
    let suffix = &rest[close + 1..];
    let suffix_ok = suffix
        .strip_suffix('\n')
        .unwrap_or(suffix)
        .chars()
        .all(|c| c != '[' && c != ']');
    (!ranges.is_empty() && !ranges.contains('[') && suffix_ok).then_some((prefix, ranges, suffix))
}

/// `expand_hostlist("gpu[101-103,105],cmp001")` -> set of names, zero-padding kept from
/// the range start (`{index:0{len(start)}d}`); a bad range keeps the part verbatim.
/// Python returns a frozenset; callers that iterate must sort ([`expand_nodelist`]).
pub fn expand_hostlist(spec: Option<&str>) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for part in split_top_level(spec.unwrap_or("")) {
        let Some((prefix, ranges, suffix)) = hostlist_part(part) else {
            names.insert(part.to_owned());
            continue;
        };
        let mut expanded = BTreeSet::new();
        for item in ranges.split(',') {
            let (start, end) = item.split_once('-').unwrap_or((item, ""));
            let bounds = (
                pyfmt::parse_int(start),
                pyfmt::parse_int(if end.is_empty() { start } else { end }),
            );
            let (Some(first), Some(last)) = bounds else {
                expanded = BTreeSet::from([part.to_owned()]);
                break;
            };
            let width = start.chars().count();
            expanded.extend((first..=last).map(|index| format!("{prefix}{index:0width$}{suffix}")));
        }
        names.extend(expanded);
    }
    names
}

/// `expand_nodelist(nodelist)`: sorted names; `""`, `(null)`, `None` -> empty.
pub fn expand_nodelist(nodelist: &str) -> Vec<String> {
    let nodelist = pyfmt::strip(nodelist);
    if matches!(nodelist, "" | "(null)" | "None") {
        return Vec::new();
    }
    expand_hostlist(Some(nodelist)).into_iter().collect()
}

/// `user_from_userid("alice(1000)")` -> `alice`.
pub fn user_from_userid(value: &str) -> &str {
    value.split('(').next().unwrap_or("")
}

/// `base_job_id("123_4")` -> `123`.
pub fn base_job_id(job_id: &str) -> &str {
    job_id.split('_').next().unwrap_or("")
}

/// `normalize_sacct_state("CANCELLED by 123")` -> `CANCELLED` (first token, `+` stripped).
pub fn normalize_sacct_state(state: &str) -> String {
    let stripped = pyfmt::strip(state);
    let token = stripped.split(' ').next().unwrap_or("");
    token.trim_end_matches('+').to_owned()
}

/// `job_state_is_finished(state)`.
pub fn job_state_is_finished(state: &str) -> bool {
    let token = normalize_sacct_state(state);
    !token.is_empty() && !NON_TERMINAL_JOB_STATES.contains(&token.as_str())
}

/// `is_success_exit_code("0:0")`.
pub fn is_success_exit_code(exit_code: &str) -> bool {
    match exit_code.split_once(':') {
        Some((status, signal)) => status == "0" && signal == "0",
        None => false,
    }
}

#[cfg(test)]
mod tests {
    //! Port of `tests/test_parsing.py` (parsers) and the expand_* parts of `test_slurm.py`
    //! and `test_fit.py`.
    use super::*;

    fn rec(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn prop_line_basics() {
        assert_eq!(
            parse_prop_line("PartitionName=gpu"),
            rec(&[("PartitionName", "gpu")])
        );
        let r = parse_prop_line("PartitionName=gpu AllowGroups=ALL State=UP");
        assert_eq!(r["PartitionName"], "gpu");
        assert_eq!(r["AllowGroups"], "ALL");
        assert_eq!(r["State"], "UP");
        let r = parse_prop_line("PartitionName=gpu TRESBillingWeights=CPU=1.0,Mem=0.1G State=UP");
        assert_eq!(r["TRESBillingWeights"], "CPU=1.0,Mem=0.1G");
        assert_eq!(
            parse_prop_line("PartitionName=gpu DenyUsers= State=UP")["DenyUsers"],
            ""
        );
        assert!(parse_prop_line("").is_empty());
        assert!(parse_prop_line("   ").is_empty());
        assert_eq!(
            parse_prop_line("  PartitionName=gpu  ")["PartitionName"],
            "gpu"
        );
    }

    #[test]
    fn prop_line_realistic() {
        let line = "PartitionName=gpusmall AllowGroups=ALL AllowAccounts=ALL \
            AllowQos=ALL AllocNodes=ALL Default=NO QoS=N/A \
            DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 \
            Hidden=NO MaxNodes=UNLIMITED MaxTime=1-00:00:00 MinNodes=0 \
            Nodes=gpu[001-100] PriorityJobFactor=1 PriorityTier=1 \
            RootOnly=NO OverSubscribe=NO State=UP TotalCPUs=4800 TotalNodes=100 \
            TRES=cpu=4800,mem=25600G,node=100";
        let r = parse_prop_line(line);
        assert_eq!(r["PartitionName"], "gpusmall");
        assert_eq!(r["MaxTime"], "1-00:00:00");
        assert_eq!(r["State"], "UP");
        assert_eq!(r["TotalNodes"], "100");
        let line = "NodeName=gpu001 Arch=x86_64 CoresPerSocket=24 CPUTot=48 CPUAlloc=0 \
            State=IDLE+COMPLETING Gres=gpu:rtx_a5000:4 RealMemory=256000 AllocMem=0 \
            Partitions=gpumicro,gpushort,gpusmall,gpu ActiveFeatures=a5000,rtx,nvlink,gpu4";
        let r = parse_prop_line(line);
        assert_eq!(r["NodeName"], "gpu001");
        assert_eq!(r["CPUTot"], "48");
        assert_eq!(r["CPUAlloc"], "0");
        assert_eq!(r["State"], "IDLE+COMPLETING");
        assert_eq!(r["Gres"], "gpu:rtx_a5000:4");
        assert_eq!(r["Partitions"], "gpumicro,gpushort,gpusmall,gpu");
    }

    #[test]
    fn pipe_rows() {
        let headers = ["JobIDRaw", "JobName", "State", "ExitCode", "End", "Reason"];
        let rows = parse_pipe_rows(
            "12345|training|COMPLETED|0:0|2024-01-01T10:00:00|None",
            &headers,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["JobIDRaw"], "12345");
        assert_eq!(rows[0]["State"], "COMPLETED");
        assert_eq!(rows[0]["ExitCode"], "0:0");
        let rows = parse_pipe_rows(
            "12345|job1|COMPLETED|0:0|2024-01-01T10:00:00|None\n12346|job2|FAILED|1:0|2024-01-02T11:00:00|None",
            &headers,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["JobIDRaw"], "12346");
        assert!(parse_pipe_rows("", &["A", "B", "C"]).is_empty());
        assert_eq!(
            parse_pipe_rows("\n\n12345|job|COMPLETED|0:0\n\n", &["A", "B", "C", "D"]).len(),
            1
        );
        let rows = parse_pipe_rows("val1|val2", &["A", "B", "C", "D"]);
        assert_eq!(
            rows[0],
            rec(&[("A", "val1"), ("B", "val2"), ("C", ""), ("D", "")])
        );
        assert_eq!(parse_pipe_rows("val1||val3", &["A", "B", "C"])[0]["B"], "");
        let output = "12345|12345|training|COMPLETED|0:0|2024-05-01T10:00:00|None\n\
            12346|12346|inference|FAILED|137:0|2024-05-02T11:30:00|None\n\
            12347|12347|data_prep|CANCELLED by 12000|0:9|2024-05-03T09:00:00|None\n";
        let headers = [
            "JobIDRaw", "JobID", "JobName", "State", "ExitCode", "End", "Reason",
        ];
        let rows = parse_pipe_rows(output, &headers);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2]["State"], "CANCELLED by 12000");
    }

    #[test]
    fn parse_int_cases() {
        assert_eq!(parse_int(Some("42"), 0), 42);
        assert_eq!(parse_int(Some("3.7"), 0), 3);
        assert_eq!(parse_int(None, 0), 0);
        assert_eq!(parse_int(None, 99), 99);
        assert_eq!(parse_int(Some(""), 0), 0);
        assert_eq!(parse_int(Some("(null)"), 0), 0);
        assert_eq!(parse_int(Some("N/A"), 0), 0);
        assert_eq!(parse_int(Some("-5"), 0), -5);
        assert_eq!(parse_int(Some("abc"), 0), 0);
        assert_eq!(parse_int(Some("abc"), 7), 7);
        assert_eq!(parse_int(Some("0"), 0), 0);
        assert_eq!(parse_int(Some("-3.7"), 0), -3);
        assert_eq!(parse_int(Some("1e3"), 0), 1000);
    }

    #[test]
    fn sacct_states() {
        assert_eq!(normalize_sacct_state("COMPLETED"), "COMPLETED");
        assert_eq!(normalize_sacct_state("CANCELLED by 12345"), "CANCELLED");
        assert_eq!(normalize_sacct_state("RUNNING+"), "RUNNING");
        assert_eq!(normalize_sacct_state("  FAILED  "), "FAILED");
        assert_eq!(normalize_sacct_state("NODE_FAIL"), "NODE_FAIL");
        for state in [
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
        ] {
            assert!(job_state_is_finished(state), "{state}");
        }
        for state in ["RUNNING", "PENDING", "SUSPENDED", "COMPLETING", ""] {
            assert!(!job_state_is_finished(state), "{state}");
        }
        assert!(job_state_is_finished("CANCELLED by 12345"));
    }

    #[test]
    fn job_ids() {
        assert_eq!(base_job_id("12345"), "12345");
        assert_eq!(base_job_id("12345_3"), "12345");
        assert_eq!(base_job_id("12345_100"), "12345");
    }

    #[test]
    fn nodelists() {
        assert_eq!(expand_nodelist("gpu001"), ["gpu001"]);
        assert_eq!(
            expand_nodelist("gpu001,gpu002,gpu003"),
            ["gpu001", "gpu002", "gpu003"]
        );
        assert!(expand_nodelist("").is_empty());
        assert!(expand_nodelist("(null)").is_empty());
        assert!(expand_nodelist("None").is_empty());
        assert_eq!(
            expand_nodelist("gpu[001-003]"),
            ["gpu001", "gpu002", "gpu003"]
        );
        let set = expand_hostlist(Some("gpu[101-103,105],cmp001"));
        assert_eq!(
            set.into_iter().collect::<Vec<_>>(),
            ["cmp001", "gpu101", "gpu102", "gpu103", "gpu105"]
        );
        assert_eq!(
            expand_hostlist(Some("gpu[060-061]"))
                .into_iter()
                .collect::<Vec<_>>(),
            ["gpu060", "gpu061"]
        );
        assert!(expand_hostlist(Some("")).is_empty());
        assert!(expand_hostlist(None).is_empty());
        assert_eq!(
            expand_hostlist(Some("n[1-2,x]a, m[9]-ib ,k[a]b[c]"))
                .into_iter()
                .collect::<Vec<_>>(),
            ["k[a]b[c]", "m9-ib", "n[1-2,x]a"]
        );
    }
}
