//! Port of `core.py` (owned by P2-C; `ui` shared with P2-A).
//!
//! Records: Python passes Slurm records around as `dict[str, str]`. Here:
//! * [`Record`] (`HashMap<String, String>`) for `scontrol --oneliner` / pipe-separated
//!   rows (partitions, job details, reservations, squeue/sacct rows). Use
//!   [`RecordExt::get_str`] for `.get(key, '')`.
//! * [`nodes::Node`]: a typed struct for the node dicts `nodes_from_json`/`node_from_props`
//!   build (always the same 13 keys).
//! * `serde_json::Value` for `scontrol --json show job` records ([`jobs::JobRecord`]).
//!
//! Submodules:
//! * [`slurm`]: `run`, `exec_replace`, parallel and prefetched read-only queries.
//! * [`fast`]: cheaper queries with the same information as `ab`'s (version-gated).
//! * [`parse`]: `parse_prop_line`, `parse_pipe_rows`, nullish/int/mem helpers, hostlists.
//! * [`timefmt`]: Slurm times, `format_seconds`, ETA/start/history-end formatting.
//! * [`gpu`]: GPU type canonicalization, labels, GRES parsing.
//! * [`users`]: current user/groups (via [`crate::hooks`]), `user_accounts`.
//! * [`nodes`]: `load_nodes`, node facts cache, `is_gpu_partition`, `node_free`.
//! * [`partitions`]: partitions, access, sorting, stats/summaries, `choose_partition`.
//! * [`jobs`]: job details/records/history/pending, dev jobs, job output.
//! * [`reservations`]: reservation loading and helpers.
//! * [`ui`]: `warn`, `hint`, `render_table`, the `*_text` styling helpers.
//!
//! Functions that may load nodes on demand (`nodes::node_facts`,
//! `nodes::is_gpu_partition(.., detect)`, `partitions::sort_partitions`) return `Result`:
//! the load itself is unchecked, so the only error is Ctrl-C (`AbError::Interrupted`).
//!
//! Unit tests: every Slurm call, the clock/user/groups/hostname hooks, settings and the
//! node facts come from the per-thread `crate::hooks::testing::Ctx` (with
//! `slurm::fake::FakeSlurm`); `nodes::testing` and `fit::records` port the builders of
//! `tests/slurm_records.py`.

pub mod fast;
pub mod gpu;
pub mod jobs;
pub mod nodes;
pub mod parse;
pub mod partitions;
pub mod reservations;
pub mod slurm;
pub mod timefmt;
pub mod ui;
pub mod users;

use std::collections::HashMap;

/// Program name in all user-facing text (Python `CLI_NAME = "ab"`).
pub const CLI_NAME: &str = "bab";
/// Slurm job name of dev jobs. Kept as `ab-dev` so `ab` and `bab` see each other's.
pub const DEV_JOB_NAME: &str = "ab-dev";
pub const DEV_FRIENDLY_NAME: &str = "dev job";
pub const DEFAULT_DEV_CPUS: i64 = 2;
pub const DEFAULT_DEV_MEM: &str = "8G";
pub const DEFAULT_DEV_TIME: &str = "08:00:00";
pub const DEFAULT_JOB_OUTPUT_TAIL_LINES: usize = 200;
pub const SACCT_HISTORY_HEADERS: [&str; 7] = [
    "JobIDRaw", "JobID", "JobName", "State", "ExitCode", "End", "Reason",
];
/// Compared after `strip().lower()`.
pub const NULLISH_VALUES: [&str; 5] = ["", "(null)", "none", "n/a", "unknown"];

pub const RUNNING_STATES: [&str; 3] = ["RUNNING", "COMPLETING", "CONFIGURING"];
/// `RUNNING_STATES | {"PENDING"}`.
pub const ACTIVE_STATES: [&str; 4] = ["RUNNING", "COMPLETING", "CONFIGURING", "PENDING"];
/// `ACTIVE_STATES | {"SUSPENDED"}`.
pub const NON_TERMINAL_JOB_STATES: [&str; 5] = [
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "SUSPENDED",
];
pub const BAD_NODE_FLAGS: [&str; 6] = ["DOWN", "DRAIN", "FAIL", "INVAL", "REBOOT", "MAINT"];

/// `STATE_STYLES`; states not listed render `white`.
pub const STATE_STYLES: [(&str, &str); 16] = [
    ("RUNNING", "bold green"),
    ("PENDING", "bold yellow"),
    ("COMPLETING", "cyan"),
    ("CONFIGURING", "cyan"),
    ("IDLE", "green"),
    ("MIXED", "yellow"),
    ("ALLOCATED", "blue"),
    ("REBOOT", "yellow"),
    ("MAINT", "magenta"),
    ("DOWN", "bold red"),
    ("FAILED", "bold red"),
    ("CANCELLED", "red"),
    ("TIMEOUT", "red"),
    ("OUT_OF_MEMORY", "bold red"),
    ("NODE_FAIL", "bold red"),
    ("PREEMPTED", "red"),
];

/// A Slurm record as Python's `dict[str, str]`.
pub type Record = HashMap<String, String>;

/// `dict.get` helpers for [`Record`].
pub trait RecordExt {
    /// `record.get(key, '')`.
    fn get_str(&self, key: &str) -> &str;
    /// `record.get(key)` (None when missing).
    fn get_opt(&self, key: &str) -> Option<&str>;
}

impl RecordExt for Record {
    fn get_str(&self, key: &str) -> &str {
        self.get(key).map(String::as_str).unwrap_or("")
    }

    fn get_opt(&self, key: &str) -> Option<&str> {
        self.get(key).map(String::as_str)
    }
}
