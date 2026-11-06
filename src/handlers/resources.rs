//! `bab info`, `bab free`, `bab queue`/`q`, `bab node` (and hidden `gpualloc`).
//! Port of `handlers/resources.py`. Owned by P3-D.
//!
//! Notes:
//! * `MAX_WIDTH = 100`: the "You" grid and the "Nodes with free GPUs" packing never get
//!   wider than `min(console.width, 100)`.
//! * `cmd_queue` asks [`super::why::queue_hints`] for bottleneck hints of pending jobs
//!   whose reason is in `fit::HINT_REASONS`.
//! * `--watch` is handled by `cli::watch::run_watched` around these functions; they must
//!   not print outside the consoles. Nothing here is cached between frames: like Python,
//!   every frame queries Slurm again; only the node facts (`core::nodes`, Python's
//!   process-global `_NODE_FACTS`) survive from one frame to the next, as in Python.
//! * Concurrency: each command starts the read-only queries it is about to make at once
//!   (`core::slurm::prefetch`; one may go unused when an earlier call dies) and evaluates
//!   them in Python's call order, so failures are reported as Python reports them.

use crate::core::gpu::{
    collect_display_gpu_types, format_gpu_type_header, gpu_type_label, parse_gpu_count,
    parse_gpu_ids, parse_node_gpu_types,
};
use crate::core::jobs::{
    display_job_id, find_dev_jobs, get_user_job_etas, get_user_jobs, job_gres_on_node, job_on_node,
    load_running_jobs_on, load_user_pending_jobs,
};
use crate::core::nodes::{
    Node, NodeFree, is_gpu_partition, load_nodes, node_facts, node_free, node_is_idle,
    node_is_mixed, node_is_usable,
};
use crate::core::parse::{
    expand_nodelist, format_mem_value, is_nullish, parse_index_blob, truncate, user_from_userid,
};
use crate::core::partitions::{
    PartitionStats, get_accessible_partitions, summarize_partitions, summarize_selected_partitions,
};
use crate::core::reservations::{
    active_reservations, load_reservations, recent_active_reservations,
    reservation_blocked_pending_jobs, reserved_nodes_map, upcoming_reservations,
    user_can_use_reservation,
};
use crate::core::slurm;
use crate::core::timefmt::{
    format_eta_now, format_pending_start, format_seconds, iso_to_dt, parse_slurm_time,
    total_seconds,
};
use crate::core::ui::{
    TableRow, gpu_free_text, hint, job_id_text, muted, node_text, partition_text, ratio_text,
    render_table, state_text, warn,
};
use crate::core::users::{accounts_argv, accounts_from, current_groups, current_user};
use crate::core::{CLI_NAME, Record, RecordExt};
use crate::error::{Result, die};
use crate::fit::HINT_REASONS;
use crate::pyfmt;
use crate::render::console::{self, PrintOptions};
use crate::render::table::{Cell, Column, Justify, Table};
use crate::render::text::Text;
use crate::render::theme;
use std::collections::{BTreeSet, HashMap};

/// Never wider than this: long lists wrap rather than span a wide terminal.
pub const MAX_WIDTH: usize = 100;

/// `--kind` of `bab free`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gpu,
    Cpu,
}

/// `job.get('NodeList') not in {'', '(null)'}` (a missing key counts as set, as in Python;
/// expanding it then gives no nodes).
fn has_nodelist(job: &Record) -> bool {
    !matches!(job.get_opt("NodeList"), Some("" | "(null)"))
}

/// `guess_node(args_node)`: the argument, this host (full or short hostname via
/// `hooks::hostname`) if it is a node, the node of your running dev job, or of your
/// first running GPU job; else `die("could not guess the node; name it: bab node <node>")`.
pub fn guess_node(node: Option<&str>) -> Result<String> {
    if let Some(node) = node.filter(|n| !n.is_empty()) {
        return Ok(node.to_owned());
    }

    let hostname = crate::hooks::hostname();
    let facts = node_facts()?;
    let short = hostname.split('.').next().unwrap_or("");
    for name in [hostname, short] {
        if facts.names.contains(name) {
            return Ok(name.to_owned());
        }
    }

    for job in find_dev_jobs(None, None)? {
        if job.get_str("JobState") == "RUNNING"
            && has_nodelist(&job)
            && let Some(first) = expand_nodelist(job.get_str("NodeList")).into_iter().next()
        {
            return Ok(first);
        }
    }

    let user_jobs: Vec<Record> = get_user_jobs(None, None)?
        .into_iter()
        .filter(|job| {
            job.get_str("JobState") == "RUNNING"
                && parse_gpu_count(job.get_opt("JOB_GRES")) > 0
                && has_nodelist(job)
        })
        .collect();
    let nodes = user_jobs
        .first()
        .map(|job| expand_nodelist(job.get_str("NodeList")))
        .unwrap_or_default();
    match nodes.into_iter().next() {
        Some(first) => Ok(first),
        None => Err(die(format!(
            "could not guess the node; name it: {CLI_NAME} node <node>"
        ))),
    }
}

/// `_your_jobs_text(user)`: `1 running (next ends in 6h35m) · 2 pending`, or `none`.
fn your_jobs_text(etas: (Vec<Option<f64>>, Vec<Option<f64>>)) -> Text {
    let (pending, running) = etas;
    let mut parts: Vec<Text> = Vec::new();
    for (deltas, word, verb) in [
        (&running, "running", "next ends"),
        (&pending, "pending", "first starts"),
    ] {
        if deltas.is_empty() {
            continue;
        }
        let mut part = Text::new(format!("{} {word}", deltas.len()));
        let soonest = deltas
            .iter()
            .flatten()
            .copied()
            .reduce(|best, delta| if delta < best { delta } else { best });
        if let Some(soonest) = soonest {
            part.append(&format!(" ({verb} in "), None);
            part.append(&format_seconds(Some(soonest.trunc() as i64)), Some("cyan"));
            part.append(")", None);
        }
        parts.push(part);
    }
    if parts.is_empty() {
        return Text::new("none");
    }
    Text::new(" · ").join(parts)
}

/// `_accounts_text(rows)`: `proj1 (default), proj2`: the accounts you can charge,
/// default first.
fn accounts_text(rows: &[Record]) -> Text {
    let default = rows
        .iter()
        .map(|row| row.get_str("DefaultAccount"))
        .find(|account| !account.is_empty())
        .unwrap_or("");
    let mut accounts: Vec<&str> = rows
        .iter()
        .map(|row| row.get_str("Account"))
        .filter(|account| !account.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    accounts.sort_by_key(|account| (*account != default, *account));
    if accounts.is_empty() {
        return muted("-");
    }
    let parts = accounts.into_iter().map(|account| {
        if account == default {
            Text::assemble([account.into(), (" (default)", "dim").into()])
        } else {
            Text::assemble([account.into(), "".into()])
        }
    });
    Text::new(", ").join(parts)
}

/// `_render_you()`: who you are on the cluster (user, accounts, QOS, jobs). Slurm
/// queries are made by the caller (`accounts` = `user_accounts(user)`, `etas` =
/// `get_user_job_etas(user)`).
fn render_you(user: &str, rows: &[Record], etas: (Vec<Option<f64>>, Vec<Option<f64>>)) {
    let qos: BTreeSet<&str> = rows
        .iter()
        .flat_map(|row| row.get_str("QOS").split(','))
        .filter(|q| !q.is_empty())
        .collect();
    let mut grid = Table::grid(0, 2);
    grid.add_column_spec(Column {
        style: theme::style("dim"),
        no_wrap: true,
        ..Column::default()
    });
    grid.add_column_spec(Column::default());
    grid.add_row(vec!["user".into(), user.into()], false);
    grid.add_row(vec!["accounts".into(), accounts_text(rows).into()], false);
    if !qos.is_empty() {
        let joined = qos.into_iter().collect::<Vec<_>>().join(", ");
        grid.add_row(vec!["qos".into(), joined.into()], false);
    }
    grid.add_row(vec!["jobs".into(), your_jobs_text(etas).into()], false);
    let out = console::stdout();
    out.print(&Text::styled("You", "bold"));
    out.print_with(
        &grid,
        PrintOptions {
            width: Some(out.width().min(MAX_WIDTH)),
            ..PrintOptions::default()
        },
    );
}

/// `_short_limit(value)`: Slurm time limit as `1h`, `2d2h`, `7d`; unparseable values as given.
fn short_limit(value: &str) -> Text {
    let seconds = if value.is_empty() {
        None
    } else {
        parse_slurm_time(value)
    };
    match seconds {
        None => {
            let lower = value.to_lowercase();
            muted(if lower.is_empty() { "-" } else { &lower })
        }
        Some(seconds) => Text::new(format_seconds(Some(seconds))),
    }
}

/// `_node_state_with_resv(state, resv_name, upcoming_resv_name)`.
fn node_state_with_resv(state: &str, resv_name: Option<&str>, upcoming: Option<&str>) -> Text {
    let mut text = state_text(state);
    if let Some(name) = resv_name.filter(|n| !n.is_empty()) {
        text.append(&format!(" [{name}]"), Some("magenta"));
    } else if let Some(name) = upcoming.filter(|n| !n.is_empty()) {
        text.append(&format!(" [{name}]"), Some("yellow"));
    }
    text
}

/// `_resv_nodes(resv)`.
fn resv_nodes(resv: &Record) -> String {
    let nodes = resv.get_str("Nodes");
    if is_nullish(Some(nodes)) {
        return "?".to_owned();
    }
    if nodes.chars().count() <= 30 {
        nodes.to_owned()
    } else {
        format!("{} nodes", resv.get_opt("NodeCnt").unwrap_or("?"))
    }
}

/// `_render_reservations()`.
fn render_reservations() -> Result<()> {
    let all_resvs = load_reservations()?;
    let now = crate::hooks::now();
    let active = recent_active_reservations(&all_resvs, now);
    let upcoming = upcoming_reservations(&all_resvs, now);
    if active.is_empty() && upcoming.is_empty() {
        return Ok(());
    }

    let user = current_user();
    let groups = current_groups();
    let mut rows: Vec<TableRow> = Vec::new();
    let tagged = active
        .iter()
        .map(|r| (r, true))
        .chain(upcoming.iter().map(|r| (r, false)));
    for (resv, is_active) in tagged {
        let when = if is_active {
            Text::assemble([
                ("now", "magenta").into(),
                (
                    format!(", ends in {}", format_eta_now(resv.get_str("EndTime"))),
                    "dim",
                )
                    .into(),
            ])
        } else {
            let start = iso_to_dt(resv.get_str("StartTime"));
            let end = iso_to_dt(resv.get_str("EndTime"));
            let length = match (start, end) {
                (Some(start), Some(end)) => format!(
                    ", for {}",
                    format_seconds(Some(total_seconds(end, start).trunc() as i64))
                ),
                _ => String::new(),
            };
            Text::assemble([
                (
                    format!("in {}", format_eta_now(resv.get_str("StartTime"))),
                    "yellow",
                )
                    .into(),
                (length, "dim").into(),
            ])
        };
        let access = if is_nullish(resv.get_opt("Users")) && is_nullish(resv.get_opt("Groups")) {
            muted("maintenance")
        } else if user_can_use_reservation(resv, user, groups) {
            Text::styled("you", "green")
        } else {
            muted("others")
        };
        rows.push(vec![
            resv.get_opt("ReservationName").unwrap_or("?").into(),
            resv_nodes(resv).into(),
            when.into(),
            access.into(),
        ]);
    }

    console::stdout().newline();
    render_table(
        &[
            ("Reservation", Justify::Left),
            ("Nodes", Justify::Left),
            ("When", Justify::Left),
            ("For", Justify::Left),
        ],
        rows,
        Some("Reservations"),
        (!upcoming.is_empty())
            .then_some("jobs on these nodes must end before the reservation starts"),
        &[],
        None,
    );

    if !active.is_empty() {
        let blocked =
            reservation_blocked_pending_jobs(&active, &load_user_pending_jobs(Some(user))?);
        if !blocked.is_empty() {
            let n = blocked.len();
            let label = if n == 1 {
                "1 of your pending jobs is".to_owned()
            } else {
                format!("{n} of your pending jobs are")
            };
            warn(format!(
                "{label} blocked by an active reservation; see `{CLI_NAME} why`"
            ));
        }
    }
    Ok(())
}

/// `_overview_row(name, limit, stats)`.
fn overview_row(name: Text, limit: Text, stats: &PartitionStats) -> TableRow {
    vec![
        name.into(),
        limit.into(),
        ratio_text(stats.idle_nodes, stats.nodes, "", "green").into(),
        gpu_free_text(stats.gpu_free, stats.gpu_usable, stats.gpu_total).into(),
        ratio_text(stats.cpu_free, stats.cpu_total, "", "green").into(),
        ratio_text(stats.mem_free_gb, stats.mem_total_gb, "G", "green").into(),
    ]
}

/// `_free_caption(some_unusable)`: the `+n` of `gpu_free_text`, explained only when shown.
fn free_caption(some_unusable: bool) -> &'static str {
    if some_unusable {
        "free/total · +n idle, but on full or drained nodes"
    } else {
        "free/total"
    }
}

/// `cmd_info(all_partitions=...)`.
pub fn cmd_info(all_partitions: bool) -> Result<()> {
    // Every query below is read-only and made unless an earlier one dies: start them now.
    let user = current_user();
    let _prefetch = slurm::prefetch(vec![
        crate::core::nodes::query_argv(),
        slurm::argv(&crate::core::partitions::ARGV),
        crate::core::jobs::user_etas_argv(user),
        slurm::argv(&crate::core::reservations::ARGV),
    ]);
    let sacctmgr = accounts_argv(user);
    let _accounts = slurm::prefetch(vec![sacctmgr.clone()]);
    let nodes = load_nodes(true)?;
    let partitions = get_accessible_partitions(all_partitions)?;
    let names: Vec<String> = partitions
        .iter()
        .map(|p| p.get_str("PartitionName").to_owned())
        .collect();
    let summaries = summarize_partitions(&names, &nodes);
    let cluster_stats = summarize_selected_partitions(&names.iter().cloned().collect(), &nodes);

    let mut rows: Vec<TableRow> = vec![overview_row(
        Text::styled("all", "bold"),
        muted("-"),
        &cluster_stats,
    )];
    let mut breaks: Vec<usize> = vec![0];
    let mut previous: Option<&str> = None;
    for (partition, name) in partitions.iter().zip(&names) {
        if let Some(prev) = previous
            && is_gpu_partition(prev, true)? != is_gpu_partition(name, true)?
        {
            breaks.push(rows.len() - 1); // between GPU and CPU partitions
        }
        previous = Some(name);
        let limit = short_limit(partition.get_str("MaxTime"));
        rows.push(overview_row(partition_text(name), limit, &summaries[name]));
    }

    // `_render_you`: sacctmgr (unchecked) and squeue (+ scontrol per job) are both made
    // unconditionally, in this order, so they may run side by side; a squeue failure is
    // reported after both are done, like Python's die from get_user_job_etas.
    let now = crate::hooks::now();
    let (accounts, etas) = slurm::join2(
        || accounts_from(&sacctmgr),
        || get_user_job_etas(Some(user), now),
    );
    let accounts = accounts?;
    let etas = etas?;
    render_you(user, &accounts, etas);

    console::stdout().newline();
    let some_unusable = std::iter::once(&cluster_stats)
        .chain(summaries.values())
        .any(|st| st.gpu_usable < st.gpu_free);
    render_table(
        &[
            ("Partition", Justify::Left),
            ("Time limit", Justify::Left),
            ("Idle nodes", Justify::Right),
            ("Free GPUs", Justify::Right),
            ("Free CPUs", Justify::Right),
            ("Free memory", Justify::Right),
        ],
        rows,
        Some("Resource overview"),
        Some(free_caption(some_unusable)),
        &breaks,
        None,
    );
    render_reservations()
}

/// `_slurm_duration(value)`: squeue's `1-00:00:00` / `6:35:00` / `15:00` as `1d` /
/// `6h35m` / `15m`; unparseable values (UNLIMITED, INVALID, ...) as given.
fn slurm_duration(value: &str) -> String {
    match parse_slurm_time(value) {
        None => value.to_owned(),
        Some(seconds) => format_seconds(Some(seconds)),
    }
}

/// `str.strip('()')`.
fn strip_parens(value: &str) -> &str {
    value.trim_matches(|c| c == '(' || c == ')')
}

/// `cmd_queue(all_users=, user=, partition=)`.
pub fn cmd_queue(all_users: bool, user: Option<&str>, partition: Option<&str>) -> Result<()> {
    let user = user.filter(|u| !u.is_empty());
    let partition = partition.filter(|p| !p.is_empty());
    let selected_user = if all_users {
        None
    } else {
        Some(user.unwrap_or_else(|| current_user()))
    };
    // %A is the raw job id, which matches `scontrol --json show job` records.
    let mut argv = vec![
        "squeue".to_owned(),
        "-h".to_owned(),
        "--format=%i|%A|%u|%j|%T|%P|%M|%L|%S|%R".to_owned(),
    ];
    if let Some(selected) = selected_user.filter(|u| !u.is_empty()) {
        argv.extend(["-u".to_owned(), selected.to_owned()]);
    }
    if let Some(partition) = partition {
        argv.extend(["-p".to_owned(), partition.to_owned()]);
    }
    // All users: some pending job almost always gets a bottleneck hint, which needs the
    // nodes and the partitions; start those now (the hinted jobs are known only later).
    let _prefetch = selected_user.is_none().then(|| {
        slurm::prefetch(vec![
            crate::core::nodes::query_argv(),
            slurm::argv(&crate::core::partitions::ARGV),
        ])
    });
    let jobs = slurm::run_parsable_rows(
        &argv,
        &[
            "jobid",
            "rawid",
            "user",
            "name",
            "state",
            "partition",
            "elapsed",
            "left",
            "start",
            "where",
        ],
        true,
    )?;
    let out = console::stdout();
    if jobs.is_empty() {
        let whose = match user {
            Some(user) if !all_users => format!(" for {user}"),
            _ => String::new(),
        };
        out.print(&muted(&format!("No jobs{whose} in the queue.")));
        return Ok(());
    }

    let show_user = selected_user.is_none();
    let is_pending = |job: &Record| job.get_str("state") == "PENDING";
    let show_start = jobs.iter().any(is_pending);
    let show_elapsed = jobs.iter().any(|job| !is_pending(job));
    let hint_ids: BTreeSet<String> = jobs
        .iter()
        .filter(|job| is_pending(job) && HINT_REASONS.contains(&strip_parens(job.get_str("where"))))
        .map(|job| job.get_str("rawid").to_owned())
        .collect();
    let (hints, hint_error) = if hint_ids.is_empty() {
        (HashMap::new(), None)
    } else {
        let result = super::why::queue_hints(&hint_ids, selected_user);
        // Python's KeyboardInterrupt is not caught by queue_hints' `except Exception`.
        slurm::check_interrupted()?;
        result
    };

    let now = crate::hooks::now();
    let mut rows: Vec<TableRow> = Vec::new();
    for job in &jobs {
        let pending = is_pending(job);
        let mut row: TableRow = vec![job_id_text(job.get_str("jobid")).into()];
        if show_user {
            row.push(job.get_str("user").into());
        }
        row.push(truncate(job.get_str("name"), 28).into());
        row.push(state_text(job.get_str("state")).into());
        row.push(partition_text(job.get_str("partition")).into());
        if show_elapsed {
            // squeue reports 0:00 for pending jobs; leave the cell empty
            row.push(if pending {
                Text::new("").into()
            } else {
                slurm_duration(job.get_str("elapsed")).into()
            });
        }
        row.push(slurm_duration(job.get_str("left")).into());
        if show_start {
            // 'in 2h10m' without the clock; '' renders as a dim dash when Slurm has
            // no estimate yet
            row.push(if pending {
                let start = format_pending_start(job.get_str("start"), now).unwrap_or_default();
                let head = start.split(" (").next().unwrap_or("").to_owned();
                Cell::Str(head)
            } else {
                Text::new("").into()
            });
        }
        let reason_or_node = if pending {
            // the bottleneck hint explains the wait better than Slurm's reason code
            let reason = hints
                .get(job.get_str("rawid"))
                .map(String::as_str)
                .filter(|h| !h.is_empty())
                .unwrap_or_else(|| strip_parens(job.get_str("where")));
            Text::styled(truncate(reason, 32), "yellow")
        } else {
            node_text(
                &truncate(job.get_str("where"), 32),
                job.get_str("partition"),
            )
        };
        row.push(reason_or_node.into());
        rows.push(row);
    }

    let mut columns: Vec<(&str, Justify)> = vec![("JobID", Justify::Left)];
    if show_user {
        columns.push(("User", Justify::Left));
    }
    columns.extend([
        ("Name", Justify::Left),
        ("State", Justify::Left),
        ("Partition", Justify::Left),
    ]);
    if show_elapsed {
        columns.push(("Elapsed", Justify::Right));
    }
    columns.push(("Time left", Justify::Right));
    if show_start {
        columns.push(("Start", Justify::Left));
    }
    columns.push(("Node/Reason", Justify::Left));
    let title = if all_users {
        "Queue · all users".to_owned()
    } else if let Some(user) = user {
        format!("Queue · {user}")
    } else {
        "Queue".to_owned()
    };
    render_table(&columns, rows, Some(&title), None, &[], None);
    if let Some(error) = hint_error {
        out.print(&muted(&format!("bottleneck hints unavailable: {error}")));
    }
    if !hints.is_empty() {
        hint(&format!(
            "`{CLI_NAME} why` shows what is short and what would start now"
        ));
    }
    Ok(())
}

/// `_FreeNode`: one node row of `bab free`: what `node_free()` says plus how to show it.
#[derive(Debug, Clone)]
struct FreeNode {
    name: String,
    state: String,
    partitions: String,
    free: NodeFree,
    idle: bool,
    /// IDLE or MIXED: can take a job now.
    schedulable: bool,
}

impl FreeNode {
    fn gpu_free(&self) -> i64 {
        self.free.gpu_free_by_type.values().sum()
    }

    fn mem_gb(&self) -> (i64, i64) {
        (
            pyfmt::round_div(self.free.mem_free_mb, 1024),
            pyfmt::round_div(self.free.mem_total_mb, 1024),
        )
    }

    fn gpu_free_of(&self, gpu_type: &str) -> i64 {
        self.free
            .gpu_free_by_type
            .get(gpu_type)
            .copied()
            .unwrap_or(0)
    }
}

/// `_free_nodes(...)`: the nodes `bab free` counts, most free GPUs, then CPUs, first.
fn free_nodes(
    nodes: &[Node],
    partition_names: &BTreeSet<&str>,
    kind: Kind,
    gpu_type: Option<&str>,
    reserved_by: &HashMap<String, &Record>,
) -> Vec<FreeNode> {
    let mut entries: Vec<FreeNode> = Vec::new();
    for node in nodes {
        let overlap: Vec<&str> = node
            .partitions
            .split(',')
            .filter(|p| partition_names.contains(p))
            .collect();
        let free = node_free(node);
        let gpu_total: i64 = free.gpu_total_by_type.values().sum();
        if overlap.is_empty() || (kind == Kind::Gpu) != (gpu_total != 0) {
            continue;
        }
        if let Some(gpu_type) = gpu_type
            && free.gpu_total_by_type.get(gpu_type).copied().unwrap_or(0) == 0
        {
            continue;
        }
        let mut state = node.state.clone();
        if !node_is_usable(node) {
            // Surface MAINT+RESERVED nodes when the user can use the reservation.
            let upper = state.to_uppercase();
            let usable_resv = reserved_by.get(&node.name).is_some_and(|resv| {
                !resv.is_empty() && user_can_use_reservation(resv, current_user(), current_groups())
            });
            if !(upper.contains("MAINT") && upper.contains("RESERVED") && usable_resv) {
                continue;
            }
            state = "RESERVED".to_owned();
        }
        let idle = node_is_idle(node);
        entries.push(FreeNode {
            name: node.name.clone(),
            state,
            partitions: overlap.join(","),
            free,
            idle,
            schedulable: idle || node_is_mixed(node),
        });
    }
    entries.sort_by(|a, b| {
        b.gpu_free()
            .cmp(&a.gpu_free())
            .then(b.free.cpu_free.cmp(&a.free.cpu_free))
            .then_with(|| a.name.cmp(&b.name))
    });
    entries
}

/// `_render_free_gpu_partitions(...)`.
fn render_free_gpu_partitions(
    names: &[String],
    summaries: &HashMap<String, PartitionStats>,
    cluster_stats: &PartitionStats,
    gpu_types: &[String],
) {
    let row = |name: Text, stats: &PartitionStats| -> TableRow {
        let get = |map: &crate::core::gpu::GpuCounts, gt: &str| map.get(gt).copied().unwrap_or(0);
        std::iter::once(Cell::from(name))
            .chain(gpu_types.iter().map(|gt| {
                gpu_free_text(
                    get(&stats.gpu_free_by_type, gt),
                    get(&stats.gpu_usable_by_type, gt),
                    get(&stats.gpu_total_by_type, gt),
                )
                .into()
            }))
            .collect()
    };
    let with_gpus: Vec<&String> = names
        .iter()
        .filter(|name| summaries[*name].gpu_total > 0)
        .collect();
    let some_unusable = std::iter::once(cluster_stats)
        .chain(with_gpus.iter().map(|name| &summaries[*name]))
        .any(|st| st.gpu_usable < st.gpu_free);
    let mut rows = vec![row(Text::styled("all", "bold"), cluster_stats)];
    rows.extend(
        with_gpus
            .iter()
            .map(|name| row(partition_text(name), &summaries[*name])),
    );
    let headers: Vec<String> = gpu_types
        .iter()
        .map(|gt| format_gpu_type_header("", gt))
        .collect();
    let mut columns: Vec<(&str, Justify)> = vec![("Partition", Justify::Left)];
    columns.extend(headers.iter().map(|h| (h.as_str(), Justify::Right)));
    render_table(
        &columns,
        rows,
        Some("Free GPUs"),
        Some(free_caption(some_unusable)),
        &[0],
        None,
    );
}

/// `_render_free_by_node(entries, gpu_type)`: nodes with GPUs a job can get right now,
/// most free first: `gpu084 (4), gpu101 (3)`.
fn render_free_by_node(entries: &[FreeNode], gpu_type: Option<&str>) {
    let mut found: Vec<(&str, i64)> = entries
        .iter()
        .filter(|e| e.schedulable)
        .map(|e| {
            let count = match gpu_type {
                Some(gt) => e.gpu_free_of(gt),
                None => e.gpu_free(),
            };
            (e.name.as_str(), count)
        })
        .filter(|(_, count)| *count > 0)
        .collect();
    found.sort_by_key(|item| std::cmp::Reverse(item.1));
    let out = console::stdout();
    out.newline();
    if found.is_empty() {
        out.print(&muted("No node can take a GPU job right now."));
        return;
    }
    out.print(&Text::styled("Nodes with free GPUs", "bold"));
    // Packed by hand: rich would break a line between a node and its count.
    let width = out.width().min(MAX_WIDTH);
    let mut line = Text::new("");
    for (i, (name, count)) in found.iter().enumerate() {
        let mut item = Text::assemble([
            node_text(name, "").into(),
            (format!(" ({count})"), "green").into(),
        ]);
        if i < found.len() - 1 {
            item.append(",", Some("dim"));
        }
        if !line.is_empty() && line.len() + 1 + item.len() > width {
            out.print(&line);
            line = Text::new("");
        }
        if !line.is_empty() {
            line.append(" ", None);
        }
        line.append_text(&item);
    }
    out.print(&line);
}

/// `_render_free_node_table(...)`: `bab free --nodes`, one row per node plus a total row.
fn render_free_node_table(
    entries: &[FreeNode],
    kind: Kind,
    gpu_types: &[String],
    resv_map: &HashMap<String, String>,
    upcoming_resv_map: &HashMap<String, String>,
) {
    let n = gpu_types.len();
    let (mut usable, mut free_gpus, mut total_gpus) = (vec![0; n], vec![0; n], vec![0; n]);
    let (mut cpu_free, mut cpu_total, mut mem_free, mut mem_total) = (0, 0, 0, 0);
    let mut rows: Vec<TableRow> = Vec::new();
    for e in entries {
        let (mem_free_gb, mem_total_gb) = e.mem_gb();
        cpu_free += e.free.cpu_free;
        cpu_total += e.free.cpu_total;
        mem_free += mem_free_gb;
        mem_total += mem_total_gb;
        let state = node_state_with_resv(
            &e.state,
            resv_map.get(&e.name).map(String::as_str),
            upcoming_resv_map.get(&e.name).map(String::as_str),
        );
        let mut row: TableRow = vec![node_text(&e.name, "").into(), state.into()];
        if kind == Kind::Cpu {
            row.push(e.partitions.clone().into());
        }
        let style = if e.idle {
            "green"
        } else if e.schedulable {
            "yellow"
        } else {
            "dim"
        };
        for (i, gt) in gpu_types.iter().enumerate() {
            let g_free = e.gpu_free_of(gt);
            let g_total = e.free.gpu_total_by_type.get(gt).copied().unwrap_or(0);
            usable[i] += if e.schedulable { g_free } else { 0 };
            free_gpus[i] += g_free;
            total_gpus[i] += g_total;
            row.push(ratio_text(g_free, g_total, "", style).into());
        }
        row.push(ratio_text(e.free.cpu_free, e.free.cpu_total, "", "green").into());
        row.push(ratio_text(mem_free_gb, mem_total_gb, "G", "green").into());
        rows.push(row);
    }

    let headers: Vec<String> = gpu_types
        .iter()
        .map(|gt| format_gpu_type_header("", gt))
        .collect();
    let mut columns: Vec<(&str, Justify)> = vec![("Node", Justify::Left), ("State", Justify::Left)];
    if kind == Kind::Cpu {
        columns.push(("Partitions", Justify::Left));
    }
    columns.extend(headers.iter().map(|h| (h.as_str(), Justify::Right)));
    columns.extend([
        ("Free CPUs", Justify::Right),
        ("Free memory", Justify::Right),
    ]);
    if !rows.is_empty() {
        let mut total: TableRow = vec![Text::styled("total", "bold").into(), Text::new("").into()];
        if kind == Kind::Cpu {
            total.push(Text::new("").into());
        }
        for i in 0..n {
            total.push(gpu_free_text(free_gpus[i], usable[i], total_gpus[i]).into());
        }
        total.push(ratio_text(cpu_free, cpu_total, "", "green").into());
        total.push(ratio_text(mem_free, mem_total, "G", "green").into());
        rows.push(total);
    }
    let some_unusable = (0..n).any(|i| usable[i] < free_gpus[i]);
    // Python's `{len(rows) - 2}` with no rows is {-2}: matches nothing.
    let breaks: Vec<usize> = rows.len().checked_sub(2).into_iter().collect();
    render_table(
        &columns,
        rows,
        Some(if kind == Kind::Cpu {
            "CPU nodes"
        } else {
            "GPU nodes"
        }),
        Some(free_caption(some_unusable)),
        &breaks,
        None,
    );
}

/// `cmd_free(kind=, show_nodes=, partition_filter=, gpu_type=)`; `gpu_type` is already
/// canonical (cli's `_normalize_gpu_type` callback).
pub fn cmd_free(
    kind: Kind,
    show_nodes: bool,
    partition_filter: Option<&str>,
    gpu_type: Option<&str>,
) -> Result<()> {
    let partition_filter = partition_filter.filter(|p| !p.is_empty());
    let gpu_type = gpu_type.filter(|g| !g.is_empty());
    let _prefetch = slurm::prefetch(vec![
        crate::core::nodes::query_argv(),
        slurm::argv(&crate::core::reservations::ARGV),
        slurm::argv(&crate::core::partitions::ARGV),
    ]);
    let nodes = load_nodes(true)?;
    // The reservation query is unchecked and nothing between it and the (checked)
    // partition query can fail, so Python always makes both: they run side by side.
    let (all_resvs, partitions) =
        slurm::join2(load_reservations, || get_accessible_partitions(false));
    let all_resvs = all_resvs?;
    let now = crate::hooks::now();
    let active_resvs = active_reservations(&all_resvs);
    let resv_map = reserved_nodes_map(&active_resvs);
    let upcoming = upcoming_reservations(&all_resvs, now);
    let upcoming_resv_map = reserved_nodes_map(&upcoming);
    let mut active_resv_by_name: HashMap<&str, &Record> = HashMap::new();
    for resv in &active_resvs {
        active_resv_by_name.insert(resv.get_str("ReservationName"), resv);
    }

    let partitions = partitions?;
    let mut names: Vec<String> = Vec::new();
    for partition in &partitions {
        let name = partition.get_str("PartitionName");
        if (kind == Kind::Gpu) == is_gpu_partition(name, true)?
            && partition_filter.is_none_or(|f| f == name)
        {
            names.push(name.to_owned());
        }
    }
    if names.is_empty() {
        if kind == Kind::Gpu && partition_filter.is_none() && !partitions.is_empty() {
            return Err(die(format!(
                "no partition you can use has GPUs; try `{CLI_NAME} free --kind cpu`"
            )));
        }
        return Err(die("no partitions matched this filter"));
    }
    let summaries = summarize_partitions(&names, &nodes);
    let cluster_stats = summarize_selected_partitions(&names.iter().cloned().collect(), &nodes);

    let mut count_maps = vec![
        &cluster_stats.gpu_total_by_type,
        &cluster_stats.gpu_free_by_type,
    ];
    count_maps.extend(names.iter().map(|name| &summaries[name].gpu_free_by_type));
    let mut gpu_types = collect_display_gpu_types(&count_maps);
    if let Some(gpu_type) = gpu_type {
        gpu_types.retain(|gt| gt == gpu_type);
    }

    let reserved_by: HashMap<String, &Record> = resv_map
        .iter()
        .filter_map(|(node, resv)| {
            active_resv_by_name
                .get(resv.as_str())
                .map(|r| (node.clone(), *r))
        })
        .collect();
    let name_set: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let entries = free_nodes(&nodes, &name_set, kind, gpu_type, &reserved_by);

    if show_nodes {
        render_free_node_table(
            &entries,
            kind,
            if kind == Kind::Gpu { &gpu_types } else { &[] },
            &resv_map,
            &upcoming_resv_map,
        );
    } else if kind == Kind::Cpu {
        // Partition overview; GPU columns are always empty there.
        let mut overview = vec![overview_row(
            Text::styled("all", "bold"),
            muted("-"),
            &cluster_stats,
        )];
        overview.extend(
            names
                .iter()
                .map(|name| overview_row(partition_text(name), muted("-"), &summaries[name])),
        );
        let rows: Vec<TableRow> = overview
            .into_iter()
            .map(|row| {
                let mut cells = row.into_iter();
                let first = cells.next().expect("name");
                let rest: Vec<Cell> = cells.collect();
                vec![first, rest[1].clone(), rest[3].clone(), rest[4].clone()]
            })
            .collect();
        render_table(
            &[
                ("Partition", Justify::Left),
                ("Idle nodes", Justify::Right),
                ("Free CPUs", Justify::Right),
                ("Free memory", Justify::Right),
            ],
            rows,
            Some("Free CPUs"),
            Some("free/total"),
            &[0],
            None,
        );
    } else {
        if !gpu_types.is_empty() {
            render_free_gpu_partitions(&names, &summaries, &cluster_stats, &gpu_types);
        }
        render_free_by_node(&entries, gpu_type);
    }

    // One warning per reservation that covers a node counted above.
    let shown: BTreeSet<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    let active_names: BTreeSet<&str> = shown
        .iter()
        .filter_map(|n| resv_map.get(*n).map(String::as_str))
        .collect();
    for resv in &active_resvs {
        if resv
            .get_opt("ReservationName")
            .is_some_and(|name| active_names.contains(name))
        {
            warn(format!(
                "{} reserves {} for {} more",
                resv.get_str("ReservationName"),
                resv_nodes(resv),
                format_eta_now(resv.get_str("EndTime"))
            ));
        }
    }
    let upcoming_names: BTreeSet<&str> = shown
        .iter()
        .filter_map(|n| upcoming_resv_map.get(*n).map(String::as_str))
        .collect();
    for resv in &upcoming {
        if resv
            .get_opt("ReservationName")
            .is_some_and(|name| upcoming_names.contains(name))
        {
            warn(format!(
                "{} reserves {} in {}; jobs there must end by then",
                resv.get_str("ReservationName"),
                resv_nodes(resv),
                format_eta_now(resv.get_str("StartTime"))
            ));
        }
    }
    Ok(())
}

/// `_ids(ids)`: sorted, comma-separated.
fn ids_text(ids: impl IntoIterator<Item = i64>) -> String {
    let sorted: BTreeSet<i64> = ids.into_iter().collect();
    sorted
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// `cmd_node(node=)` (also `gpualloc`).
pub fn cmd_node(node: Option<&str>) -> Result<()> {
    let jobs_argv = if crate::core::fast::enabled() {
        node.and_then(crate::core::jobs::node_jobs_argv)
    } else {
        Some(crate::core::jobs::job_details_argv(None))
    };
    let mut argvs = vec![crate::core::nodes::query_argv()];
    argvs.extend(jobs_argv);
    let _prefetch = slurm::prefetch(argvs);
    let nodes = load_nodes(true)?; // also primes node_facts(), so guess_node won't reload
    let node_name = guess_node(node)?;
    let Some(node_info) = nodes.iter().find(|n| n.name == node_name) else {
        return Err(die(format!("unknown node: {node_name}")));
    };

    let total_gpus = parse_gpu_count(Some(&node_info.gres));
    let mut jobs: Vec<(Vec<i64>, String, Record)> = load_running_jobs_on(&node_name)?
        .into_iter()
        .filter(|job| expand_nodelist(job.get_str("NodeList")).contains(&node_name))
        .map(|job| {
            let mut ids = parse_gpu_ids(Some(&job_gres_on_node(&job, &node_name)));
            if ids.is_empty() {
                ids = vec![total_gpus];
            }
            (ids, display_job_id(&job), job)
        })
        .collect();
    jobs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Heading: the node and what is free on it, the answer to "can I use this node?"
    let free = node_free(node_info);
    let free_gpus: i64 = free.gpu_free_by_type.values().sum();
    let usable = node_is_usable(node_info);
    let types: BTreeSet<String> = parse_node_gpu_types(Some(&node_info.gres))
        .values()
        .map(|t| gpu_type_label(t))
        .collect();
    let gpu_unit = if types.len() == 1 {
        format!(" {}", types.iter().next().expect("one type"))
    } else {
        " GPUs".to_owned()
    };
    // A drained or down node says so first: nothing on it can be scheduled.
    let state = if usable {
        Text::new("")
    } else {
        Text::assemble(["  ".into(), state_text(&node_info.state).into()])
    };
    let mut heading = Text::assemble([
        (node_name.as_str(), "bold").into(),
        state.into(),
        muted(if usable { "  free " } else { "  idle " }).into(),
    ]);
    if total_gpus != 0 {
        heading.append_text(&ratio_text(free_gpus, total_gpus, &gpu_unit, "green"));
        heading.append_text(&muted(" · "));
    }
    heading.append_text(&ratio_text(free.cpu_free, free.cpu_total, " CPUs", "green"));
    heading.append_text(&muted(" · "));
    heading.append_text(&ratio_text(
        pyfmt::round_div(free.mem_free_mb, 1024),
        pyfmt::round_div(free.mem_total_mb, 1024),
        "G mem",
        "green",
    ));
    let out = console::stdout();
    out.print(&heading);

    let mut owned: BTreeSet<i64> = BTreeSet::new();
    let mut rows: Vec<TableRow> = Vec::new();
    for (_, display_id, job) in &jobs {
        let share = job_on_node(job, &node_name);
        let gpu_ids = parse_gpu_ids(Some(&job_gres_on_node(job, &node_name)));
        owned.extend(gpu_ids.iter().copied());
        // One node: NumCPUs is exact. Several: count this node's CPU_IDs.
        let single = expand_nodelist(job.get_str("NodeList")).len() == 1;
        let cpus = if single {
            job.get_str("NumCPUs").to_owned()
        } else {
            match parse_index_blob(share.get_str("CPU_IDs")).len() {
                0 => String::new(),
                count => count.to_string(),
            }
        };
        let mut row: TableRow = vec![
            job_id_text(display_id).into(),
            user_from_userid(job.get_str("UserId")).into(),
            truncate(job.get_str("JobName"), 24).into(),
        ];
        if total_gpus != 0 {
            row.push(if gpu_ids.is_empty() {
                muted("-").into()
            } else {
                ids_text(gpu_ids.iter().copied()).into()
            });
        }
        row.push(
            if cpus.is_empty() {
                "-".to_owned()
            } else {
                cpus
            }
            .into(),
        );
        row.push(format_mem_value(share.get_opt("Mem")).into());
        row.push(format_eta_now(job.get_str("EndTime")).into());
        rows.push(row);
    }

    // GresUsed is the authoritative record of GPUs in use; a used GPU no job claims is
    // reported as "owner unknown" so we never overstate what is free.
    let used_ids: BTreeSet<i64> = parse_gpu_ids(Some(&node_info.gres_used))
        .into_iter()
        .collect();
    let used_count = total_gpus - free_gpus;
    let unknown_ids: BTreeSet<i64> = used_ids.difference(&owned).copied().collect();
    let unknown = (unknown_ids.len() as i64).max(used_count - owned.len() as i64);
    let mut notes: Vec<String> = Vec::new();
    if !rows.is_empty() && used_ids.union(&owned).count() as i64 >= used_count {
        // we know which GPUs are free
        let free_ids: Vec<i64> = (0..total_gpus)
            .filter(|i| !used_ids.contains(i) && !owned.contains(i))
            .collect();
        if !free_ids.is_empty() {
            notes.push(format!(
                "{} GPUs: {}",
                if usable { "free" } else { "idle" },
                ids_text(free_ids)
            ));
        }
    }
    if unknown != 0 {
        let which = if unknown_ids.len() as i64 == unknown {
            format!("GPUs {}", ids_text(unknown_ids.iter().copied()))
        } else {
            format!("{unknown} GPUs")
        };
        notes.push(format!("{which} in use, owner unknown"));
    }

    if rows.is_empty() {
        out.print(&muted(&format!("No jobs on {node_name}.")));
        if !notes.is_empty() {
            out.print(&muted(&notes.join(" · ")));
        }
        return Ok(());
    }
    let mut columns: Vec<(&str, Justify)> = vec![
        ("JobID", Justify::Left),
        ("User", Justify::Left),
        ("Name", Justify::Left),
    ];
    if total_gpus != 0 {
        columns.push(("GPUs", Justify::Left));
    }
    columns.extend([
        ("CPUs", Justify::Right),
        ("Memory", Justify::Right),
        ("Time left", Justify::Right),
    ]);
    let caption = notes.join(" · ");
    render_table(
        &columns,
        rows,
        Some(&format!("Jobs on {node_name}")),
        (!caption.is_empty()).then_some(caption.as_str()),
        &[],
        None,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Handler-level ports from `tests/test_gpu.py`, `test_cluster_output.py` and
    //! `test_queue.py` (whole-command output: `tests/snapshots.rs`).
    use super::*;
    use crate::core::nodes::testing::{NodeSpec, node_dict};
    use crate::core::slurm::fake::{FakeSlurm, fail};
    use crate::error::AbError;
    use crate::hooks::testing::Ctx;

    fn rec(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn free_nodes_sorted_filtered_and_reserved_nodes_for_the_user() {
        let _ctx = Ctx::default().install();
        let spec = |state: &'static str, gpus_used: i64| NodeSpec {
            state,
            gpus_used,
            ..NodeSpec::default()
        };
        let nodes = vec![
            node_dict("gpu1", spec("MIXED", 3)),
            node_dict("gpu2", spec("MIXED", 1)),
            node_dict("gpu3", spec("MAINT+RESERVED", 0)),
            node_dict("gpu4", spec("MAINT+RESERVED", 0)),
            node_dict(
                "cmp1",
                NodeSpec {
                    gres: "(null)".to_owned(),
                    ..NodeSpec::default()
                },
            ),
            node_dict(
                "gpu5",
                NodeSpec {
                    partitions: "other",
                    ..NodeSpec::default()
                },
            ),
        ];
        let resv = rec(&[("ReservationName", "mine"), ("Users", "alice")]);
        let reserved_by = HashMap::from([("gpu3".to_owned(), &resv)]);
        let entries = free_nodes(
            &nodes,
            &BTreeSet::from(["gpu"]),
            Kind::Gpu,
            None,
            &reserved_by,
        );
        let got: Vec<(&str, &str, i64)> = entries
            .iter()
            .map(|e| (e.name.as_str(), e.state.as_str(), e.gpu_free()))
            .collect();
        assert_eq!(
            got,
            [
                ("gpu3", "RESERVED", 4),
                ("gpu2", "MIXED", 3),
                ("gpu1", "MIXED", 1)
            ]
        );

        let cpu = free_nodes(
            &nodes,
            &BTreeSet::from(["gpu"]),
            Kind::Cpu,
            None,
            &reserved_by,
        );
        assert_eq!(
            cpu.iter().map(|e| e.name.as_str()).collect::<Vec<_>>(),
            ["cmp1"]
        );
        let typed = free_nodes(
            &nodes,
            &BTreeSet::from(["gpu"]),
            Kind::Gpu,
            Some("a100"),
            &reserved_by,
        );
        assert!(typed.is_empty());
    }

    #[test]
    fn queue_failure_of_squeue_stops_before_hints() {
        let fake = FakeSlurm::always(fail(1, "down"));
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        assert_eq!(
            cmd_queue(false, Some("carol"), Some("gpu")),
            Err(AbError::Exit(1))
        );
        assert_eq!(
            fake.calls(),
            [[
                "squeue",
                "-h",
                "--format=%i|%A|%u|%j|%T|%P|%M|%L|%S|%R",
                "-u",
                "carol",
                "-p",
                "gpu"
            ]]
        );
    }
}
