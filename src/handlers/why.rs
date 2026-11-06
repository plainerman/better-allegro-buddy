//! `bab why [JOBID] [--nodes] [--fix [auto|mem|cpus|gpus]] [-y]`, plus `queue_hints` used
//! by `bab q`. Port of `handlers/why.py`. Owned by P3-E.
//!
//! Notes:
//! * `--fix` confirmation: `hooks::interactive()` replaces `sys.stdin.isatty()`;
//!   `click.confirm('Apply?', default=False)` is `cli::prompt::confirm`.
//! * `_apply_fix` ends with a bare `raise SystemExit(1)` when the re-read does not match:
//!   return `Err(AbError::Exit(1))` (nothing printed by that line itself).
//! * `_start_text` uses `datetime.fromtimestamp(planned_start)` (local time):
//!   `core::timefmt::local_iso_from_epoch`.
//! * [`load_inputs`] starts its three read-only queries (jobs, nodes, partitions) at once
//!   and evaluates them in Python's order (conditional fallbacks stay sequential); with
//!   `core::fast` it asks only for the job records the caller reads.

use crate::core::CLI_NAME;
use crate::core::gpu::gpu_type_label;
use crate::core::jobs::{JobRecord, job_records_json, load_job_records};
use crate::core::nodes::{fetch_nodes, load_nodes, remember_nodes};
use crate::core::parse::{base_job_id, parse_prop_line};
use crate::core::slurm::{self, Output, RunError};
use crate::core::timefmt::{format_pending_start, local_iso_from_epoch};
use crate::core::ui::{
    TableRow, hint, job_id_text, muted, partition_text, render_table, state_text, warn,
};
use crate::core::users::current_user;
use crate::error::{AbError, Result, die, die_with_code};
use crate::fit::{
    Diagnosis, Dim, Fix, HINT_REASONS, JobRequest, MemLimits, MemMode, NodeCapacity, diagnose,
    find_record, format_free, format_mem, join_names, mem_per_cpu_limits, never_fits_reason,
    queue_hint, skip_summary,
};
use crate::pyfmt;
use crate::render::console;
use crate::render::table::{Cell, Justify};
use crate::render::text::Text;
use std::collections::{BTreeSet, HashMap};

/// `--fix` choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixChoice {
    Auto,
    Mem,
    Cpus,
    Gpus,
}

/// `Inputs`: job records, node capacities, partition memory limits.
pub type Inputs = (Vec<JobRecord>, Vec<NodeCapacity>, MemLimits);

type Ran = std::result::Result<Output, RunError>;

/// The tail of `core.load_job_records` once `scontrol --json show job` ran.
fn job_records_from(ran: Ran, strict: bool) -> Result<Vec<JobRecord>> {
    let output = ran.map_err(|err| err.die())?;
    if !output.success() {
        return Ok(Vec::new());
    }
    match job_records_json(&output.stdout) {
        Some(jobs) => Ok(jobs),
        None if strict => Err(die_with_code(
            "this needs a Slurm whose `scontrol --json show job` works",
            2,
        )),
        None => Ok(Vec::new()),
    }
}

/// The tail of `core.load_partitions` -> `mem_per_cpu_limits`.
fn limits_from(ran: Ran) -> Result<MemLimits> {
    let output = ran.map_err(|err| err.die())?;
    if !output.success() {
        return Ok(MemLimits::new());
    }
    let partitions: Vec<_> = pyfmt::splitlines(&output.stdout)
        .into_iter()
        .filter(|line| !pyfmt::strip(line).is_empty())
        .map(parse_prop_line)
        .collect();
    Ok(mem_per_cpu_limits(&partitions))
}

/// The job records a caller of [`load_inputs`] reads; the others only cost time.
#[derive(Debug, Clone, Copy)]
pub enum Wanted<'a> {
    /// `bab why JOBID`: `scontrol --json show job JOBID` (base id).
    Job(&'a str),
    /// `bab why`: the user's pending jobs.
    PendingOf(&'a str),
    /// `bab q` hints: these raw job ids, pending jobs of `user` (`None`: of anyone).
    Hints {
        ids: &'a BTreeSet<String>,
        user: Option<&'a str>,
    },
}

use crate::core::partitions::ARGV as PARTITIONS_ARGV;

/// The `squeue --json` that prints just the records `wanted` reads, in `scontrol --json
/// show job` order and shape (see [`crate::core::fast`]).
fn fast_jobs_argv(wanted: Wanted<'_>) -> Option<Vec<String>> {
    if !crate::core::fast::enabled() {
        return None;
    }
    let mut argv = slurm::argv(&["squeue", "--json", "--state=PD"]);
    match wanted {
        Wanted::Job(_) => return None,
        Wanted::PendingOf(user)
        | Wanted::Hints {
            user: Some(user), ..
        } => {
            if user.is_empty() || user.starts_with('-') {
                return None;
            }
            argv.push(format!("--user={user}"));
        }
        // Everyone's pending jobs can be thousands; only the hinted ones are read.
        Wanted::Hints { ids, user: None } => {
            let numeric = |id: &String| !id.is_empty() && id.bytes().all(|b| b.is_ascii_digit());
            if ids.is_empty() || !ids.iter().all(numeric) {
                return None;
            }
            argv.push(format!(
                "--jobs={}",
                ids.iter().map(String::as_str).collect::<Vec<_>>().join(",")
            ));
        }
    }
    Some(argv)
}

/// `load_inputs(strict=, job_id=)`: `load_job_records`, `load_nodes`, `load_partitions`,
/// evaluated in that order (strict: each may die and stop the rest). All three queries
/// start at once ([`slurm::prefetch`]); with [`crate::core::fast`] the jobs come from a
/// filtered `squeue --json` and the nodes from the text form, falling back to Python's
/// queries when one fails (or, for `Ids`, finds none of the jobs).
pub fn load_inputs(strict: bool, wanted: Wanted<'_>) -> Result<Inputs> {
    let job_id = match wanted {
        Wanted::Job(id) => Some(id).filter(|id| !id.is_empty()),
        _ => None,
    };
    let mut job_argv = slurm::argv(&["scontrol", "--json", "show", "job"]);
    job_argv.extend(job_id.map(str::to_owned));
    let fast_argv = fast_jobs_argv(wanted);
    let _prefetch = slurm::prefetch(vec![
        fast_argv.clone().unwrap_or_else(|| job_argv.clone()),
        crate::core::nodes::query_argv(),
        slurm::argv(&PARTITIONS_ARGV),
    ]);
    let mut records = None;
    if let Some(argv) = &fast_argv {
        let output = slurm::run(argv, false)?;
        if output.success() {
            records = job_records_json(&output.stdout).filter(|jobs| match wanted {
                // the pending jobs `bab q` just listed; one gone: ask like Python
                Wanted::Hints { ids, .. } => ids.iter().all(|id| {
                    jobs.iter()
                        .any(|job| job.get("job_id").is_some_and(|v| pyfmt::json_str(v) == *id))
                }),
                _ => true,
            });
        }
    }
    let records = match records {
        Some(records) => records,
        None => job_records_from(
            slurm::try_run(&job_argv, strict && job_id.is_none()),
            strict,
        )?,
    };
    slurm::check_interrupted()?;
    let nodes = fetch_nodes(strict)?;
    remember_nodes(&nodes);
    let nodes = nodes.iter().map(NodeCapacity::from_node).collect();
    let limits = limits_from(slurm::try_run(&PARTITIONS_ARGV, strict))?;
    slurm::check_interrupted()?;
    Ok((records, nodes, limits))
}

/// `queue_hints(job_ids)` -> (raw job id -> hint, error text). Never fails: the hints
/// decorate `bab q` and must never break it. Python turns unexpected exceptions into
/// `"<ExceptionType>: <message>"`; the Rust port of the record parsing is total, so the
/// only error left is Ctrl-C (Python's KeyboardInterrupt), which yields no hints here:
/// callers notice it with `core::slurm::check_interrupted()`.
pub fn queue_hints(
    job_ids: &BTreeSet<String>,
    user: Option<&str>,
) -> (HashMap<String, String>, Option<String>) {
    let wanted = Wanted::Hints { ids: job_ids, user };
    let Ok((records, nodes, limits)) = load_inputs(false, wanted) else {
        return (HashMap::new(), None);
    };
    if records.is_empty() || nodes.is_empty() {
        return (
            HashMap::new(),
            Some("scontrol returned no job or node data".to_owned()),
        );
    }
    let mut hints = HashMap::new();
    for record in &records {
        let job_id = record
            .get("job_id")
            .map_or_else(String::new, pyfmt::json_str);
        if job_ids.contains(&job_id) {
            let request = JobRequest::from_record(record);
            hints.insert(job_id, queue_hint(&diagnose(&request, &nodes, &limits)));
        }
    }
    (hints, None)
}
// ---------------------------------------------------------------------------
// Wording
// ---------------------------------------------------------------------------

/// `ROW_LABELS`.
fn row_label(dim: Dim) -> &'static str {
    match dim {
        Dim::Gpus => "GPU",
        Dim::Cpus => "CPU",
        Dim::Mem => "Mem",
    }
}

/// `PROSE_LABELS`.
fn prose_label(dim: Dim) -> &'static str {
    match dim {
        Dim::Gpus => "GPUs",
        Dim::Cpus => "CPUs",
        Dim::Mem => "memory",
    }
}

fn is_hint_reason(reason: &str) -> bool {
    HINT_REASONS.contains(&reason)
}

fn gpu_label(request: &JobRequest) -> String {
    match request.gpu_kind.as_deref() {
        Some(kind) if !kind.is_empty() => gpu_type_label(kind),
        _ => "GPU".to_owned(),
    }
}

fn mem_mode_upper(mode: MemMode) -> &'static str {
    match mode {
        MemMode::Node => "NODE",
        MemMode::Cpu => "CPU",
        MemMode::Gpu => "GPU",
    }
}

/// `describe_request(request, detailed=)`: `4x A5000 · 16 CPU · 40G`; detailed adds the
/// per-CPU/per-GPU settings.
pub fn describe_request(request: &JobRequest, detailed: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    if request.gpus != 0 {
        parts.push(format!("{}x {}", request.gpus, gpu_label(request)));
    }
    let mut cpus = format!("{} CPU", request.cpus);
    if detailed && let Some(per_gpu) = request.cpus_per_gpu {
        cpus += &format!(" ({per_gpu}/GPU)");
    }
    parts.push(cpus);
    if request.whole_node_mem() {
        parts.push("all memory".to_owned());
    } else {
        let mut mem = format_mem(request.amount().mem);
        if detailed && request.mem_mode != MemMode::Node {
            mem += &format!(
                " ({}/{})",
                format_mem(request.mem_value),
                mem_mode_upper(request.mem_mode)
            );
        }
        parts.push(mem);
    }
    let mut text = parts.join(" · ");
    if request.node_count > 1 {
        text += &format!(" on each of {} nodes", request.node_count);
    }
    text
}

/// `_node_names(names, planned, limit)`: `*` on the nodes backfill has set aside.
fn node_names(names: &[String], planned: &BTreeSet<String>, limit: usize) -> String {
    let marked: Vec<String> = names
        .iter()
        .map(|name| {
            if planned.contains(name) {
                format!("{name}*")
            } else {
                name.clone()
            }
        })
        .collect();
    join_names(&marked, limit)
}

/// `fix_label(fix)`: `--mem 56G → 32G` or `GPUs 4 → 2 (→ 8 CPU, 20G)`.
pub fn fix_label(fix: &Fix) -> String {
    let show = |value: i64| {
        if fix.knob == Dim::Mem {
            format_mem(value)
        } else {
            value.to_string()
        }
    };
    let mut label = format!(
        "{} {} → {}",
        fix.request.knob_flag(fix.knob),
        show(fix.old),
        show(fix.new)
    );
    let after = fix.request.amount();
    if fix.knob == Dim::Gpus
        && (fix.request.cpus_per_gpu.is_some() || fix.request.mem_mode == MemMode::Gpu)
    {
        label += &format!(" (→ {} CPU, {})", after.cpus, format_mem(after.mem));
    } else if fix.knob == Dim::Cpus && fix.request.mem_mode == MemMode::Cpu {
        label += &format!(" (→ {})", format_mem(after.mem));
    }
    label
}

/// `_and(words)`: `a, b and c`.
fn and_words(words: &[&str]) -> String {
    match words {
        [] => String::new(),
        [only] => (*only).to_owned(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// `_start_text(request)`: `in 2h10m (Tue 12:10)` from Slurm's backfill estimate.
fn start_text(request: &JobRequest) -> Option<String> {
    let start = request.planned_start?;
    format_pending_start(&local_iso_from_epoch(start), crate::hooks::now())
}

/// `_why(diagnosis)`: why the job waits, plus Slurm's start estimate.
fn why_text(diagnosis: &Diagnosis) -> Text {
    let request = &diagnosis.request;
    if diagnosis.rows.is_empty() {
        let skipped = skip_summary(&diagnosis.skipped, false);
        let place = if skipped.is_empty() {
            format!("none in {}", request.partitions.join(","))
        } else {
            skipped
        };
        return Text::styled(format!("no usable nodes: {place}"), "red");
    }
    if diagnosis.never_fits() {
        return Text::styled(
            format!(
                "never fits: {}; waiting will not help",
                never_fits_reason(diagnosis)
            ),
            "red",
        );
    }
    let mut text = if diagnosis.fits_now() {
        let mut text = Text::styled(
            format!("fits now on {}", join_names(&diagnosis.fits(), 1)),
            "green",
        );
        if request.reason == "Priority" {
            text.append(", held for higher-priority jobs", None);
        }
        text
    } else {
        let short: Vec<&str> = diagnosis
            .dims()
            .into_iter()
            .filter(|&dim| diagnosis.short_count(dim) != 0)
            .map(prose_label)
            .collect();
        Text::assemble([
            "waiting for ".into(),
            (and_words(&short), "yellow").into(),
            " to free up".into(),
        ])
    };
    if let Some(start) = start_text(request).filter(|s| !s.is_empty()) {
        // format_pending_start says 'due (Tue 12:10)' once the estimate passed
        let verb = if start.starts_with("due") {
            "start"
        } else {
            "starts"
        };
        text.append(&format!(" · {verb} {start}"), None);
        if !request.planned_nodes.is_empty() {
            text.append(&format!(" on {}", request.planned_nodes), None);
        }
    }
    text
}

/// `_runnable_fix(diagnosis)`: the fix plain `--fix` applies (or `--fix gpus`).
fn runnable_fix(diagnosis: &Diagnosis) -> Option<(&Fix, &'static str)> {
    let fix = diagnosis.lead_fix()?;
    // A held job, a dependency or a QOS limit still blocks it after the change.
    if fix.command.is_none() || !is_hint_reason(&diagnosis.request.reason) {
        return None;
    }
    Some((
        fix,
        if fix.knob == Dim::Gpus {
            "--fix gpus"
        } else {
            "--fix"
        },
    ))
}

fn fix_command(diagnosis: &Diagnosis) -> Option<String> {
    let (_, flag) = runnable_fix(diagnosis)?;
    Some(format!(
        "{CLI_NAME} why {} {flag}",
        diagnosis.request.update_id
    ))
}

/// `_change(diagnosis)`: the one change that starts the job now, or that there is none.
fn change_text(diagnosis: &Diagnosis) -> Option<Text> {
    if diagnosis.rows.is_empty() || diagnosis.fits_now() {
        return None;
    }
    let Some(fix) = diagnosis.lead_fix() else {
        return Some(muted("no single change starts it now"));
    };
    let blocked = !is_hint_reason(&diagnosis.request.reason);
    let lead = if blocked {
        ("would fit now with ", "yellow")
    } else {
        ("starts now with ", "bold green")
    };
    let mut text = Text::assemble([lead.into(), (fix_label(fix), "bold cyan").into()]);
    if fix.command.is_none() {
        text.append(" (change the script and resubmit)", Some("dim"));
    }
    Some(text)
}

/// `_render_summary(diagnosis, detailed=)`.
fn render_summary(diagnosis: &Diagnosis, detailed: bool) {
    let out = console::stdout();
    let request = &diagnosis.request;
    let mut head = Text::assemble([
        job_id_text(&request.display_id).into(),
        "  ".into(),
        partition_text(&request.partitions.join(",")).into(),
        format!("  {}", describe_request(request, detailed)).into(),
    ]);
    if request.pending_tasks > 1 {
        head.append(&format!("  ({} tasks)", request.pending_tasks), Some("dim"));
    }
    out.print(&head);
    let mut why = why_text(diagnosis);
    if !is_hint_reason(&request.reason) {
        why = Text::assemble([
            (request.reason.as_str(), "yellow").into(),
            " first, then ".into(),
            why.into(),
        ]);
    }
    out.print(&Text::assemble(["  ".into(), why.into()]));
    if let Some(change) = change_text(diagnosis) {
        out.print(&Text::assemble(["  ".into(), change.into()]));
        let command = if detailed {
            None
        } else {
            fix_command(diagnosis)
        };
        if let Some(command) = command {
            out.print(&muted(&format!(
                "  ⓘ `{command}` changes the job so it starts now"
            )));
        }
    }
}

// ---------------------------------------------------------------------------
// bab why (overview)
// ---------------------------------------------------------------------------

/// `record.get(key)` compared with `==` against a string.
fn field_is(record: &JobRecord, key: &str, value: &str) -> bool {
    record.get(key).and_then(|v| v.as_str()) == Some(value)
}

/// `'PENDING' in (record.get('job_state') or [])`.
fn has_pending_state(record: &JobRecord) -> bool {
    match record.get("job_state") {
        Some(serde_json::Value::Array(items)) => {
            items.iter().any(|s| s.as_str() == Some("PENDING"))
        }
        Some(serde_json::Value::String(s)) => s.contains("PENDING"),
        _ => false,
    }
}

fn render_overview(records: &[JobRecord], nodes: &[NodeCapacity], limits: &MemLimits) {
    let user = current_user();
    let requests: Vec<JobRequest> = records
        .iter()
        .filter(|record| {
            field_is(record, "user_name", user)
                && has_pending_state(record)
                && HINT_REASONS
                    .iter()
                    .any(|r| field_is(record, "state_reason", r))
        })
        .map(JobRequest::from_record)
        .collect();
    if requests.is_empty() {
        console::stdout().print(&muted(
            "None of your pending jobs is waiting for resources.",
        ));
        return;
    }
    for request in &requests {
        render_summary(&diagnose(request, nodes, limits), false);
    }
}

// ---------------------------------------------------------------------------
// bab why JOBID
// ---------------------------------------------------------------------------

fn render_other_fixes(diagnosis: &Diagnosis) {
    let shown = diagnosis.lead_fix();
    let others: Vec<String> = diagnosis
        .fixes
        .iter()
        .filter(|fix| !shown.is_some_and(|s| std::ptr::eq(*fix, s)))
        .map(fix_label)
        .collect();
    if !others.is_empty() && !diagnosis.fits_now() {
        console::stdout().print(&muted(&format!("  or {}", others.join(", or "))));
    }
}

fn render_slots(diagnosis: &Diagnosis) {
    let request = &diagnosis.request;
    let mut columns: Vec<(&str, Justify)> = Vec::new();
    if request.gpus != 0 {
        columns.push(("GPUs", Justify::Left));
    }
    columns.extend([
        ("CPUs", Justify::Right),
        ("Memory", Justify::Right),
        ("Nodes", Justify::Left),
    ]);
    let planned = diagnosis.planned();
    let mut rows: Vec<TableRow> = Vec::new();
    for slot in &diagnosis.slots {
        let mut row: TableRow = Vec::new();
        if request.gpus != 0 {
            row.push(Cell::Str(format!(
                "{}x {}",
                slot.free.gpus,
                gpu_label(request)
            )));
        }
        row.push(Cell::Str(slot.free.cpus.to_string()));
        row.push(Cell::Str(format_mem(slot.free.mem)));
        row.push(Cell::Str(node_names(&slot.nodes, &planned, 4)));
        rows.push(row);
    }
    let any_planned = diagnosis
        .slots
        .iter()
        .any(|slot| slot.nodes.iter().any(|name| planned.contains(name)));
    let caption = any_planned.then_some("* held for another job");
    console::stdout().newline();
    render_table(
        &columns,
        rows,
        Some("Biggest free slots"),
        caption,
        &[],
        None,
    );
}

fn render_nodes(diagnosis: &Diagnosis) {
    let out = console::stdout();
    let dims = diagnosis.dims();
    let mut columns: Vec<(&str, Justify)> = vec![("Node", Justify::Left), ("State", Justify::Left)];
    columns.extend(dims.iter().map(|&dim| (row_label(dim), Justify::Right)));
    let mut ordered: Vec<_> = diagnosis.rows.iter().collect();
    ordered.sort_by_key(|row| {
        (
            row.short().len(),
            -row.free.gpus,
            -row.free.cpus,
            -row.free.mem,
        )
    });
    let mut rows: Vec<TableRow> = Vec::new();
    for fit in ordered {
        let short = fit.short();
        let mut row: TableRow = vec![
            Cell::Str(fit.node.name.clone()),
            Cell::Str(fit.node.state.clone()),
        ];
        for &dim in &dims {
            let mut free = format_free(dim, fit.free.get(dim));
            let total = format_free(dim, fit.total.get(dim));
            if free.ends_with('G') && total.ends_with('G') {
                free.pop(); // 73/89G, as in bab free
            }
            let style = if short.contains(&dim) { "red" } else { "" };
            row.push(Cell::Text(Text::styled(format!("{free}/{total}"), style)));
        }
        rows.push(row);
    }
    out.newline();
    render_table(
        &columns,
        rows,
        Some("Eligible nodes"),
        Some("free/total, red is short"),
        &[],
        None,
    );
    let unavailable = diagnosis.unavailable();
    if !unavailable.is_empty() {
        out.print(&muted(&format!(
            "Not counted: {}.",
            skip_summary(&unavailable, false)
        )));
    }
    if diagnosis.request.features.is_none() {
        out.print(&muted("Feature constraint not checked; every node counts."));
    }
}

fn render_detail(diagnosis: &Diagnosis, show_nodes: bool) {
    render_summary(diagnosis, true);
    if diagnosis.rows.is_empty() {
        return;
    }
    render_other_fixes(diagnosis);
    if !diagnosis.fits_now() && diagnosis.fixes.is_empty() && !diagnosis.slots.is_empty() {
        render_slots(diagnosis);
    }
    if show_nodes {
        render_nodes(diagnosis);
    }
    if let Some(command) = fix_command(diagnosis)
        && !diagnosis.fits_now()
    {
        hint(&format!("`{command}` changes the job so it starts now"));
    }
}

// ---------------------------------------------------------------------------
// bab why JOBID --fix
// ---------------------------------------------------------------------------

/// `_resources(request)`: the fields a fix changes.
fn resources(request: &JobRequest) -> (i64, i64, Option<i64>, MemMode, i64) {
    (
        request.gpus,
        request.cpus,
        request.cpus_per_gpu,
        request.mem_mode,
        request.mem_value,
    )
}

fn choice_name(choice: FixChoice) -> &'static str {
    match choice {
        FixChoice::Auto => "auto",
        FixChoice::Mem => "mem",
        FixChoice::Cpus => "cpus",
        FixChoice::Gpus => "gpus",
    }
}

/// `_choose_fix(diagnosis, choice)`: only fixes with an in-place command qualify.
fn choose_fix(diagnosis: &Diagnosis, choice: FixChoice) -> Result<&Fix> {
    let request = &diagnosis.request;
    let fix = match choice {
        FixChoice::Auto => {
            let fix = diagnosis.lead_fix();
            if let Some(fix) = fix
                && fix.knob == Dim::Gpus
                && fix.command.is_some()
            {
                return Err(die(format!(
                    "only fewer GPUs get {} started in place; \
                     if your script copes with that, rerun with --fix gpus",
                    request.display_id
                )));
            }
            fix
        }
        FixChoice::Mem => diagnosis.fix_for(Dim::Mem),
        FixChoice::Cpus => diagnosis.fix_for(Dim::Cpus),
        FixChoice::Gpus => diagnosis.fix_for(Dim::Gpus),
    };
    let Some(fix) = fix else {
        let what = if choice == FixChoice::Auto {
            "single"
        } else {
            choice_name(choice)
        };
        return Err(die(format!(
            "no {what} change gets {} started right now",
            request.display_id
        )));
    };
    if fix.command.is_none() {
        return Err(die(format!(
            "{} would start {}, but Slurm cannot change that in place; \
             change the job script and resubmit",
            fix_label(fix),
            request.display_id
        )));
    }
    Ok(fix)
}

/// `_report_fix(fix, limits)`: re-read the job; true when Slurm now holds the request.
fn report_fix(fix: &Fix, limits: &MemLimits) -> Result<bool> {
    let out = console::stdout();
    let expected = &fix.request;
    let records = load_job_records(Some(&expected.job_id), false)?;
    if records.is_empty() {
        warn(format!(
            "could not re-read the job; check `scontrol show job {}`",
            expected.update_id
        ));
        return Ok(false);
    }
    let Some(record) = find_record(&records, &expected.job_id) else {
        out.print(&Text::new(format!(
            "Updated; {} is no longer listed, it may have started.",
            expected.display_id
        )));
        return Ok(true);
    };
    let updated = JobRequest::from_record(record);
    if updated.state != "PENDING" {
        out.print(&Text::assemble([
            format!("Updated; {} is now ", updated.display_id).into(),
            state_text(&updated.state).into(),
        ]));
        return Ok(true);
    }
    if resources(&updated) != resources(expected) {
        warn(format!(
            "Slurm now reports {}, expected {}; check `scontrol show job {}`",
            describe_request(&updated, true),
            describe_request(expected, true),
            updated.update_id
        ));
        return Ok(false);
    }
    let nodes: Vec<NodeCapacity> = load_nodes(false)?
        .iter()
        .map(NodeCapacity::from_node)
        .collect();
    let after = diagnose(&updated, &nodes, limits);
    if after.fits_now() {
        let count = after.fits().len();
        out.print(&Text::styled(
            format!(
                "Updated; {} now asks for {} and fits on {count} {}.",
                updated.display_id,
                describe_request(&updated, false),
                if count == 1 { "node" } else { "nodes" }
            ),
            "green",
        ));
    } else {
        warn(format!(
            "updated to {}, but the cluster changed and it no longer fits; \
             run `{CLI_NAME} why {}` again",
            describe_request(&updated, false),
            updated.update_id
        ));
    }
    Ok(true)
}

fn apply_fix(
    diagnosis: &Diagnosis,
    choice: FixChoice,
    assume_yes: bool,
    limits: &MemLimits,
) -> Result<()> {
    let out = console::stdout();
    let request = &diagnosis.request;
    if request.user != current_user() {
        return Err(die(format!(
            "{} belongs to {}; --fix only changes your own jobs",
            request.display_id, request.user
        )));
    }
    if request.het {
        return Err(die("--fix does not change heterogeneous jobs"));
    }
    if request.node_count > 1 {
        return Err(die(format!(
            "--fix does not change multi-node jobs; use the commands from `{CLI_NAME} why {}`",
            request.update_id
        )));
    }
    if !is_hint_reason(&request.reason) {
        return Err(die(format!(
            "{} waits for {}, not resources; --fix would not start it",
            request.display_id, request.reason
        )));
    }
    if diagnosis.fits_now() {
        out.print(&Text::new(format!(
            "{} already fits on {}; nothing to change.",
            request.display_id,
            join_names(&diagnosis.fits(), 1)
        )));
        return Ok(());
    }

    let fix = choose_fix(diagnosis, choice)?;
    let command = fix.command.clone().unwrap_or_default();
    out.print(&Text::assemble([
        (fix_label(fix), "cyan").into(),
        format!(" starts {} now. This runs:", request.display_id).into(),
    ]));
    out.print(&Text::styled(
        format!("  $ {}", pyfmt::shlex_join(&command)),
        "bold",
    ));
    let mut note = "Only the allocation changes, not the job script".to_owned();
    if request.pending_tasks > 1 {
        note += &format!("; applies to all {} pending tasks", request.pending_tasks);
    }
    out.print(&muted(&format!("{note}.")));

    if !assume_yes {
        if !crate::hooks::interactive() {
            return Err(die(
                "refusing to change the job without confirmation; rerun with --yes",
            ));
        }
        if !crate::cli::prompt::confirm("Apply?", false)? {
            out.print_str("Left unchanged.", None);
            return Ok(());
        }
    }

    let proc = slurm::run(&command, false)?;
    if !proc.success() {
        // An array update also reports its running tasks, which cannot change, so a
        // non-zero exit can still have updated the pending ones: re-read and compare.
        let detail = if !proc.stderr.is_empty() {
            &proc.stderr
        } else {
            &proc.stdout
        };
        let detail = pyfmt::strip(detail);
        if detail.is_empty() {
            warn(format!("scontrol: exit code {}", proc.status));
        } else {
            warn(format!("scontrol: {detail}"));
        }
    }
    if !report_fix(fix, limits)? {
        return Err(AbError::Exit(1));
    }
    Ok(())
}

/// `cmd_why(jobid=, show_nodes=, fix=, assume_yes=)`.
pub fn cmd_why(
    jobid: Option<&str>,
    show_nodes: bool,
    fix: Option<FixChoice>,
    assume_yes: bool,
) -> Result<()> {
    if fix.is_some() && jobid.is_none() {
        return Err(die(format!(
            "--fix needs a job id, e.g. {CLI_NAME} why 65177 --fix"
        )));
    }
    if assume_yes && fix.is_none() {
        return Err(die("--yes only applies together with --fix"));
    }

    // %A_[...] and 65177_3 both live under the array's base id
    let wanted = match jobid {
        Some(jobid) => Wanted::Job(base_job_id(jobid)),
        None => Wanted::PendingOf(current_user()),
    };
    let (records, nodes, limits) = load_inputs(true, wanted)?;
    let Some(jobid) = jobid else {
        render_overview(&records, &nodes, &limits);
        return Ok(());
    };

    let Some(record) = find_record(&records, jobid) else {
        return Err(die(format!(
            "job {jobid} not found; finished jobs drop out of scontrol soon after they end"
        )));
    };
    let request = JobRequest::from_record(record);
    if request.state != "PENDING" {
        console::stdout().print(&Text::assemble([
            format!("{} is ", request.display_id).into(),
            state_text(&request.state).into(),
            ", not waiting.".into(),
        ]));
        return Ok(());
    }
    let diagnosis = diagnose(&request, &nodes, &limits);
    if let Some(choice) = fix {
        return apply_fix(&diagnosis, choice, assume_yes, &limits);
    }
    render_detail(&diagnosis, show_nodes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, fail, ok};
    use crate::fit::records::array_job;
    use crate::hooks::testing::Ctx;
    use serde_json::json;

    fn sorted(calls: Vec<Vec<String>>) -> Vec<String> {
        let mut calls: Vec<String> = calls.iter().map(|argv| argv.join(" ")).collect();
        calls.sort();
        calls
    }

    #[test]
    fn strict_load_stops_at_the_first_failure() {
        // The partition (and node) queries already started are read-only and go unused.
        let fake = FakeSlurm::always(fail(1, "down"));
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        assert_eq!(
            load_inputs(true, Wanted::PendingOf("alice")),
            Err(AbError::Exit(1))
        );
        assert_eq!(
            sorted(fake.calls()),
            [
                "scontrol --json show job",
                "scontrol --json show nodes",
                "scontrol --oneliner show partition",
            ]
        );

        let fake = FakeSlurm::always(fail(1, "down"));
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        assert_eq!(
            load_inputs(true, Wanted::Job("65177")),
            Err(AbError::Exit(1))
        );
        assert_eq!(
            sorted(fake.calls()),
            [
                "scontrol --json show job 65177",
                "scontrol --json show nodes",
                "scontrol --oneliner show node",
                "scontrol --oneliner show partition",
            ]
        );
    }

    #[test]
    fn fast_queries_and_their_fallback() {
        let node = "NodeName=n1 CPUAlloc=0 CPUTot=4 AvailableFeatures=(null) \
            ActiveFeatures=(null) Gres=(null) RealMemory=1000 AllocMem=0 State=IDLE \
            Owner=N/A BootTime=None CurrentWatts=0 AveWatts=0\n";
        for squeue_ok in [true, false] {
            let jobs = json!({"jobs": [array_job()]}).to_string();
            let fake = FakeSlurm::new(move |argv| {
                Some(match argv.join(" ").as_str() {
                    "squeue --json --state=PD --user=alice" if squeue_ok => ok(&jobs),
                    "scontrol --json show job" => ok(&jobs),
                    "scontrol --oneliner --details show node" => ok(node),
                    "scontrol --oneliner show partition" => ok(""),
                    _ => fail(1, "down"),
                })
            });
            let _ctx = Ctx::default()
                .slurm_version((24, 11))
                .slurm(fake.clone())
                .install();
            let (records, nodes, _) = load_inputs(true, Wanted::PendingOf("alice")).unwrap();
            assert_eq!((records.len(), nodes.len()), (1, 1));
            let mut want = vec![
                "scontrol --oneliner --details show node",
                "scontrol --oneliner show partition",
                "squeue --json --state=PD --user=alice",
            ];
            if !squeue_ok {
                want.insert(0, "scontrol --json show job");
            }
            assert_eq!(sorted(fake.calls()), want);
        }
    }

    #[test]
    fn queue_hints_without_data_say_why() {
        let jobs = json!({"jobs": [array_job()]}).to_string();
        let fake = FakeSlurm::new(move |argv| {
            Some(match argv.join(" ").as_str() {
                "scontrol --json show job" => ok(&jobs),
                _ => fail(1, "down"),
            })
        });
        let _ctx = Ctx::default().slurm(fake.clone()).install();
        let ids = BTreeSet::from(["65177".to_owned()]);
        let (hints, error) = queue_hints(&ids, None);
        assert!(hints.is_empty());
        assert_eq!(
            error.as_deref(),
            Some("scontrol returned no job or node data")
        );
        assert_eq!(
            sorted(fake.calls()),
            [
                "scontrol --json show job",
                "scontrol --json show nodes",
                "scontrol --oneliner show node",
                "scontrol --oneliner show partition",
            ]
        );
    }

    #[test]
    fn wording() {
        let request = JobRequest::from_record(&array_job());
        assert_eq!(describe_request(&request, false), "4x A5000 · 16 CPU · 40G");
        assert_eq!(
            describe_request(&request, true),
            "4x A5000 · 16 CPU (4/GPU) · 40G (10G/GPU)"
        );
        let fix = Fix {
            knob: Dim::Gpus,
            old: 4,
            new: 2,
            request: request.with_knob(Dim::Gpus, 2),
            nodes: Vec::new(),
            command: None,
        };
        assert_eq!(fix_label(&fix), "GPUs 4 → 2 (→ 8 CPU, 20G)");
        assert_eq!(and_words(&[]), "");
        assert_eq!(and_words(&["GPUs"]), "GPUs");
        assert_eq!(
            and_words(&["GPUs", "CPUs", "memory"]),
            "GPUs, CPUs and memory"
        );
    }
}
