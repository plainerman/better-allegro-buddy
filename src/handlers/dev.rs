//! `bab dev start|status|login|stop` and `bab l`. Port of `handlers/dev.py`. Owned by P3-F.
//!
//! Notes:
//! * Mutations (scancel, sbatch) run strictly sequentially, in Python's order, after every
//!   check that can fail (see the comments in dev.py: a bad `--replace` must never cancel).
//! * `wait_for_job_running` shows `console.status(...)` (spinner only on a terminal),
//!   polls with `signals::sleep(poll_interval)` and real elapsed time (`Instant`).
//!   On Ctrl-C during `dev start`'s wait, print
//!   `muted("Detached; {id} stays queued (bab dev stop cancels it).")` and return
//!   `Err(AbError::Interrupted)`.
//! * `cmd_dev_login` ends in `core::slurm::exec_replace` (ssh / srun); a not-running job
//!   without `--wait` ends with a bare `SystemExit(1)` -> `Err(AbError::Exit(1))`.

use crate::core::gpu::{
    GPU_RESOURCE_CHUNK, GpuCounts, canonical_gpu_type, collect_display_gpu_types, gpu_type_label,
};
use crate::core::jobs::{display_job_id, find_dev_jobs, get_job_detail, pathlib_expanduser};
use crate::core::nodes::{is_gpu_partition, load_nodes};
use crate::core::parse::{base_job_id, expand_nodelist};
use crate::core::partitions::choose_partition;
use crate::core::slurm::{self, exec_replace};
use crate::core::timefmt::{format_seconds, parse_slurm_time};
use crate::core::ui::{
    build_dev_job_rows, hint, job_id_text, muted, node_text, partition_text, render_table, warn,
};
use crate::core::{ACTIVE_STATES, CLI_NAME, DEV_FRIENDLY_NAME, DEV_JOB_NAME, Record, RecordExt};
use crate::error::{AbError, Result, die};
use crate::pyfmt;
use crate::render::console;
use crate::render::table::Justify;
use crate::render::text::Text;
use crate::settings;
use crate::signals;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Arguments of `dev start` after defaults/default_map/callbacks (`gpu_type` canonical).
#[derive(Debug, Clone, PartialEq)]
pub struct DevStartArgs {
    pub partition: String,
    pub time_limit: String,
    pub gpu: Option<i64>,
    pub gpu_type: Option<String>,
    pub cpus: i64,
    pub mem: String,
    pub account: Option<String>,
    pub constraint: Option<String>,
    pub exclude: Option<String>,
    pub node: Option<String>,
    pub replace: bool,
    pub no_wait: bool,
    pub login: bool,
    pub timeout: Option<i64>,
    pub poll_interval: f64,
}

/// Arguments of `dev login` / `l`.
#[derive(Debug, Clone, PartialEq)]
pub struct DevLoginArgs {
    pub jobid: Option<String>,
    pub wait: bool,
    pub timeout: Option<i64>,
    pub poll_interval: f64,
    pub srun: bool,
    pub command: Option<String>,
}

/// `DEV_JOB_COLUMNS`.
const DEV_JOB_COLUMNS: [(&str, Justify); 8] = [
    ("JobID", Justify::Left),
    ("State", Justify::Left),
    ("Partition", Justify::Left),
    ("Node", Justify::Left),
    ("CPUs", Justify::Right),
    ("Memory", Justify::Right),
    ("GPUs", Justify::Right),
    ("Time left", Justify::Left),
];

fn render_dev_table(jobs: &[Record]) {
    render_table(
        &DEV_JOB_COLUMNS,
        build_dev_job_rows(jobs),
        Some("Dev jobs"),
        None,
        &[],
        None,
    );
}

/// `pick_single_dev_job(job_id)`.
fn pick_single_dev_job(job_id: Option<&str>) -> Result<Record> {
    let mut jobs = find_dev_jobs(None, job_id)?;
    if let Some(job_id) = job_id {
        if jobs.is_empty() {
            return Err(die(format!(
                "no active {DEV_FRIENDLY_NAME} matched {job_id}"
            )));
        }
        return Ok(jobs.swap_remove(0));
    }
    if jobs.is_empty() {
        return Err(die(format!(
            "no active {DEV_FRIENDLY_NAME}; start one with `{CLI_NAME} dev start`"
        )));
    }
    if jobs.len() > 1 {
        render_dev_table(&jobs);
        return Err(die(format!(
            "{} {DEV_FRIENDLY_NAME}s are active; pick one with `--jobid <id>`",
            jobs.len()
        )));
    }
    Ok(jobs.swap_remove(0))
}

/// `resolve_job_node(job)`.
fn resolve_job_node(job: &Record) -> Result<String> {
    match expand_nodelist(job.get_str("NodeList")).into_iter().next() {
        Some(node) => Ok(node),
        None => Err(die(format!(
            "job {} has no assigned node yet",
            display_job_id(job)
        ))),
    }
}

/// `settings.load().get(section, key) or default`.
fn setting_or<'a>(section: &str, key: &str, default: &'a str) -> std::borrow::Cow<'a, str> {
    match settings::get().get(section, key) {
        Some(value) if !value.is_empty() => value.to_owned().into(),
        _ => default.into(),
    }
}

/// `dev_log_dir()`.
fn dev_log_dir() -> PathBuf {
    PathBuf::from(pathlib_expanduser(&setting_or(
        "dev",
        "log_dir",
        "~/slurm-output",
    )))
}

/// `max_dev_allocations()` (validated by settings, so `int()` cannot fail).
fn max_dev_allocations() -> i64 {
    pyfmt::parse_int(&setting_or("dev", "max_allocations", "1")).unwrap_or(1)
}

fn node_login_is_srun() -> bool {
    settings::get().get("cluster", "node_login") == Some("srun")
}

/// `resolve_dev_gres_type(gpu_type, gpu_count)`: the GRES type exactly as the nodes spell
/// it (Slurm compares GRES types case-sensitively).
fn resolve_dev_gres_type(gpu_type: Option<&str>, gpu_count: i64) -> Result<Option<String>> {
    let Some(gpu_type) = gpu_type else {
        return Ok(None);
    };
    if gpu_count <= 0 {
        return Err(die("--gpu-type requires `--gpu` to be at least 1"));
    }
    let wanted = canonical_gpu_type(Some(gpu_type));
    let mut known: Vec<(String, String)> = Vec::new();
    for node in load_nodes(true)? {
        for caps in GPU_RESOURCE_CHUNK.captures_iter(&node.gres) {
            if let Some(kind) = caps
                .name("type")
                .map(|m| m.as_str())
                .filter(|t| !t.is_empty())
            {
                let canonical = canonical_gpu_type(Some(kind));
                if !known.iter().any(|(c, _)| *c == canonical) {
                    known.push((canonical, kind.to_owned()));
                }
            }
        }
    }
    if let Some((_, raw)) = known.iter().find(|(c, _)| *c == wanted) {
        return Ok(Some(raw.clone()));
    }
    let counts: GpuCounts = known.iter().map(|(c, _)| (c.clone(), 1)).collect();
    let choices = collect_display_gpu_types(&[&counts])
        .iter()
        .map(|name| gpu_type_label(name).to_lowercase())
        .collect::<Vec<_>>()
        .join(", ");
    let detail = if known.is_empty() {
        "this cluster has no typed GPUs".to_owned()
    } else {
        format!("this cluster has: {choices}")
    };
    Err(die(format!(
        "unknown --gpu-type {}; {detail}",
        pyfmt::repr_str(gpu_type)
    )))
}

/// `login_hint()`.
fn login_hint() {
    hint(&format!(
        "`{CLI_NAME} l` opens a shell in it (short for `{CLI_NAME} dev login`)"
    ));
    if node_login_is_srun() {
        hint("it joins with srun, inside the job's Slurm environment");
    } else {
        hint("it joins with ssh; --srun keeps the job's Slurm environment (CUDA_VISIBLE_DEVICES)");
    }
}

/// `render_dev_jobs(jobs, show_hint=)`.
fn render_dev_jobs(jobs: &[Record], show_hint: bool) {
    if jobs.is_empty() {
        console::stdout().print(&muted(&format!("No active {DEV_FRIENDLY_NAME}.")));
        if show_hint {
            hint(&format!("`{CLI_NAME} dev start` requests one"));
        }
        return;
    }
    render_dev_table(jobs);
    if show_hint && jobs.iter().any(|job| job.get_str("JobState") == "RUNNING") {
        login_hint();
    }
}

/// `wait_for_job_running(job_id, poll_interval, timeout)`.
fn wait_for_job_running(job_id: &str, poll_interval: f64, timeout: Option<i64>) -> Result<Record> {
    let pause = poll_pause(poll_interval)?;
    let start = Instant::now();
    let job = {
        // Transient spinner: gone once the job runs, and silent off a terminal.
        let status = console::stdout().status(Text::assemble([
            "Waiting for ".into(),
            job_id_text(job_id).into(),
        ]));
        loop {
            let Some(job) = get_job_detail(job_id)? else {
                return Err(die(format!("job {job_id} disappeared before it started")));
            };
            let state = job.get_str("JobState");
            if state == "RUNNING" {
                break job;
            }
            if !ACTIVE_STATES.contains(&state) {
                return Err(die(format!(
                    "job {job_id} ended ({state}) before it started"
                )));
            }
            if let Some(timeout) = timeout
                && start.elapsed().as_secs_f64() > timeout as f64
            {
                return Err(die(format!("timed out waiting for {job_id} to start")));
            }
            status.update(Text::assemble([
                "Waiting for ".into(),
                job_id_text(job_id).into(),
                muted(&format!(
                    " ({}: {})",
                    state.to_lowercase(),
                    job.get_opt("Reason").unwrap_or("unknown")
                ))
                .into(),
            ]));
            signals::sleep(pause)?;
        }
    };
    console::stdout().print(&Text::assemble([
        job_id_text(job_id).into(),
        " is running on ".into(),
        node_text(job.get_str("NodeList"), job.get_str("Partition")).into(),
    ]));
    Ok(job)
}

/// `--poll-interval` as a sleep, or the usage error for one `time.sleep` cannot take.
fn poll_pause(poll_interval: f64) -> Result<Duration> {
    crate::cli::parser::sleep_duration(&crate::cli::commands::POLL_INTERVAL, poll_interval)
}

/// `cmd_dev_start(...)`.
pub fn cmd_dev_start(args: &DevStartArgs) -> Result<()> {
    let out = console::stdout();
    // Validate mutually-exclusive flags BEFORE any state-changing call so a bad
    // invocation can't silently cancel the user's existing dev job.
    if args.login && args.no_wait {
        return Err(die("`--no-wait` cannot be used together with `--login`"));
    }

    let existing = find_dev_jobs(None, None)?;
    let limit = max_dev_allocations();
    if existing.len() as i64 >= limit && !args.replace {
        render_dev_jobs(&existing, false);
        return Err(die(format!(
            "you already have {} {DEV_FRIENDLY_NAME}(s) (limit {limit}); \
             stop with `{CLI_NAME} dev stop` or pass `--replace`",
            existing.len()
        )));
    }

    let selected_partition = choose_partition(&args.partition, &args.time_limit)?;
    let gpu_partition = is_gpu_partition(&selected_partition, true)?;
    // No --gpu and no [dev] gpu: one GPU on a GPU partition, none elsewhere.
    let gpu_count = args.gpu.unwrap_or(i64::from(gpu_partition));
    let gres_type = resolve_dev_gres_type(args.gpu_type.as_deref(), gpu_count)?;

    if gpu_partition && gpu_count <= 0 {
        return Err(die("GPU partitions require `--gpu` to be at least 1"));
    }
    if !gpu_partition && gpu_count > 0 {
        return Err(die(format!(
            "partition {selected_partition} has no GPU nodes; drop --gpu or choose a GPU partition"
        )));
    }

    // bab-only: Python would submit (and maybe cancel the old job) before crashing in
    // `time.sleep(nan)`; reject an unusable --poll-interval while nothing has changed yet.
    if args.login || !args.no_wait {
        poll_pause(args.poll_interval)?;
    }

    let log_dir = dev_log_dir();
    if let Err(err) = std::fs::create_dir_all(&log_dir) {
        return Err(die(pyfmt::os_error_str(&err, &log_dir)));
    }
    let output_pattern = log_dir
        .join(format!("{DEV_JOB_NAME}-%j.out"))
        .display()
        .to_string();
    // Trap SIGTERM/SIGINT to no-op; background sleep + wait unblocks cleanly on Slurm shutdown.
    let wrap = "trap : TERM INT; sleep infinity & wait";

    let cpus = args.cpus;
    let cpus_per_gpu = if gpu_count > 0 {
        pyfmt::ceil_div(cpus, gpu_count)
    } else {
        cpus
    };
    if gpu_count > 0 && cpus % gpu_count != 0 {
        warn(format!(
            "--cpus {cpus} is not divisible by --gpu {gpu_count}; \
             using {} CPUs ({cpus_per_gpu} per GPU)",
            cpus_per_gpu * gpu_count
        ));
    }
    let mut argv = vec![
        "sbatch".to_owned(),
        "--parsable".to_owned(),
        format!("--job-name={DEV_JOB_NAME}"),
        format!("--partition={selected_partition}"),
        format!("--time={}", args.time_limit),
        format!("--mem={}", args.mem),
        format!("--output={output_pattern}"),
        format!("--wrap={wrap}"),
    ];
    if gpu_count > 0 {
        argv.push(format!("--cpus-per-gpu={cpus_per_gpu}"));
        argv.push(match &gres_type {
            Some(kind) if !kind.is_empty() => format!("--gres=gpu:{kind}:{gpu_count}"),
            _ => format!("--gres=gpu:{gpu_count}"),
        });
    } else {
        argv.push(format!("--cpus-per-task={cpus}"));
    }
    let optional = [
        ("account", &args.account),
        ("constraint", &args.constraint),
        ("exclude", &args.exclude),
        ("nodelist", &args.node),
    ];
    for (flag, value) in optional {
        if let Some(value) = value.as_deref().filter(|v| !v.is_empty()) {
            argv.push(format!("--{flag}={value}"));
        }
    }

    // Cancel only now, after every check that can still fail, so a bad --replace
    // request never costs the user their running allocation.
    if !existing.is_empty() && args.replace {
        let mut scancel = vec!["scancel".to_owned()];
        scancel.extend(existing.iter().map(|job| job.get_str("JobId").to_owned()));
        slurm::run_inherit(&scancel)?;
        let ids: Vec<String> = existing.iter().map(display_job_id).collect();
        out.print(&muted(&format!("Canceled {}.", ids.join(", "))));
        signals::sleep(Duration::from_secs(1))?;
    }

    let proc = slurm::run(&argv, true)?;
    // `sbatch --parsable` prints "<jobid>" or "<jobid>;<cluster>" (multi-cluster).
    let job_id = pyfmt::strip(&proc.stdout)
        .split(';')
        .next()
        .unwrap_or("")
        .to_owned();

    let gres_label = gres_type
        .as_deref()
        .filter(|t| !t.is_empty())
        .map(|t| format!(" {t}"))
        .unwrap_or_default();
    let request = [
        if gpu_count != 0 {
            format!("{gpu_count} GPU{gres_label}")
        } else {
            String::new()
        },
        format!("{cpus} CPUs"),
        args.mem.clone(),
        match parse_slurm_time(&args.time_limit) {
            Some(seconds) => format_seconds(Some(seconds)),
            None => args.time_limit.clone(),
        },
        match args.constraint.as_deref().filter(|c| !c.is_empty()) {
            Some(constraint) => format!("constraint {constraint}"),
            None => String::new(),
        },
    ];
    let request: Vec<&str> = request
        .iter()
        .map(String::as_str)
        .filter(|part| !part.is_empty())
        .collect();
    out.print(&Text::assemble([
        format!("Submitted {DEV_FRIENDLY_NAME} ").into(),
        job_id_text(&job_id).into(),
        " on ".into(),
        partition_text(&selected_partition).into(),
        muted(&format!(" ({})", request.join(", "))).into(),
    ]));

    if args.login || !args.no_wait {
        match wait_for_job_running(&job_id, args.poll_interval, args.timeout) {
            Err(AbError::Interrupted) => {
                out.print(&muted(&format!(
                    "Detached; {job_id} stays queued ({CLI_NAME} dev stop cancels it)."
                )));
                return Err(AbError::Interrupted);
            }
            other => {
                other?;
            }
        }
        if args.login {
            // Replaces this process, so no next step.
            return cmd_dev_login(&DevLoginArgs {
                jobid: Some(job_id),
                wait: false,
                timeout: args.timeout,
                poll_interval: args.poll_interval,
                srun: false,
                command: None,
            });
        }
        login_hint();
        return Ok(());
    }

    let detail = get_job_detail(&job_id)?.unwrap_or_default();
    if detail.get_opt("JobState") == Some("RUNNING") {
        login_hint();
    } else {
        hint(&format!(
            "`{CLI_NAME} l --wait` joins it once it runs (short for `{CLI_NAME} dev login --wait`)"
        ));
    }
    Ok(())
}

/// `cmd_dev_status(jobid=)`.
pub fn cmd_dev_status(jobid: Option<&str>) -> Result<()> {
    render_dev_jobs(&find_dev_jobs(None, jobid)?, true);
    Ok(())
}

/// `cmd_dev_login(...)`: only returns on error (exec replaces the process).
pub fn cmd_dev_login(args: &DevLoginArgs) -> Result<()> {
    let mut job = pick_single_dev_job(args.jobid.as_deref())?;
    if job.get_opt("JobState") != Some("RUNNING") {
        if args.wait {
            let job_id = job.get_str("JobId").to_owned();
            job = wait_for_job_running(&job_id, args.poll_interval, args.timeout)?;
        } else {
            console::stdout().print(&Text::assemble([
                job_id_text(&display_job_id(&job)).into(),
                format!(" is {}", job.get_str("JobState").to_lowercase()).into(),
                muted(&format!(
                    " ({})",
                    job.get_opt("Reason").unwrap_or("unknown")
                ))
                .into(),
            ]));
            hint(&format!(
                "`{CLI_NAME} l --wait` waits for it to start, then joins"
            ));
            return Err(AbError::Exit(1)); // did not log in or run --command
        }
    }

    let node = resolve_job_node(&job)?;

    if args.srun || node_login_is_srun() {
        let job_id = base_job_id(job.get_str("JobId"));
        if let Some(command) = args.command.as_deref().filter(|c| !c.is_empty()) {
            return Err(exec_replace(&[
                "srun",
                "--jobid",
                job_id,
                "--overlap",
                "-N1",
                "-n1",
                "bash",
                "-lc",
                command,
            ]));
        }
        return Err(exec_replace(&[
            "srun",
            "--jobid",
            job_id,
            "--overlap",
            "--pty",
            "bash",
            "-l",
        ]));
    }

    if let Some(command) = args.command.as_deref().filter(|c| !c.is_empty()) {
        // ssh joins its args and the remote shell re-splits them: quote once.
        let remote = format!("bash -lc {}", pyfmt::shlex_quote(command));
        return Err(exec_replace(&["ssh", "-t", &node, &remote]));
    }
    Err(exec_replace(&["ssh", "-t", &node]))
}

/// `cmd_dev_stop(jobid=)`. The final `Canceled ...` line is a plain `str` print
/// (markup + ReprHighlighter: job ids are highlighted as numbers).
pub fn cmd_dev_stop(jobid: Option<&str>) -> Result<()> {
    let out = console::stdout();
    let jobs = find_dev_jobs(None, jobid)?;
    if jobs.is_empty() {
        out.print(&muted(&format!("No active {DEV_FRIENDLY_NAME} to stop.")));
        return Ok(());
    }
    let mut scancel = vec!["scancel".to_owned()];
    scancel.extend(jobs.iter().map(|job| job.get_str("JobId").to_owned()));
    slurm::run_inherit(&scancel)?;
    let ids: Vec<String> = jobs.iter().map(display_job_id).collect();
    out.print_str(&format!("Canceled {}.", ids.join(", ")), None);
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Port of `tests/test_dev.py`: what gets cancelled, submitted and executed.
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, fail, ok};
    use crate::hooks::testing::Ctx;
    use std::sync::Arc;

    const RUNNING: &str = "JobId=111 JobName=ab-dev UserId=alice(1000) JobState=RUNNING \
        Reason=None Partition=gpu NodeList=gpu001 SubmitTime=2030-01-01T08:00:00";
    const PARTITIONS: &str = "PartitionName=gpu AllowGroups=ALL MaxTime=UNLIMITED\n";

    fn scratch_log_dir() -> String {
        let exe = std::env::current_exe().expect("test exe");
        exe.parent()
            .and_then(std::path::Path::parent)
            .expect("target dir")
            .join("bab-test-scratch")
            .join(format!("dev-logs-{}", std::process::id()))
            .display()
            .to_string()
    }

    /// A cluster with one `ab-dev` job (`detail`, or none) answering every dev call.
    fn cluster(detail: Option<&'static str>) -> Arc<FakeSlurm> {
        FakeSlurm::new(move |argv| {
            Some(match argv[0].as_str() {
                "squeue" => ok(if detail.is_some() { "111\n" } else { "" }),
                "scontrol" if argv.get(3).map(String::as_str) == Some("partition") => {
                    ok(PARTITIONS)
                }
                "scontrol" if argv.last().map(String::as_str) == Some("111") => {
                    ok(detail.unwrap_or(""))
                }
                "scontrol" => fail(1, "Invalid job id specified"),
                "sbatch" => ok("222\n"),
                _ => ok(""),
            })
        })
    }

    fn install(slurm: &Arc<FakeSlurm>) -> crate::hooks::testing::Guard {
        let log_dir = format!("log_dir = {}", scratch_log_dir());
        let settings = crate::settings::testing::with_preset(
            "allegro",
            &["[dev]", &log_dir, "[cluster]", "gpu_partitions = gpu"],
        );
        Ctx::default()
            .groups(&["users"])
            .slurm(slurm.clone())
            .settings(settings)
            .install()
    }

    fn start_args() -> DevStartArgs {
        DevStartArgs {
            partition: "gpu".into(),
            time_limit: "08:00:00".into(),
            gpu: Some(2),
            gpu_type: None,
            cpus: 4,
            mem: "8G".into(),
            account: None,
            constraint: None,
            exclude: None,
            node: None,
            replace: false,
            no_wait: true,
            login: false,
            timeout: None,
            poll_interval: 5.0,
        }
    }

    fn commands(slurm: &FakeSlurm, names: &[&str]) -> Vec<Vec<String>> {
        slurm
            .calls()
            .into_iter()
            .filter(|argv| names.contains(&argv[0].as_str()))
            .collect()
    }

    #[test]
    fn replace_cancels_then_submits_gpu_request() {
        let slurm = cluster(Some(RUNNING));
        let _ctx = install(&slurm);
        let args = DevStartArgs {
            replace: true,
            ..start_args()
        };
        assert_eq!(cmd_dev_start(&args), Ok(()));
        let calls = commands(&slurm, &["scancel", "sbatch"]);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ["scancel", "111"]);
        let sbatch = &calls[1];
        assert_eq!(sbatch[..2], ["sbatch", "--parsable"]);
        for expected in [
            "--job-name=ab-dev",
            "--partition=gpu",
            "--gres=gpu:2",
            "--cpus-per-gpu=2",
            "--time=08:00:00",
        ] {
            assert!(
                sbatch.iter().any(|a| a == expected),
                "{expected} in {sbatch:?}"
            );
        }
        assert!(sbatch.contains(&format!("--output={}/ab-dev-%j.out", scratch_log_dir())));
    }

    #[test]
    fn login_and_no_wait_conflict_before_any_call() {
        let slurm = cluster(Some(RUNNING));
        let _ctx = install(&slurm);
        let args = DevStartArgs {
            login: true,
            replace: true,
            ..start_args()
        };
        assert_eq!(cmd_dev_start(&args), Err(AbError::Exit(1)));
        assert!(slurm.calls().is_empty());
    }

    fn login_args(command: Option<&str>, srun: bool) -> DevLoginArgs {
        DevLoginArgs {
            jobid: None,
            wait: false,
            timeout: None,
            poll_interval: 5.0,
            srun,
            command: command.map(str::to_owned),
        }
    }

    #[test]
    fn login_command_is_one_quoted_remote_string() {
        let slurm = cluster(Some(RUNNING));
        let _ctx = install(&slurm);
        let _ = cmd_dev_login(&login_args(Some("cd x && make"), false));
        assert_eq!(
            commands(&slurm, &["ssh"]),
            [["ssh", "-t", "gpu001", "bash -lc 'cd x && make'"]]
        );
    }

    #[test]
    fn login_srun_command_and_shell() {
        let slurm = cluster(Some(RUNNING));
        let _ctx = install(&slurm);
        let _ = cmd_dev_login(&login_args(Some("make"), true));
        let _ = cmd_dev_login(&login_args(None, true));
        let calls = commands(&slurm, &["srun"]);
        assert_eq!(
            calls[0],
            [
                "srun",
                "--jobid",
                "111",
                "--overlap",
                "-N1",
                "-n1",
                "bash",
                "-lc",
                "make"
            ]
        );
        assert_eq!(
            calls[1],
            ["srun", "--jobid", "111", "--overlap", "--pty", "bash", "-l"]
        );
    }
}
