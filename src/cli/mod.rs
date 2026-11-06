//! The command-line layer: click/rich-click compatible parsing, help, completion,
//! prompts, `--watch`, and dispatch to the handlers. Port of `cli.py`. Owned by P2-B
//! (framework) and P3-F (wiring details of completion/dev/job/config).
//!
//! # Contract
//!
//! * [`main`] is called by `main.rs` with the raw argv and returns the process exit code.
//!   It installs the SIGINT flag handler and `SIGPIPE=SIG_DFL` ([`crate::signals`]),
//!   answers completion requests (`_BAB_COMPLETE`), parses argv against
//!   [`commands::MAIN`], runs the handler via [`dispatch`], and turns an
//!   [`AbError`] into output + exit code (see [`crate::error`]).
//! * Program name: always [`crate::core::CLI_NAME`] (`bab`) in usage lines, independent
//!   of argv[0] (the Python reference is patched to `bab` the same way). The completion
//!   env var follows click: `_{BASENAME(argv[0])}_COMPLETE`, so `--prog` scripts work.
//! * `ctx.default_map`: built by [`load_default_map`] from the merged config `[dev]`
//!   section (`settings::DEV_START_KEYS` maps config keys to `dev start` params) at the
//!   moment click runs the main group callback; a `ConfigError` becomes
//!   `AbError::Click(message)`.
//! * Handlers get typed arguments (see [`dispatch`]); `--watch` commands are wrapped in
//!   [`watch::run_watched`].
//! * [`echo`] is `click.echo` (plain write + newline to stdout, no Rich).

#![warn(unused, dead_code)]

pub mod commands;
pub mod completion;
pub mod help;
pub mod parser;
pub mod prompt;
pub mod rich;
pub mod themes;
pub mod watch;

use crate::error::{AbError, Result};
use crate::handlers::{self, dev, resources, why};
use commands::Action;
use parser::{Args, DefaultMap, Parsed};
use std::ffi::OsString;

/// Entry point: returns the exit code (only `main.rs` exits).
pub fn main(argv: Vec<OsString>) -> i32 {
    crate::signals::install();
    // Python imports cli.py, whose RichHelpConfiguration applies the theme, before anything
    help::apply_theme_warning();
    let prog_name = argv
        .first()
        .map(|a| program_name(&a.to_string_lossy()))
        .unwrap_or_else(|| crate::core::CLI_NAME.to_owned());
    if let Some(code) = completion::handle_env_request(&prog_name) {
        return code;
    }
    let args: Vec<String> = argv
        .iter()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let result = run(&args).and_then(|()| crate::signals::check());
    match result {
        Ok(()) => 0,
        Err(err) => report(err),
    }
}

/// click's `_detect_program_name`: the basename of argv[0].
fn program_name(argv0: &str) -> String {
    let name = argv0.rsplit('/').next().unwrap_or(argv0);
    if name.is_empty() {
        crate::core::CLI_NAME.to_owned()
    } else {
        name.to_owned()
    }
}

/// Parse and run (no exit-code mapping).
pub fn run(argv: &[String]) -> Result<()> {
    match parser::parse(argv, &load_default_map)? {
        Parsed::Help {
            path,
            default_map,
            code,
        } => {
            help::print_help(&path, default_map.as_deref());
            if code == 0 {
                Ok(())
            } else {
                Err(AbError::Exit(code))
            }
        }
        Parsed::Version => {
            echo(&format!(
                "{}, version {}",
                crate::core::CLI_NAME,
                crate::VERSION_DISPLAY
            ));
            Ok(())
        }
        Parsed::Run {
            path,
            command,
            args,
        } => {
            let action = command
                .action
                .expect("parser only returns runnable commands");
            dispatch(action, &args).map_err(|err| match err {
                AbError::Usage(mut usage)
                    if usage.command_path.is_empty() && usage.show_help_hint =>
                {
                    usage.command_path = path;
                    AbError::Usage(usage)
                }
                other => other,
            })
        }
    }
}

/// Print what Python/click prints for an error that reached the top, return the exit code.
pub fn report(err: AbError) -> i32 {
    match &err {
        AbError::Exit(_) => {}
        AbError::Usage(usage) => {
            help::python_warning_filters_mutated();
            help::print_usage_error(usage)
        }
        AbError::Click(message) => {
            help::python_warning_filters_mutated();
            help::print_click_exception(message)
        }
        AbError::Abort => help::print_aborted(),
        AbError::Interrupted => {
            write_stream(&mut std::io::stderr(), "\n");
            help::print_aborted();
        }
    }
    err.exit_code()
}

/// `load_default_map()`: `[dev]` config values for `dev start` params (by dest).
pub fn load_default_map() -> Result<DefaultMap> {
    let settings = crate::settings::load().map_err(|e| AbError::Click(e.0))?;
    let mut map = DefaultMap::new();
    for (key, value) in settings.section("dev") {
        if let Some((_, dest)) = crate::settings::DEV_START_KEYS
            .iter()
            .find(|(config_key, _)| *config_key == key)
        {
            map.push((dest, value));
        }
    }
    Ok(map)
}

/// `click.echo(text)`: plain stdout write plus newline.
pub fn echo(text: &str) {
    write_stream(&mut std::io::stdout(), &format!("{text}\n"));
}

/// Write and flush, ignoring errors (a closed pipe ends the process via SIGPIPE).
pub(crate) fn write_stream(stream: &mut dyn std::io::Write, text: &str) {
    let _ = stream.write_all(text.as_bytes());
    let _ = stream.flush();
}

/// Run the handler for `action` with typed arguments (Python's `*_command` functions).
pub fn dispatch(action: Action, args: &Args) -> Result<()> {
    match action {
        Action::Info => resources::cmd_info(args.flag("all_partitions")),
        Action::Queue => {
            let (all_users, user, partition) = (
                args.flag("all_users"),
                args.opt_str("user"),
                args.opt_str("partition"),
            );
            watch::run_watched(
                &mut || resources::cmd_queue(all_users, user, partition),
                args.opt_float("watch"),
            )
        }
        Action::Why => why::cmd_why(
            args.opt_str("jobid"),
            args.flag("show_nodes"),
            args.opt_str("fix").map(|fix| match fix {
                "auto" => why::FixChoice::Auto,
                "mem" => why::FixChoice::Mem,
                "cpus" => why::FixChoice::Cpus,
                "gpus" => why::FixChoice::Gpus,
                other => unreachable!("Choice validated --fix {other}"),
            }),
            args.flag("assume_yes"),
        ),
        Action::Completion => completion::cmd_completion(
            args.opt_str("shell"),
            args.str("prog"),
            args.flag("install"),
        ),
        Action::Free => {
            let kind = match args.str("kind") {
                "cpu" => resources::Kind::Cpu,
                _ => resources::Kind::Gpu,
            };
            let (nodes, partition, gpu_type) = (
                args.flag("nodes"),
                args.opt_str("partition"),
                args.opt_str("gpu_type"),
            );
            watch::run_watched(
                &mut || resources::cmd_free(kind, nodes, partition, gpu_type),
                args.opt_float("watch"),
            )
        }
        Action::Node | Action::Gpualloc => resources::cmd_node(args.opt_str("node")),
        Action::JobHistory => handlers::job::cmd_job_history(args.int("limit")),
        Action::JobOutput => handlers::job::cmd_job_output(args.str("jobid"), args.flag("stderr")),
        Action::DevStart => dev::cmd_dev_start(&dev::DevStartArgs {
            partition: args.str("partition").to_owned(),
            time_limit: args.str("time_limit").to_owned(),
            gpu: args.opt_int("gpu"),
            gpu_type: args.opt_str("gpu_type").map(str::to_owned),
            cpus: args.int("cpus"),
            mem: args.str("mem").to_owned(),
            account: args.opt_str("account").map(str::to_owned),
            constraint: args.opt_str("constraint").map(str::to_owned),
            exclude: args.opt_str("exclude").map(str::to_owned),
            node: args.opt_str("node").map(str::to_owned),
            replace: args.flag("replace"),
            no_wait: args.flag("no_wait"),
            login: args.flag("login"),
            timeout: args.opt_int("timeout"),
            poll_interval: args.float("poll_interval"),
        }),
        Action::DevStatus => dev::cmd_dev_status(args.opt_str("jobid")),
        Action::DevLogin => dev::cmd_dev_login(&dev::DevLoginArgs {
            jobid: args.opt_str("jobid").map(str::to_owned),
            wait: args.flag("wait"),
            timeout: args.opt_int("timeout"),
            poll_interval: args.float("poll_interval"),
            srun: args.flag("srun"),
            command: args.opt_str("command").map(str::to_owned),
        }),
        Action::DevStop => dev::cmd_dev_stop(args.opt_str("jobid")),
        Action::ConfigShow => handlers::config::cmd_config_show(),
    }
}
