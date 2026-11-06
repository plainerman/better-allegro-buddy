//! Declarative command tree: every command, option and argument of Python `cli.py` with
//! exact names, help texts, defaults and order. Data only; the parser ([`super::parser`]),
//! help renderer ([`super::help`]) and completion ([`super::completion`]) interpret it.
//!
//! Order rules (click): params are listed in source/decorator order (done here); click
//! appends the `-h/--help` option last (not listed here: the framework adds it to every
//! command, `context_settings help_option_names=['-h', '--help']`, help
//! `Show this message and exit.`). The main group additionally has click's
//! `version_option` (`--version`, `Show the version and exit.`, eager, prints
//! `bab, version 2.7`), declared below as [`ParamType::Version`].
//! Subcommands are listed sorted by name in help, except `l`, which `_Main.list_commands`
//! moves right after `dev`. `queue` has the rich-click alias `q` (help shows `queue, q`).
//! `gpualloc` is hidden. `l` is a copy of `dev login` registered as `l` with its own
//! short help.
//!
//! Text changes vs Python (ab -> bab renames): `dev start`'s docstring and `l`'s short help.

use crate::core::{DEFAULT_DEV_CPUS, DEFAULT_DEV_MEM, DEFAULT_DEV_TIME};

/// Python `DEFAULT_POLL_INTERVAL`.
pub const DEFAULT_POLL_INTERVAL: f64 = 5.0;
/// Python `DEFAULT_WATCH_INTERVAL`.
pub const DEFAULT_WATCH_INTERVAL: f64 = 10.0;

/// Option vs positional argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    Option,
    Argument,
}

/// click parameter types used by allegro-buddy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamType {
    /// `is_flag=True` boolean flag.
    Flag,
    /// `click.STRING` (the default type).
    Str,
    /// `click.INT` (`type=int`).
    Int,
    /// `click.FloatRange(min=...)` (closed lower bound, no upper bound).
    FloatRange { min: f64 },
    /// `click.Choice([...])` (case-sensitive).
    Choice(&'static [&'static str]),
    /// click's `version_option` (`--version`): eager, prints the version and exits 0.
    Version,
}

/// A constant default / flag value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    None,
    Bool(bool),
    Str(&'static str),
    Int(i64),
    Float(f64),
}

/// click `show_default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShowDefault {
    No,
    /// `show_default=True`: `[default: <value>]`.
    Yes,
    /// `show_default='text'`: `[default: text]`.
    Text(&'static str),
}

/// Dynamic shell completion for a parameter (`shell_complete=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completer {
    /// `complete_jobid`: `core::jobs::job_history_completion_ids`.
    JobHistory,
    /// `complete_dev_jobid`: ids of `find_dev_jobs()` starting with the prefix.
    DevJob,
    /// `complete_pending_jobid`: `JobID`s of `load_user_pending_jobs()` with the prefix.
    PendingJob,
}

/// Value callbacks (`callback=`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Callback {
    /// `_normalize_gpu_type`: None stays None, else `core::gpu::canonical_gpu_type`.
    NormalizeGpuType,
}

/// One click parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param {
    pub kind: ParamKind,
    /// Declarations as in Python (`["-w", "--watch"]`; arguments: `["jobid"]`).
    pub decls: &'static [&'static str],
    /// Python parameter (dest) name, also the default_map key.
    pub dest: &'static str,
    pub ty: ParamType,
    /// `flag_value` for options with an optional value (`--watch`, `--fix`):
    /// `is_flag=False, flag_value=X` -> the option may be given without a value.
    pub flag_value: Option<Const>,
    pub default: Const,
    pub show_default: ShowDefault,
    pub metavar: Option<&'static str>,
    pub help: Option<&'static str>,
    /// Arguments: click.argument default `required=True` unless `required=False`.
    pub required: bool,
    pub completer: Option<Completer>,
    pub callback: Option<Callback>,
    pub hidden: bool,
}

impl Param {
    const fn option(decls: &'static [&'static str], dest: &'static str) -> Param {
        Param {
            kind: ParamKind::Option,
            decls,
            dest,
            ty: ParamType::Str,
            flag_value: None,
            default: Const::None,
            show_default: ShowDefault::No,
            metavar: None,
            help: None,
            required: false,
            completer: None,
            callback: None,
            hidden: false,
        }
    }

    const fn flag(decls: &'static [&'static str], dest: &'static str, help: &'static str) -> Param {
        Param {
            ty: ParamType::Flag,
            default: Const::Bool(false),
            help: Some(help),
            ..Param::option(decls, dest)
        }
    }

    const fn argument(name: &'static str, required: bool) -> Param {
        Param {
            kind: ParamKind::Argument,
            required,
            ..Param::option(&[], name)
        }
    }

    const fn help(self, help: &'static str) -> Param {
        Param {
            help: Some(help),
            ..self
        }
    }
}

/// What a leaf command runs (dispatched in [`super::dispatch`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Info,
    Queue,
    Why,
    Completion,
    Free,
    Node,
    Gpualloc,
    JobHistory,
    JobOutput,
    DevStart,
    DevStatus,
    DevLogin,
    DevStop,
    ConfigShow,
}

/// A command or group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    /// Name it is registered under (`l` for the login copy).
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    /// `short_help=` (command lists).
    pub short_help: Option<&'static str>,
    /// Docstring (`help`), shown on the command's own help page.
    pub help: Option<&'static str>,
    pub hidden: bool,
    pub params: &'static [Param],
    /// Non-empty (or `is_group`) for groups; registration order (help sorts by name).
    pub subcommands: &'static [Command],
    pub is_group: bool,
    /// Main group: `invoke_without_command=True` (runs `info`).
    pub invoke_without_command: bool,
    pub action: Option<Action>,
}

impl Command {
    const fn leaf(name: &'static str, short_help: &'static str, action: Action) -> Command {
        Command {
            name,
            aliases: &[],
            short_help: Some(short_help),
            help: None,
            hidden: false,
            params: &[],
            subcommands: &[],
            is_group: false,
            invoke_without_command: false,
            action: Some(action),
        }
    }

    const fn group(
        name: &'static str,
        short_help: &'static str,
        subcommands: &'static [Command],
    ) -> Command {
        Command {
            name,
            aliases: &[],
            short_help: Some(short_help),
            help: None,
            hidden: false,
            params: &[],
            subcommands,
            is_group: true,
            invoke_without_command: false,
            action: None,
        }
    }
}

/// `watch_option`.
pub const WATCH: Param = Param {
    ty: ParamType::FloatRange { min: 10.0 },
    flag_value: Some(Const::Float(DEFAULT_WATCH_INTERVAL)),
    metavar: Some("[SECONDS]"),
    help: Some("refresh every SECONDS (default 10) until Ctrl-C"),
    ..Param::option(&["-w", "--watch"], "watch")
};

const DEV_JOBID: Param = Param {
    completer: Some(Completer::DevJob),
    ..Param::option(&["--jobid"], "jobid").help("pick one dev job by id")
};

pub const POLL_INTERVAL: Param = Param {
    ty: ParamType::FloatRange { min: 1.0 },
    default: Const::Float(DEFAULT_POLL_INTERVAL),
    show_default: ShowDefault::Yes,
    ..Param::option(&["--poll-interval"], "poll_interval")
        .help("seconds between checks while waiting")
};

/// `add_dev_login_options` (shown order).
const DEV_LOGIN_PARAMS: &[Param] = &[
    DEV_JOBID,
    Param::flag(&["--wait"], "wait", "wait until the job is RUNNING"),
    Param {
        ty: ParamType::Int,
        ..Param::option(&["--timeout"], "timeout").help("max seconds to wait (with --wait)")
    },
    POLL_INTERVAL,
    Param::flag(
        &["--srun"],
        "srun",
        "use srun --overlap instead of ssh (keeps the job's GPU setup)",
    ),
    Param::option(&["--command"], "command").help("run a command instead of opening a shell"),
];

const INFO: Command = Command {
    params: &[Param::flag(
        &["--all-partitions"],
        "all_partitions",
        "include partitions you cannot use",
    )],
    ..Command::leaf(
        "info",
        "show your partitions and a cluster summary",
        Action::Info,
    )
};

const QUEUE: Command = Command {
    aliases: &["q"],
    params: &[
        Param::flag(&["--all"], "all_users", "show everyone's jobs"),
        Param::option(&["--user"], "user").help("show jobs of USER"),
        Param::option(&["-p", "--partition"], "partition").help("filter by partition"),
        WATCH,
    ],
    ..Command::leaf("queue", "show your jobs in the queue", Action::Queue)
};

const WHY: Command = Command {
    help: Some(
        "Explain why a pending job has not started and what would start it now.\n\n\
         Without JOBID, cover each of your jobs waiting for resources.",
    ),
    params: &[
        Param {
            completer: Some(Completer::PendingJob),
            ..Param::argument("jobid", false)
        },
        Param::flag(&["--nodes"], "show_nodes", "add a per-node table"),
        Param {
            ty: ParamType::Choice(&["auto", "mem", "cpus", "gpus"]),
            flag_value: Some(Const::Str("auto")),
            ..Param::option(&["--fix"], "fix")
                .help("edit the job so it starts now (auto: best mem/cpus change)")
        },
        Param::flag(&["-y", "--yes"], "assume_yes", "apply --fix without asking"),
    ],
    ..Command::leaf(
        "why",
        "explain what a pending job is waiting for",
        Action::Why,
    )
};

const COMPLETION: Command = Command {
    help: Some("Print the completion script for SHELL (default: $SHELL), or --install it."),
    params: &[
        Param {
            ty: ParamType::Choice(&["bash", "zsh", "fish"]),
            ..Param::argument("shell", false)
        },
        Param {
            default: Const::Str(crate::core::CLI_NAME),
            show_default: ShowDefault::Yes,
            ..Param::option(&["--prog"], "prog").help("program name to complete")
        },
        Param::flag(
            &["--install"],
            "install",
            "install it and source it from your shell startup file",
        ),
    ],
    ..Command::leaf(
        "completion",
        "generate or install shell completion",
        Action::Completion,
    )
};

const FREE: Command = Command {
    params: &[
        Param {
            ty: ParamType::Choice(&["gpu", "cpu"]),
            default: Const::Str("gpu"),
            show_default: ShowDefault::Yes,
            ..Param::option(&["--kind"], "kind").help("partitions to show")
        },
        Param::flag(&["--nodes"], "nodes", "add a per-node table"),
        Param::option(&["-p", "--partition"], "partition")
            .help("limit the node table to one partition"),
        Param {
            callback: Some(Callback::NormalizeGpuType),
            ..Param::option(&["-G", "--gpu-type"], "gpu_type")
                .help("only this GPU type, e.g. a5000")
        },
        WATCH,
    ],
    ..Command::leaf("free", "show free GPUs and cluster resources", Action::Free)
};

const NODE: Command = Command {
    help: Some("Show the jobs on NODE (default: this node, or the node of your job)."),
    params: &[Param::argument("node", false)],
    ..Command::leaf(
        "node",
        "show which jobs use a node's GPUs, CPUs and memory",
        Action::Node,
    )
};

/// The old name of `node`: hidden, no help texts.
const GPUALLOC: Command = Command {
    short_help: None,
    hidden: true,
    params: &[Param::argument("node", false)],
    ..Command::leaf("gpualloc", "", Action::Gpualloc)
};

const JOB: Command = Command::group(
    "job",
    "show finished jobs and their output",
    &[
        Command {
            params: &[Param {
                ty: ParamType::Int,
                default: Const::Int(50),
                show_default: ShowDefault::Yes,
                ..Param::option(&["--limit"], "limit").help("how many jobs to show (0: all)")
            }],
            ..Command::leaf(
                "history",
                "list your finished jobs (last 30 days)",
                Action::JobHistory,
            )
        },
        Command {
            params: &[
                Param {
                    completer: Some(Completer::JobHistory),
                    ..Param::argument("jobid", true)
                },
                Param::flag(&["--stderr"], "stderr", "show stderr instead of stdout"),
            ],
            ..Command::leaf(
                "output",
                "show the end of a job's output",
                Action::JobOutput,
            )
        },
    ],
);

const DEV_START_PARAMS: &[Param] = &[
    Param {
        default: Const::Str("auto"),
        show_default: ShowDefault::Yes,
        ..Param::option(&["-p", "--partition"], "partition")
            .help("partition, or auto for Slurm's default")
    },
    Param {
        default: Const::Str(DEFAULT_DEV_TIME),
        show_default: ShowDefault::Yes,
        ..Param::option(&["-t", "--time"], "time_limit").help("Slurm time limit")
    },
    Param {
        ty: ParamType::Int,
        show_default: ShowDefault::Text("1 on GPU partitions, else 0"),
        ..Param::option(&["-g", "--gpu"], "gpu").help("GPUs to request")
    },
    Param {
        callback: Some(Callback::NormalizeGpuType),
        show_default: ShowDefault::Text("any"),
        ..Param::option(&["--gpu-type"], "gpu_type").help("GPU type, e.g. a5000")
    },
    Param {
        ty: ParamType::Int,
        default: Const::Int(DEFAULT_DEV_CPUS),
        show_default: ShowDefault::Yes,
        ..Param::option(&["-c", "--cpus"], "cpus").help("CPUs to request")
    },
    Param {
        default: Const::Str(DEFAULT_DEV_MEM),
        show_default: ShowDefault::Yes,
        ..Param::option(&["-m", "--mem"], "mem").help("memory to request")
    },
    Param::option(&["--account"], "account").help("Slurm account"),
    Param::option(&["--constraint"], "constraint")
        .help("Slurm node feature constraint (prefer --gpu-type)"),
    Param::option(&["--exclude"], "exclude").help("comma-separated nodes to exclude"),
    Param::option(&["-n", "--node"], "node").help("request a specific node"),
    Param::flag(&["--replace"], "replace", "cancel existing dev jobs first"),
    Param::flag(
        &["--no-wait"],
        "no_wait",
        "return right after submitting (not with --login)",
    ),
    Param::flag(
        &["--login"],
        "login",
        "open a shell once the job is RUNNING (not with --no-wait)",
    ),
    Param {
        ty: ParamType::Int,
        ..Param::option(&["--timeout"], "timeout").help("max seconds to wait for RUNNING")
    },
    POLL_INTERVAL,
];

const DEV: Command = Command::group(
    "dev",
    "start, join and stop a dev job",
    &[
        Command {
            help: Some(
                "Submit a detached dev job. Flags override the [dev] config (bab config show).",
            ),
            params: DEV_START_PARAMS,
            ..Command::leaf("start", "submit a detached dev job", Action::DevStart)
        },
        Command {
            params: &[DEV_JOBID],
            ..Command::leaf("status", "show your dev job", Action::DevStatus)
        },
        Command {
            params: DEV_LOGIN_PARAMS,
            ..Command::leaf("login", "join a running dev job", Action::DevLogin)
        },
        Command {
            params: &[DEV_JOBID],
            ..Command::leaf("stop", "cancel your dev job", Action::DevStop)
        },
    ],
);

const CONFIG: Command = Command::group(
    "config",
    "show where settings come from",
    &[Command::leaf(
        "show",
        "list every setting and its source",
        Action::ConfigShow,
    )],
);

/// `l`: copy of `dev login` with its own short help.
const L: Command = Command {
    params: DEV_LOGIN_PARAMS,
    ..Command::leaf(
        "l",
        "join a running dev job (short for bab dev login)",
        Action::DevLogin,
    )
};

/// The `bab` group (Python `main`), subcommands in registration order.
pub static MAIN: Command = Command {
    name: crate::core::CLI_NAME,
    aliases: &[],
    short_help: None,
    help: Some("Allegro Buddy: a friendly Slurm helper."),
    hidden: false,
    params: &[Param {
        ty: ParamType::Version,
        default: Const::Bool(false),
        help: Some("Show the version and exit."),
        ..Param::option(&["--version"], "version")
    }],
    subcommands: &[
        INFO, QUEUE, WHY, COMPLETION, FREE, NODE, GPUALLOC, JOB, DEV, CONFIG, L,
    ],
    is_group: true,
    invoke_without_command: true,
    action: Some(Action::Info),
};
