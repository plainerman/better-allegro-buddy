//! Match a pending job's request against the free capacity of each node. Port of
//! `fit.py` (pure: callers load Slurm data and pass it in). Owned by P2-C; port
//! `tests/test_fit.py` (at commit `91689b9`) as unit tests.
//!
//! Requests come from `scontrol --json show job` records ([`JobRecord`]); memory/CPUs
//! from the per-unit fields (`memory_per_tres`, `memory_per_cpu`, `cpus_per_tres`), not
//! `tres_req_str` (stale after `scontrol update`).
//!
//! Python's `Counter.most_common()` order (count desc, ties in first-seen order) matters
//! for `Diagnosis::skipped`; `Counter` iteration (first-seen) for `skip_summary(short)`.

use crate::core::RecordExt;
use crate::core::gpu::{canonical_gpu_type, infer_gpu_type_from_features};
use crate::core::jobs::JobRecord;
use crate::core::nodes::{Node, node_free};
use crate::core::parse::{expand_hostlist, parse_int};
use crate::core::partitions::Partition;
use crate::pyfmt;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::sync::LazyLock;

/// `Dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dim {
    Gpus,
    Cpus,
    Mem,
}

impl Dim {
    /// `'gpus'` / `'cpus'` / `'mem'`.
    pub fn as_str(self) -> &'static str {
        match self {
            Dim::Gpus => "gpus",
            Dim::Cpus => "cpus",
            Dim::Mem => "mem",
        }
    }
}

/// `MemMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemMode {
    Node,
    Cpu,
    Gpu,
}

pub const DIMS: [Dim; 3] = [Dim::Gpus, Dim::Cpus, Dim::Mem];
/// Ties in "which resource binds" go to the scarcest first.
pub const BOTTLENECK_ORDER: [Dim; 3] = [Dim::Gpus, Dim::Mem, Dim::Cpus];
/// `HINT_REASONS`.
pub const HINT_REASONS: [&str; 2] = ["Resources", "Priority"];
pub const MAX_SLOTS: usize = 5;
pub const UNLIMITED_THRESHOLD: i64 = 1 << 62;

pub const SKIP_DOWN: &str = "down or drained";
pub const SKIP_FEATURE: &str = "missing feature";
pub const SKIP_GPU: &str = "no matching GPU";
pub const SKIP_EXCLUDED: &str = "excluded";
pub const SKIP_NOT_REQUESTED: &str = "not in requested nodes";
pub const SKIP_OUTSIDE_RESERVATION: &str = "outside its reservation";
pub const RESERVATION_PREFIX: &str = "reservation ";
/// `_SHORT_SKIPS`.
const SHORT_SKIPS: [(&str, &str); 2] = [
    (SKIP_DOWN, "down"),
    (SKIP_OUTSIDE_RESERVATION, "outside reservation"),
];
/// `_UNUSABLE_FLAGS`.
const UNUSABLE_FLAGS: [&str; 7] = [
    "DOWN",
    "DRAIN",
    "FAIL",
    "INVAL",
    "REBOOT",
    "MAINT",
    "NOT_RESPONDING",
];

static GRES_GPU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"gres/gpu(?::(?P<type>[^:,=]+))?[:=](?P<count>\d+)").expect("_GRES_GPU")
});
static MEM_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGTP]?)$").expect("_MEM_TOKEN")
});

/// `_number(record, key)`: Slurm's `{"set", "infinite", "number"}` wrapper or a plain int.
fn number(record: &JobRecord, key: &str) -> Option<i64> {
    match record.get(key)? {
        Value::Object(wrapper) => {
            let set = wrapper.get("set").is_some_and(pyfmt::json_truthy);
            let infinite = wrapper.get("infinite").is_some_and(pyfmt::json_truthy);
            if !set || infinite {
                return None;
            }
            Some(wrapper.get("number").map_or(0, json_int))
        }
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_u64().map(|u| u as i64))
            .filter(|_| !n.is_f64()),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// `int(value)` of a JSON number (floats truncate).
fn json_int(value: &Value) -> i64 {
    match value {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_u64().map(|u| u as i64))
            .unwrap_or_else(|| n.as_f64().unwrap_or(0.0).trunc() as i64),
        Value::Bool(b) => i64::from(*b),
        _ => 0,
    }
}

/// `record.get(key) or ''` as a string.
fn text<'a>(record: &'a JobRecord, key: &str) -> std::borrow::Cow<'a, str> {
    match record.get(key) {
        Some(Value::String(s)) => s.as_str().into(),
        Some(v) if pyfmt::json_truthy(v) => pyfmt::json_str(v).into(),
        _ => "".into(),
    }
}

/// `(record.get(key) or [])` as strings.
fn string_list(record: &JobRecord, key: &str) -> Vec<String> {
    match record.get(key) {
        Some(Value::Array(items)) => items.iter().map(pyfmt::json_str).collect(),
        Some(Value::String(s)) => s.chars().map(String::from).collect(),
        _ => Vec::new(),
    }
}

/// `match_int` for a `\d+` capture.
fn capture_int(text: &str) -> i64 {
    text.parse().unwrap_or(i64::MAX)
}

/// `_gres_gpu(spec)`: `gres/gpu:RTX-A5000:4` -> `(Some("RTX-A5000"), 4)`; none -> `(None, 0)`.
fn gres_gpu(spec: &str) -> (Option<String>, i64) {
    match GRES_GPU.captures(spec) {
        Some(c) => (
            c.name("type").map(|m| m.as_str().to_owned()),
            capture_int(&c["count"]),
        ),
        None => (None, 0),
    }
}

/// `_with_gpu_count(spec, count)`: the first GPU entry's count replaced.
fn with_gpu_count(spec: &str, count: i64) -> String {
    match GRES_GPU.captures(spec) {
        Some(c) => {
            let whole = c.get(0).expect("match");
            let count_start = c.name("count").expect("count").start();
            format!("{}{count}{}", &spec[..count_start], &spec[whole.end()..])
        }
        None => spec.to_owned(),
    }
}

/// `_tres(spec)`: `cpu=16,mem=40G` -> key/value pairs (later keys win).
fn tres(spec: &str) -> HashMap<String, String> {
    spec.split(',')
        .filter_map(|item| item.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

/// `_array_ranges(spec)`: Python `range`s as (start, stop, step), bad items skipped.
fn array_ranges(spec: &str) -> Vec<(i64, i64, i64)> {
    let spec = spec.split('%').next().unwrap_or("");
    spec.split(',')
        .filter_map(|item| {
            let (span, step) = item.split_once(':').unwrap_or((item, ""));
            let (start, end) = span.split_once('-').unwrap_or((span, ""));
            let start = pyfmt::parse_int(start)?;
            let end = pyfmt::parse_int(if end.is_empty() {
                span.split('-').next().unwrap_or("")
            } else {
                end
            })?;
            let step = if step.is_empty() {
                1
            } else {
                pyfmt::parse_int(step)?
            };
            (step != 0).then_some((start, end + 1, step))
        })
        .collect()
}

fn range_len((start, stop, step): (i64, i64, i64)) -> i64 {
    if step > 0 && start < stop {
        (stop - start - 1) / step + 1
    } else if step < 0 && start > stop {
        (start - stop - 1) / (-step) + 1
    } else {
        0
    }
}

fn range_contains((start, stop, step): (i64, i64, i64), value: i64) -> bool {
    let inside = if step > 0 {
        start <= value && value < stop
    } else {
        stop < value && value <= start
    };
    inside && (value - start) % step == 0
}

/// `_SIMPLE_FEATURE.match(token)`: `^[A-Za-z0-9_.-]+$` (`$` also before a final `\n`).
fn simple_feature(token: &str) -> bool {
    let token = token.strip_suffix('\n').unwrap_or(token);
    !token.is_empty()
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// `_features(spec)`: alternatives of required features (see [`JobRequest::features`]).
fn features(spec: &str) -> Option<Vec<BTreeSet<String>>> {
    let text = pyfmt::strip(spec);
    if text.is_empty() {
        return Some(Vec::new());
    }
    if text.contains('&') && text.contains('|') {
        return None;
    }
    if text.contains('|') {
        let tokens: Vec<&str> = text.split('|').collect();
        return tokens.iter().all(|t| simple_feature(t)).then(|| {
            tokens
                .iter()
                .map(|t| BTreeSet::from([t.to_string()]))
                .collect()
        });
    }
    let tokens: Vec<&str> = text.split(['&', ',']).collect();
    tokens
        .iter()
        .all(|t| simple_feature(t))
        .then(|| vec![tokens.iter().map(|t| t.to_string()).collect()])
}

/// `DIM_LABELS`.
pub fn dim_label(dim: Dim) -> &'static str {
    match dim {
        Dim::Gpus => "GPU",
        Dim::Cpus => "CPU",
        Dim::Mem => "mem",
    }
}

/// `Amount`: GPUs, CPUs and memory (MB) on one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Amount {
    pub gpus: i64,
    pub cpus: i64,
    pub mem: i64,
}

impl Amount {
    pub fn get(&self, dim: Dim) -> i64 {
        match dim {
            Dim::Gpus => self.gpus,
            Dim::Cpus => self.cpus,
            Dim::Mem => self.mem,
        }
    }

    pub fn covers(&self, need: &Amount) -> bool {
        DIMS.iter().all(|&d| self.get(d) >= need.get(d))
    }

    pub fn short_of(&self, need: &Amount) -> Vec<Dim> {
        DIMS.iter()
            .copied()
            .filter(|&d| self.get(d) < need.get(d))
            .collect()
    }
}

/// `NodeCapacity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCapacity {
    pub name: String,
    /// Upper-cased state.
    pub state: String,
    pub partitions: BTreeSet<String>,
    pub features: BTreeSet<String>,
    pub reservation: String,
    pub cpus_total: i64,
    pub cpus_free: i64,
    pub mem_total: i64,
    pub mem_free: i64,
    /// Sorted by type.
    pub gpus_total: Vec<(String, i64)>,
    pub gpus_free: Vec<(String, i64)>,
}

impl NodeCapacity {
    /// `NodeCapacity.from_node(node)`.
    pub fn from_node(node: &Node) -> NodeCapacity {
        let free = node_free(node);
        NodeCapacity {
            name: node.name.clone(),
            state: node.state.to_uppercase(),
            partitions: node.partition_list().map(str::to_owned).collect(),
            features: node
                .active_features
                .split(',')
                .filter(|f| !f.is_empty())
                .map(str::to_owned)
                .collect(),
            reservation: node.reservation.clone(),
            cpus_total: free.cpu_total,
            cpus_free: free.cpu_free,
            mem_total: free.mem_total_mb,
            mem_free: free.mem_free_mb,
            gpus_total: free.gpu_total_by_type.into_iter().collect(),
            gpus_free: free.gpu_free_by_type.into_iter().collect(),
        }
    }

    /// Backfill has set the node aside (`PLANNED` in state).
    pub fn planned(&self) -> bool {
        self.state.contains("PLANNED")
    }

    /// GPUs of one canonical type (all for None), free or total.
    pub fn gpus(&self, kind: Option<&str>, free: bool) -> i64 {
        let counts = if free {
            &self.gpus_free
        } else {
            &self.gpus_total
        };
        counts
            .iter()
            .filter(|(gpu_kind, _)| kind.is_none_or(|k| k == gpu_kind))
            .map(|(_, count)| count)
            .sum()
    }
}

/// `JobRequest` (see `fit.py` for every field's meaning).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRequest {
    pub job_id: String,
    pub display_id: String,
    pub update_id: String,
    pub user: String,
    pub state: String,
    pub reason: String,
    pub partitions: Vec<String>,
    pub node_count: i64,
    pub gpus: i64,
    pub gpu_type: Option<String>,
    pub gpu_kind: Option<String>,
    pub cpus: i64,
    pub cpus_per_gpu: Option<i64>,
    pub min_cpus: i64,
    pub mem_mode: MemMode,
    pub mem_value: i64,
    pub tres_per_node: String,
    pub gpus_per_job: bool,
    /// Alternatives of required features; `Some(vec![])` = no constraint, `None` = an
    /// expression fit does not evaluate.
    pub features: Option<Vec<BTreeSet<String>>>,
    pub excluded: BTreeSet<String>,
    pub required: BTreeSet<String>,
    pub reservation: String,
    pub exclusive: bool,
    pub het: bool,
    pub pending_tasks: i64,
    pub planned_nodes: String,
    pub planned_start: Option<i64>,
}

impl JobRequest {
    /// `JobRequest.from_record(record)`.
    pub fn from_record(record: &JobRecord) -> JobRequest {
        let job_id = record
            .get("job_id")
            .map_or_else(String::new, pyfmt::json_str);
        let array_job = number(record, "array_job_id").unwrap_or(0);
        let array_task = number(record, "array_task_id");
        let task_spec = text(record, "array_task_string");
        let (display_id, update_id, pending_tasks) = if !task_spec.is_empty() {
            (
                format!("{array_job}_[{task_spec}]"),
                array_job.to_string(),
                count_array_tasks(&task_spec),
            )
        } else if let (true, Some(task)) = (array_job != 0, array_task) {
            let id = format!("{array_job}_{task}");
            (id.clone(), id, 1)
        } else {
            (job_id.clone(), job_id.clone(), 1)
        };

        let node_count = number(record, "node_count")
            .filter(|&n| n != 0)
            .unwrap_or(1)
            .max(1);
        let tres_per_node = text(record, "tres_per_node").into_owned();
        let (gpu_type, mut gpus) = gres_gpu(&tres_per_node);
        let mut gpus_per_job = false;
        if gpus == 0 {
            let (_, per_job) = gres_gpu(&text(record, "tres_per_job"));
            let (_, per_task) = gres_gpu(&text(record, "tres_per_task"));
            let tasks = number(record, "tasks").filter(|&t| t != 0).unwrap_or(1);
            let wanted = if per_job != 0 {
                per_job
            } else {
                per_task * tasks
            };
            gpus = pyfmt::ceil_div(wanted, node_count);
            gpus_per_job = gpus > 0;
        }

        let feature_spec = text(record, "features").into_owned();
        let mut gpu_kind = gpu_type.as_deref().map(|t| canonical_gpu_type(Some(t)));
        if gpus != 0 && gpu_kind.is_none() && !feature_spec.contains('|') {
            gpu_kind = infer_gpu_type_from_features(Some(&feature_spec.replace('&', ",")));
        }

        let tres = tres(&text(record, "tres_req_str"));
        let (_, cpus_per_gpu) = gres_gpu(&text(record, "cpus_per_tres"));
        let total_cpus = number(record, "cpus")
            .filter(|&c| c != 0)
            .unwrap_or_else(|| parse_int(tres.get("cpu").map(String::as_str), 1));
        let min_cpus = number(record, "minimum_cpus_per_node")
            .filter(|&c| c != 0)
            .unwrap_or(1);
        let cpus = pyfmt::ceil_div(total_cpus, node_count)
            .max(min_cpus)
            .max(cpus_per_gpu * gpus);

        let (_, mem_per_gpu) = gres_gpu(&text(record, "memory_per_tres"));
        let mem_per_cpu = number(record, "memory_per_cpu");
        let mem_per_node = number(record, "memory_per_node");
        let (mem_mode, mem_value) = if mem_per_gpu != 0 && gpus != 0 {
            (MemMode::Gpu, mem_per_gpu)
        } else if let Some(per_cpu) = mem_per_cpu.filter(|&m| m != 0) {
            (MemMode::Cpu, per_cpu)
        } else if let Some(per_node) = mem_per_node {
            (MemMode::Node, per_node)
        } else {
            let mem = mem_to_mb(tres.get("mem").map_or("0", String::as_str));
            (MemMode::Node, pyfmt::ceil_div(mem, node_count))
        };

        JobRequest {
            job_id,
            display_id,
            update_id,
            user: text(record, "user_name").into_owned(),
            state: string_list(record, "job_state")
                .into_iter()
                .next()
                .unwrap_or_default(),
            reason: text(record, "state_reason").into_owned(),
            partitions: text(record, "partition")
                .split(',')
                .filter(|p| !p.is_empty())
                .map(str::to_owned)
                .collect(),
            node_count,
            gpus,
            gpu_type,
            gpu_kind: if gpus != 0 { gpu_kind } else { None },
            cpus,
            cpus_per_gpu: (cpus_per_gpu != 0).then_some(cpus_per_gpu),
            min_cpus,
            mem_mode,
            mem_value,
            tres_per_node,
            gpus_per_job,
            features: features(&feature_spec),
            excluded: expand_hostlist(Some(&text(record, "excluded_nodes"))),
            required: expand_hostlist(Some(&text(record, "required_nodes"))),
            reservation: text(record, "resv_name").into_owned(),
            exclusive: string_list(record, "shared").iter().any(|s| s == "none"),
            het: number(record, "het_job_id").is_some_and(|id| id != 0),
            pending_tasks,
            planned_nodes: text(record, "scheduled_nodes").into_owned(),
            planned_start: number(record, "start_time").filter(|&t| t != 0),
        }
    }

    /// `--mem=0`: all of a node's memory.
    pub fn whole_node_mem(&self) -> bool {
        self.mem_mode == MemMode::Node && self.mem_value == 0
    }

    pub fn mem_for(&self, gpus: i64, cpus: i64) -> i64 {
        match self.mem_mode {
            MemMode::Gpu => self.mem_value * gpus,
            MemMode::Cpu => self.mem_value * cpus,
            MemMode::Node => self.mem_value,
        }
    }

    /// The per-node request before node-specific adjustments.
    pub fn amount(&self) -> Amount {
        Amount {
            gpus: self.gpus,
            cpus: self.cpus,
            mem: self.mem_for(self.gpus, self.cpus),
        }
    }

    /// What the job needs on this node, including Slurm's own adjustments
    /// (`MaxMemPerCPU` raising the CPUs, exclusive jobs taking the whole node).
    pub fn need_on(&self, node: &NodeCapacity, mem_per_cpu_limit: Option<i64>) -> Amount {
        let mem = if self.whole_node_mem() {
            node.mem_total
        } else {
            self.mem_for(self.gpus, self.cpus)
        };
        let mut cpus = self.cpus;
        if let Some(limit) = mem_per_cpu_limit.filter(|&l| l != 0) {
            cpus = cpus.max(pyfmt::ceil_div(mem, limit));
        }
        let need = Amount {
            gpus: self.gpus,
            cpus,
            mem,
        };
        if self.exclusive {
            let total = self.total_on(node);
            return Amount {
                gpus: need.gpus.max(total.gpus),
                cpus: need.cpus.max(total.cpus),
                mem: need.mem.max(total.mem),
            };
        }
        need
    }

    pub fn free_on(&self, node: &NodeCapacity) -> Amount {
        let gpus = if self.gpus != 0 {
            node.gpus(self.gpu_kind.as_deref(), true)
        } else {
            0
        };
        Amount {
            gpus,
            cpus: node.cpus_free,
            mem: node.mem_free,
        }
    }

    pub fn total_on(&self, node: &NodeCapacity) -> Amount {
        let gpus = if self.gpus != 0 {
            node.gpus(self.gpu_kind.as_deref(), false)
        } else {
            0
        };
        Amount {
            gpus,
            cpus: node.cpus_total,
            mem: node.mem_total,
        }
    }

    /// The current value in the unit the job was submitted with.
    pub fn knob_value(&self, knob: Dim) -> i64 {
        match knob {
            Dim::Mem => self.mem_value,
            Dim::Cpus => self.cpus_per_gpu.filter(|&c| c != 0).unwrap_or(self.cpus),
            Dim::Gpus => self.gpus,
        }
    }

    /// The request after changing one knob, with Slurm's couplings applied.
    pub fn with_knob(&self, knob: Dim, value: i64) -> JobRequest {
        let mut changed = self.clone();
        match knob {
            Dim::Mem => changed.mem_value = value,
            Dim::Cpus => match self.cpus_per_gpu {
                Some(_) => {
                    changed.cpus_per_gpu = Some(value);
                    changed.cpus = (value * self.gpus).max(self.min_cpus);
                }
                None => {
                    changed.cpus = value;
                    changed.min_cpus = self.min_cpus.min(value);
                }
            },
            Dim::Gpus => {
                changed.gpus = value;
                if let Some(per_gpu) = self.cpus_per_gpu {
                    changed.cpus = (per_gpu * value).max(self.min_cpus);
                }
            }
        }
        changed
    }

    /// `--mem` / `--mem-per-cpu` / `--mem-per-gpu` / `--cpus-per-gpu` / `CPUs` / `GPUs`.
    pub fn knob_flag(&self, knob: Dim) -> &'static str {
        match knob {
            Dim::Mem => match self.mem_mode {
                MemMode::Node => "--mem",
                MemMode::Cpu => "--mem-per-cpu",
                MemMode::Gpu => "--mem-per-gpu",
            },
            Dim::Cpus if self.cpus_per_gpu.is_some() => "--cpus-per-gpu",
            Dim::Cpus => "CPUs",
            Dim::Gpus => "GPUs",
        }
    }

    /// The in-place `scontrol update` for a knob, if any.
    pub fn update_command(&self, knob: Dim, value: i64) -> Option<Vec<String>> {
        let field = match knob {
            Dim::Mem => match self.mem_mode {
                MemMode::Node => format!("MinMemoryNode={value}"),
                MemMode::Cpu => format!("MinMemoryCPU={value}"),
                MemMode::Gpu => format!("MemPerTres=gres/gpu:{value}"),
            },
            Dim::Cpus => {
                self.cpus_per_gpu?;
                format!("CpusPerTres=gres/gpu:{value}")
            }
            Dim::Gpus => {
                if self.gpus_per_job || !GRES_GPU.is_match(&self.tres_per_node) {
                    return None;
                }
                format!("TresPerNode={}", with_gpu_count(&self.tres_per_node, value))
            }
        };
        Some(vec![
            "scontrol".to_owned(),
            "update".to_owned(),
            format!("JobId={}", self.update_id),
            field,
        ])
    }
}

/// `NodeFit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeFit {
    pub node: NodeCapacity,
    pub need: Amount,
    pub free: Amount,
    pub total: Amount,
    pub limit: Option<i64>,
}

impl NodeFit {
    pub fn short(&self) -> Vec<Dim> {
        self.free.short_of(&self.need)
    }

    pub fn fits(&self) -> bool {
        self.free.covers(&self.need)
    }

    pub fn holds(&self) -> bool {
        self.total.covers(&self.need)
    }
}

/// `Fix`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fix {
    pub knob: Dim,
    pub old: i64,
    pub new: i64,
    pub request: JobRequest,
    pub nodes: Vec<String>,
    pub command: Option<Vec<String>>,
}

impl Fix {
    /// `(old - new) / old`.
    pub fn cut(&self) -> f64 {
        (self.old - self.new) as f64 / self.old as f64
    }
}

/// `Slot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    pub free: Amount,
    pub nodes: Vec<String>,
}

/// `Diagnosis`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnosis {
    pub request: JobRequest,
    pub rows: Vec<NodeFit>,
    /// `Counter.most_common()` of skip reasons.
    pub skipped: Vec<(String, i64)>,
    pub fixes: Vec<Fix>,
    pub slots: Vec<Slot>,
}

impl Diagnosis {
    pub fn fits(&self) -> Vec<String> {
        self.rows
            .iter()
            .filter(|row| row.fits())
            .map(|row| row.node.name.clone())
            .collect()
    }

    pub fn fits_now(&self) -> bool {
        !self.rows.is_empty() && self.fits().len() as i64 >= self.request.node_count
    }

    pub fn never_fits(&self) -> bool {
        let holding = self.rows.iter().filter(|row| row.holds()).count() as i64;
        !self.rows.is_empty() && holding < self.request.node_count
    }

    /// Skipped nodes of the right kind that are reserved or down.
    pub fn unavailable(&self) -> Vec<(String, i64)> {
        self.skipped
            .iter()
            .filter(|(reason, _)| {
                reason.starts_with(RESERVATION_PREFIX)
                    || reason == SKIP_DOWN
                    || reason == SKIP_OUTSIDE_RESERVATION
            })
            .cloned()
            .collect()
    }

    pub fn planned(&self) -> BTreeSet<String> {
        self.rows
            .iter()
            .filter(|row| row.node.planned())
            .map(|row| row.node.name.clone())
            .collect()
    }

    /// The resources this job asks for (GPUs only for GPU jobs).
    pub fn dims(&self) -> Vec<Dim> {
        DIMS.iter()
            .copied()
            .filter(|&dim| dim != Dim::Gpus || self.request.gpus != 0)
            .collect()
    }

    pub fn short_count(&self, dim: Dim) -> i64 {
        self.rows
            .iter()
            .filter(|row| row.short().contains(&dim))
            .count() as i64
    }

    pub fn best_free(&self, dim: Dim) -> i64 {
        self.rows
            .iter()
            .map(|row| row.free.get(dim))
            .max()
            .unwrap_or(0)
    }

    pub fn max_total(&self, dim: Dim) -> i64 {
        self.rows
            .iter()
            .map(|row| row.total.get(dim))
            .max()
            .unwrap_or(0)
    }

    /// The resource short on the most eligible nodes (ties: [`BOTTLENECK_ORDER`]).
    pub fn bottleneck(&self) -> Dim {
        let dims = self.dims();
        let order: Vec<Dim> = BOTTLENECK_ORDER
            .iter()
            .copied()
            .filter(|d| dims.contains(d))
            .collect();
        let mut best = order[0];
        for &dim in &order[1..] {
            if self.short_count(dim) > self.short_count(best) {
                best = dim;
            }
        }
        best
    }

    pub fn fix_for(&self, knob: Dim) -> Option<&Fix> {
        self.fixes.iter().find(|f| f.knob == knob)
    }

    /// In-place fixes first, then non-GPU, then the smallest cut (first wins ties, like
    /// Python `min`).
    pub fn lead_fix(&self) -> Option<&Fix> {
        let mut best: Option<&Fix> = None;
        for fix in &self.fixes {
            let better = best.is_none_or(|b| {
                let key = |f: &Fix| (f.command.is_none(), f.knob == Dim::Gpus);
                match key(fix).cmp(&key(b)) {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Greater => false,
                    std::cmp::Ordering::Equal => fix.cut() < b.cut(),
                }
            });
            if better {
                best = Some(fix);
            }
        }
        best
    }
}

/// Partition -> `MaxMemPerCPU` (None = no limit).
pub type MemLimits = HashMap<String, Option<i64>>;

/// `mem_to_mb("40G")` -> 40960 (ceil; unparseable 0).
pub fn mem_to_mb(token: &str) -> i64 {
    let Some(c) = MEM_TOKEN.captures(pyfmt::strip(token)) else {
        return 0;
    };
    let value: f64 = c["value"].parse().unwrap_or(0.0);
    let factor = match c["unit"].to_ascii_uppercase().as_str() {
        "K" => 1.0 / 1024.0,
        "G" => 1024.0,
        "T" => 1024.0 * 1024.0,
        "P" => 1024.0 * 1024.0 * 1024.0,
        _ => 1.0,
    };
    (value * factor).ceil() as i64
}

/// `format_mem(mb)`: `40G` or `1536M`.
pub fn format_mem(mb: i64) -> String {
    if mb >= 1024 && mb % 1024 == 0 {
        format!("{}G", mb / 1024)
    } else {
        format!("{mb}M")
    }
}

/// `round_down_mb(mb)`.
pub fn round_down_mb(mb: i64) -> i64 {
    if mb >= 1024 { mb / 1024 * 1024 } else { mb }
}

/// `count_array_tasks(spec)`: tasks in `0-11`, `1,3,5-7`, `0-10:2` or `0-11%2`.
pub fn count_array_tasks(spec: &str) -> i64 {
    array_ranges(spec).into_iter().map(range_len).sum()
}

/// `mem_per_cpu_limits(partitions)`.
pub fn mem_per_cpu_limits(partitions: &[Partition]) -> MemLimits {
    partitions
        .iter()
        .map(|p| {
            let value = parse_int(p.get_opt("MaxMemPerCPU"), 0);
            let limit = (0 < value && value < UNLIMITED_THRESHOLD).then_some(value);
            (p.get_str("PartitionName").to_owned(), limit)
        })
        .collect()
}

/// `find_record(records, jobid)`: by id, else array match (pending first).
pub fn find_record<'a>(records: &'a [JobRecord], jobid: &str) -> Option<&'a JobRecord> {
    let wanted = pyfmt::strip(jobid);
    let id_of = |record: &JobRecord| {
        record
            .get("job_id")
            .map_or_else(String::new, pyfmt::json_str)
    };
    if let Some(record) = records.iter().find(|r| id_of(r) == wanted) {
        return Some(record);
    }
    let (base, task) = wanted.split_once('_').unwrap_or((wanted, ""));
    let task = task.trim_matches(['[', ']']);
    let matches: Vec<&JobRecord> = records
        .iter()
        .filter(|record| {
            let array_job = number(record, "array_job_id").filter(|&n| n != 0);
            if array_job.map_or_else(String::new, |n| n.to_string()) != base {
                return false;
            }
            let array_task = number(record, "array_task_id");
            let spec = text(record, "array_task_string");
            task.is_empty()
                || task == spec
                || array_task.is_some_and(|t| t.to_string() == task)
                || (!spec.is_empty()
                    && !task.is_empty()
                    && task.bytes().all(|b| b.is_ascii_digit())
                    && task.parse::<i64>().is_ok_and(|t| {
                        array_ranges(&spec)
                            .into_iter()
                            .any(|r| range_contains(r, t))
                    }))
        })
        .collect();
    matches
        .iter()
        .find(|r| string_list(r, "job_state").iter().any(|s| s == "PENDING"))
        .or(matches.first())
        .copied()
}

/// `skip_reason(request, node)`: why a node in one of the job's partitions cannot take
/// the job at all, or None.
pub fn skip_reason(request: &JobRequest, node: &NodeCapacity) -> Option<String> {
    let mut flags: Vec<&str> = UNUSABLE_FLAGS.to_vec();
    if !request.reservation.is_empty() {
        if node.reservation != request.reservation {
            return Some(SKIP_OUTSIDE_RESERVATION.to_owned());
        }
        flags.retain(|f| *f != "MAINT");
    } else if !node.reservation.is_empty() {
        return Some(format!("{RESERVATION_PREFIX}{}", node.reservation));
    }
    if flags.iter().any(|flag| node.state.contains(flag)) {
        return Some(SKIP_DOWN.to_owned());
    }
    if request.excluded.contains(&node.name) {
        return Some(SKIP_EXCLUDED.to_owned());
    }
    if !request.required.is_empty() && !request.required.contains(&node.name) {
        return Some(SKIP_NOT_REQUESTED.to_owned());
    }
    if let Some(alternatives) = request.features.as_ref().filter(|a| !a.is_empty())
        && !alternatives.iter().any(|alt| alt.is_subset(&node.features))
    {
        return Some(SKIP_FEATURE.to_owned());
    }
    if request.gpus != 0 && node.gpus(request.gpu_kind.as_deref(), false) == 0 {
        return Some(SKIP_GPU.to_owned());
    }
    None
}

/// `_limit_for(node, partitions, limits)`: the most permissive `MaxMemPerCPU` among the
/// partitions the node shares with the job; None if any of them has no limit.
fn limit_for(node: &NodeCapacity, partitions: &[String], limits: &MemLimits) -> Option<i64> {
    let values: Vec<Option<i64>> = partitions
        .iter()
        .filter(|p| node.partitions.contains(*p))
        .map(|p| limits.get(p).copied().flatten())
        .collect();
    if values.is_empty() || values.iter().any(Option::is_none) {
        return None;
    }
    values.into_iter().flatten().max()
}

fn fits_with(request: &JobRequest, row: &NodeFit) -> bool {
    request
        .free_on(&row.node)
        .covers(&request.need_on(&row.node, row.limit))
}

/// `_candidates(request, rows, knob)`: values below the current one, largest first.
fn candidates(request: &JobRequest, rows: &[NodeFit], knob: Dim) -> Vec<i64> {
    let old = request.knob_value(knob);
    if knob != Dim::Mem {
        return (1..old).rev().collect();
    }
    let mut values = BTreeSet::new();
    for row in rows {
        let mut per_unit = row.free.mem;
        match request.mem_mode {
            MemMode::Cpu => per_unit = per_unit.div_euclid(row.need.cpus.max(1)),
            MemMode::Gpu => per_unit = per_unit.div_euclid(request.gpus.max(1)),
            MemMode::Node => {}
        }
        let value = round_down_mb(per_unit);
        if 0 < value && value < old {
            values.insert(value);
        }
    }
    values.into_iter().rev().collect()
}

/// `_best_fix(request, rows, knob)`.
fn best_fix(request: &JobRequest, rows: &[NodeFit], knob: Dim) -> Option<Fix> {
    match knob {
        Dim::Gpus if request.gpus == 0 => return None,
        Dim::Cpus if request.cpus_per_gpu.is_some() && request.gpus == 0 => return None,
        Dim::Mem if request.whole_node_mem() => return None,
        _ => {}
    }
    for value in candidates(request, rows, knob) {
        let changed = request.with_knob(knob, value);
        let nodes: Vec<String> = rows
            .iter()
            .filter(|row| fits_with(&changed, row))
            .map(|row| row.node.name.clone())
            .collect();
        if nodes.len() as i64 >= request.node_count {
            return Some(Fix {
                knob,
                old: request.knob_value(knob),
                new: value,
                request: changed,
                nodes,
                command: request.update_command(knob, value),
            });
        }
    }
    None
}

/// `_slots(request, rows)`: undominated free amounts (memory rounded down).
fn slots(request: &JobRequest, rows: &[NodeFit]) -> Vec<Slot> {
    let mut groups: Vec<(Amount, Vec<String>)> = Vec::new();
    for row in rows {
        let free = Amount {
            gpus: row.free.gpus,
            cpus: row.free.cpus,
            mem: round_down_mb(row.free.mem),
        };
        if (request.gpus != 0 && free.gpus == 0) || free.cpus == 0 || free.mem == 0 {
            continue;
        }
        match groups.iter_mut().find(|(amount, _)| *amount == free) {
            Some((_, names)) => names.push(row.node.name.clone()),
            None => groups.push((free, vec![row.node.name.clone()])),
        }
    }
    let mut frontier: Vec<&(Amount, Vec<String>)> = groups
        .iter()
        .filter(|(amount, _)| {
            !groups
                .iter()
                .any(|(other, _)| other != amount && other.covers(amount))
        })
        .collect();
    frontier.sort_by_key(|(amount, _)| (-amount.gpus, -amount.cpus, -amount.mem));
    frontier
        .into_iter()
        .take(MAX_SLOTS)
        .map(|(free, nodes)| Slot {
            free: *free,
            nodes: nodes.clone(),
        })
        .collect()
}

/// `diagnose(request, nodes, mem_per_cpu_limits)`: compare the request with every node in
/// the job's partitions.
pub fn diagnose(request: &JobRequest, nodes: &[NodeCapacity], limits: &MemLimits) -> Diagnosis {
    let wanted: BTreeSet<&str> = request.partitions.iter().map(String::as_str).collect();
    let mut rows = Vec::new();
    let mut skipped: Vec<(String, i64)> = Vec::new();
    for node in nodes {
        if !node.partitions.iter().any(|p| wanted.contains(p.as_str())) {
            continue;
        }
        if let Some(reason) = skip_reason(request, node) {
            match skipped.iter_mut().find(|(r, _)| *r == reason) {
                Some((_, count)) => *count += 1,
                None => skipped.push((reason, 1)),
            }
            continue;
        }
        let limit = limit_for(node, &request.partitions, limits);
        rows.push(NodeFit {
            node: node.clone(),
            need: request.need_on(node, limit),
            free: request.free_on(node),
            total: request.total_on(node),
            limit,
        });
    }
    // Counter.most_common(): count descending, ties in first-seen order (stable sort).
    skipped.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let slots = slots(request, &rows);
    let mut diagnosis = Diagnosis {
        request: request.clone(),
        rows,
        skipped,
        fixes: Vec::new(),
        slots,
    };
    if diagnosis.fits_now() || request.exclusive {
        return diagnosis;
    }
    diagnosis.fixes = [Dim::Mem, Dim::Cpus, Dim::Gpus]
        .into_iter()
        .filter_map(|knob| best_fix(request, &diagnosis.rows, knob))
        .collect();
    diagnosis
}

/// `join_names(names, limit)`: `gpu110 +2`.
pub fn join_names(names: &[String], limit: usize) -> String {
    let shown = names[..limit.min(names.len())].join(", ");
    if names.len() > limit {
        format!("{shown} +{}", names.len() - limit)
    } else {
        shown
    }
}

/// `format_free(dim, value)`: memory rounded down so it never overstates what is free.
pub fn format_free(dim: Dim, value: i64) -> String {
    if dim == Dim::Mem {
        format_mem(round_down_mb(value))
    } else {
        value.to_string()
    }
}

/// `format_need(dim, value)`.
pub fn format_need(dim: Dim, value: i64) -> String {
    if dim == Dim::Mem {
        format_mem(value)
    } else {
        value.to_string()
    }
}

fn plural(count: i64, word: &str) -> String {
    if count == 1 {
        format!("{count} {word}")
    } else {
        format!("{count} {word}s")
    }
}

/// `skip_summary(skipped, short=)`: `2 reserved, 1 down` (short) or
/// `2 in reservation maint, 1 down or drained` (long).
pub fn skip_summary(skipped: &[(String, i64)], short: bool) -> String {
    if !short {
        return skipped
            .iter()
            .map(|(reason, count)| {
                if reason.starts_with(RESERVATION_PREFIX) {
                    format!("{count} in {reason}")
                } else {
                    format!("{count} {reason}")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
    }
    let mut totals: Vec<(String, i64)> = Vec::new();
    for (reason, count) in skipped {
        let label = if reason.starts_with(RESERVATION_PREFIX) {
            "reserved"
        } else {
            reason.as_str()
        };
        let label = SHORT_SKIPS
            .iter()
            .find(|(long, _)| *long == label)
            .map_or(label, |(_, short)| short);
        match totals.iter_mut().find(|(l, _)| l == label) {
            Some((_, total)) => *total += count,
            None => totals.push((label.to_owned(), *count)),
        }
    }
    totals
        .iter()
        .map(|(label, count)| format!("{count} {label}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `never_fits_reason(diagnosis)`: e.g. `max 16 CPUs/node`.
pub fn never_fits_reason(diagnosis: &Diagnosis) -> String {
    let need = diagnosis.request.amount();
    let dims = diagnosis.dims();
    for dim in BOTTLENECK_ORDER {
        if dims.contains(&dim) && need.get(dim) > diagnosis.max_total(dim) {
            let unit = match dim {
                Dim::Gpus => " GPUs",
                Dim::Cpus => " CPUs",
                Dim::Mem => "",
            };
            return format!(
                "max {}{unit}/node",
                format_free(dim, diagnosis.max_total(dim))
            );
        }
    }
    if diagnosis.request.node_count > 1 {
        let holding = diagnosis.rows.iter().filter(|row| row.holds()).count();
        return format!(
            "needs {} nodes, {holding} are big enough",
            diagnosis.request.node_count
        );
    }
    "no node is big enough in every resource at once".to_owned()
}

/// `fix_phrase(fix)`: `mem ≤ 9G`, `≤ 3 CPUs/GPU`, `≤ 2 GPUs`.
pub fn fix_phrase(fix: &Fix) -> String {
    match fix.knob {
        Dim::Mem => {
            let unit = match fix.request.mem_mode {
                MemMode::Node => "",
                MemMode::Cpu => "/CPU",
                MemMode::Gpu => "/GPU",
            };
            format!("mem ≤ {}{unit}", format_mem(fix.new))
        }
        Dim::Cpus if fix.request.cpus_per_gpu.is_some() => format!("≤ {} CPUs/GPU", fix.new),
        Dim::Cpus => format!("≤ {}", plural(fix.new, "CPU")),
        Dim::Gpus => format!("≤ {}", plural(fix.new, "GPU")),
    }
}

/// `queue_hint(diagnosis)`: the `Node/Reason` cell of `bab q`.
pub fn queue_hint(diagnosis: &Diagnosis) -> String {
    let request = &diagnosis.request;
    if diagnosis.rows.is_empty() {
        let summary = skip_summary(&diagnosis.skipped, true);
        return if summary.is_empty() {
            "no usable nodes".to_owned()
        } else {
            format!("no usable nodes ({summary})")
        };
    }
    if diagnosis.never_fits() {
        return format!("never fits: {}", never_fits_reason(diagnosis));
    }
    if diagnosis.fits_now() {
        if request.reason == "Priority" {
            return "fits now · held for higher prio".to_owned();
        }
        return format!("fits now on {}", join_names(&diagnosis.fits(), 1));
    }
    if let Some(fix) = diagnosis.lead_fix() {
        return format!("starts now w/ {}", fix_phrase(fix));
    }
    let dim = diagnosis.bottleneck();
    let best = format_free(dim, diagnosis.best_free(dim));
    let need = format_need(dim, request.amount().get(dim));
    format!("{}-bound · max {best}/{need} free", dim_label(dim))
}

/// `tests/slurm_records.py` job records.
#[cfg(test)]
pub mod records {
    use serde_json::{Value, json};

    /// `number(value)`: Slurm's JSON wrapper for optional numbers.
    pub fn n(value: Option<i64>) -> Value {
        match value {
            None => json!({"set": false, "infinite": false, "number": 0}),
            Some(v) => json!({"set": true, "infinite": false, "number": v}),
        }
    }

    /// `{**base, **overrides}`.
    pub fn with(base: &Value, overrides: Value) -> Value {
        let mut record = base.clone();
        for (k, v) in overrides.as_object().expect("overrides object") {
            record[k] = v.clone();
        }
        record
    }

    /// 65177_[0-11]: 4 A5000, 4 CPU and 10G per GPU.
    pub fn array_job() -> Value {
        json!({
            "job_id": 65177,
            "array_job_id": n(Some(65177)),
            "array_task_id": n(None),
            "array_task_string": "0-11",
            "user_name": "alice",
            "job_state": ["PENDING"],
            "state_reason": "Resources",
            "partition": "gpu",
            "node_count": n(Some(1)),
            "tasks": n(Some(1)),
            "cpus": n(Some(16)),
            "minimum_cpus_per_node": n(Some(1)),
            "tres_per_node": "gres/gpu:RTX-A5000:4",
            "tres_per_job": "",
            "tres_per_task": "",
            "cpus_per_tres": "gres/gpu:4",
            "memory_per_tres": "gres/gpu:10240",
            "memory_per_cpu": n(None),
            "memory_per_node": n(Some(10240)),
            "tres_req_str": "cpu=16,mem=40G,node=1,billing=16",
            "features": "",
            "excluded_nodes": "",
            "required_nodes": "",
            "resv_name": "",
            "shared": [],
            "het_job_id": n(Some(0)),
            "scheduled_nodes": "gpu127",
            "start_time": n(Some(1791411962)),
        })
    }

    /// 65160: one GPU typed via --constraint, 4 CPU, --mem=12G, pending on Priority.
    pub fn plain_gpu_job() -> Value {
        with(
            &array_job(),
            json!({
                "job_id": 65160,
                "array_job_id": n(Some(0)),
                "array_task_string": "",
                "state_reason": "Priority",
                "cpus": n(Some(4)),
                "tres_per_node": "gres/gpu:1",
                "cpus_per_tres": "gres/gpu:4",
                "memory_per_tres": "",
                "memory_per_node": n(Some(12288)),
                "tres_req_str": "cpu=4,mem=12G,node=1,billing=4",
                "features": "RTX-A5000",
                "scheduled_nodes": "",
                "start_time": n(Some(0)),
            }),
        )
    }

    /// 64267_[50-142] on big: 4 CPU, --mem=56G, no GPUs.
    pub fn cpu_job() -> Value {
        with(
            &array_job(),
            json!({
                "job_id": 64267,
                "array_job_id": n(Some(64267)),
                "array_task_string": "50-142",
                "partition": "big",
                "cpus": n(Some(4)),
                "tres_per_node": "",
                "cpus_per_tres": "",
                "memory_per_tres": "",
                "memory_per_node": n(Some(57344)),
                "tres_req_str": "cpu=4,mem=56G,node=1,billing=4",
                "scheduled_nodes": "",
                "start_time": n(Some(0)),
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    //! Port of `tests/test_fit.py`.
    use super::records::{array_job, cpu_job, n, plain_gpu_job, with};
    use super::*;
    use crate::core::nodes::testing::{NodeSpec, a5000_dict, node_dict};
    use serde_json::{Value, json};

    fn request(base: Value, overrides: Value) -> JobRequest {
        JobRequest::from_record(&with(&base, overrides))
    }

    fn req() -> JobRequest {
        JobRequest::from_record(&array_job())
    }

    fn node(name: &str, spec: NodeSpec) -> NodeCapacity {
        NodeCapacity::from_node(&node_dict(name, spec))
    }

    fn gpu105() -> NodeCapacity {
        node("gpu105", NodeSpec::default())
    }

    fn cpu_spec() -> NodeSpec {
        NodeSpec {
            partitions: "small,big",
            gres: "(null)".into(),
            cpus: 48,
            mem: 192000,
            features: "Intel",
            ..NodeSpec::default()
        }
    }

    fn cpu_node(name: &str, spec: NodeSpec) -> NodeCapacity {
        node(name, spec)
    }

    fn a5000(name: &str, gpus_free: i64, cpus_free: i64, mem_free: i64) -> NodeCapacity {
        NodeCapacity::from_node(&a5000_dict(name, gpus_free, cpus_free, mem_free, "MIXED"))
    }

    fn a5000_in(
        name: &str,
        gpus_free: i64,
        cpus_free: i64,
        mem_free: i64,
        state: &'static str,
    ) -> NodeCapacity {
        NodeCapacity::from_node(&a5000_dict(name, gpus_free, cpus_free, mem_free, state))
    }

    fn amount(gpus: i64, cpus: i64, mem: i64) -> Amount {
        Amount { gpus, cpus, mem }
    }

    fn diag(request: &JobRequest, nodes: &[NodeCapacity]) -> Diagnosis {
        diagnose(request, nodes, &MemLimits::new())
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn memory_requests() {
        let r = request(
            cpu_job(),
            json!({"cpus": n(Some(2)), "memory_per_node": n(None), "memory_per_cpu": n(Some(2048)), "tres_req_str": "cpu=2,mem=8G,node=1,billing=2"}),
        );
        assert_eq!(r.mem_mode, MemMode::Cpu);
        assert_eq!(r.amount(), amount(0, 2, 4096));
        let r = JobRequest::from_record(&cpu_job());
        assert_eq!((r.gpus, r.gpu_kind.as_deref()), (0, None));
        assert_eq!(r.amount(), amount(0, 4, 57344));
        assert_eq!(
            request(cpu_job(), json!({"memory_per_node": n(None)}))
                .amount()
                .mem,
            57344
        );
    }

    // TestNodeCapacity
    #[test]
    fn node_capacity() {
        let n = node(
            "gpu105",
            NodeSpec {
                gpus_used: 3,
                cpus_alloc: 6,
                mem_alloc: 87040,
                ..NodeSpec::default()
            },
        );
        assert_eq!((n.cpus_free, n.mem_free), (10, 4960));
        assert_eq!(n.gpus(Some("rtx_a5000"), true), 1);
        assert_eq!(n.gpus(None, false), 4);
        assert_eq!(n.gpus(Some("gtx_1080ti"), false), 0);
        let n = node(
            "gpu105",
            NodeSpec {
                spec_mem: 2048,
                ..NodeSpec::default()
            },
        );
        assert_eq!((n.mem_total, n.mem_free), (89952, 89952));
        let n = node(
            "gpu105",
            NodeSpec {
                gres_used: Some("gpu:2".into()),
                ..NodeSpec::default()
            },
        );
        assert_eq!(n.gpus(Some("rtx_a5000"), true), 2);
        assert!(
            node(
                "gpu105",
                NodeSpec {
                    state: "MIXED+PLANNED",
                    ..NodeSpec::default()
                }
            )
            .planned()
        );
        assert!(!gpu105().planned());
        assert_eq!(cpu_node("cmp221", cpu_spec()).gpus(None, false), 0);
    }

    // TestEligibility
    #[test]
    fn eligibility() {
        assert_eq!(skip_reason(&req(), &gpu105()), None);
        let reserved = node(
            "gpu105",
            NodeSpec {
                state: "IDLE+RESERVED+MAINTENANCE",
                reservation: "maint",
                ..NodeSpec::default()
            },
        );
        assert_eq!(
            skip_reason(&req(), &reserved).as_deref(),
            Some("reservation maint")
        );
        let down = node(
            "gpu105",
            NodeSpec {
                state: "DOWN+DRAIN+NOT_RESPONDING",
                ..NodeSpec::default()
            },
        );
        assert_eq!(skip_reason(&req(), &down).as_deref(), Some(SKIP_DOWN));
        let gtx = node(
            "gpu105",
            NodeSpec {
                gres: "gpu:GTX-1080ti:4".into(),
                features: "Intel,GTX-1080ti",
                ..NodeSpec::default()
            },
        );
        assert_eq!(skip_reason(&req(), &gtx).as_deref(), Some(SKIP_GPU));
        let cpu = cpu_node(
            "cmp221",
            NodeSpec {
                partitions: "big",
                ..cpu_spec()
            },
        );
        assert_eq!(
            skip_reason(&request(array_job(), json!({"partition": "gpu,big"})), &cpu).as_deref(),
            Some(SKIP_GPU)
        );
        let r = request(plain_gpu_job(), json!({"features": "Intel&X4112"}));
        let n = node(
            "gpu105",
            NodeSpec {
                features: "Intel,RTX-A5000",
                ..NodeSpec::default()
            },
        );
        assert_eq!(skip_reason(&r, &n).as_deref(), Some(SKIP_FEATURE));
        assert_eq!(skip_reason(&r, &gpu105()), None);
        let r = request(plain_gpu_job(), json!({"features": "[Intel|AMD]*2"}));
        assert_eq!(
            skip_reason(
                &r,
                &node(
                    "gpu105",
                    NodeSpec {
                        features: "",
                        ..NodeSpec::default()
                    }
                )
            ),
            None
        );
        assert_eq!(
            skip_reason(
                &request(array_job(), json!({"excluded_nodes": "gpu105"})),
                &gpu105()
            )
            .as_deref(),
            Some(SKIP_EXCLUDED)
        );
        assert_eq!(
            skip_reason(
                &request(array_job(), json!({"required_nodes": "gpu106"})),
                &gpu105()
            )
            .as_deref(),
            Some(SKIP_NOT_REQUESTED)
        );
        let lab = request(array_job(), json!({"resv_name": "lab"}));
        assert_eq!(
            skip_reason(&lab, &gpu105()).as_deref(),
            Some(SKIP_OUTSIDE_RESERVATION)
        );
        let n = node(
            "gpu105",
            NodeSpec {
                state: "IDLE+RESERVED+MAINTENANCE",
                reservation: "lab",
                ..NodeSpec::default()
            },
        );
        assert_eq!(skip_reason(&lab, &n), None);
    }

    fn live_a5000() -> Vec<NodeCapacity> {
        vec![
            a5000("gpu105", 2, 10, 4960),
            a5000_in("gpu106", 2, 12, 10080, "MIXED+PLANNED"),
            a5000_in("gpu103", 0, 0, 36704, "ALLOCATED"),
        ]
    }

    fn idle(name: &str) -> NodeCapacity {
        node(
            name,
            NodeSpec {
                state: "IDLE",
                ..NodeSpec::default()
            },
        )
    }

    // TestFitsNow
    #[test]
    fn fits_now() {
        let d = diag(&req(), &[idle("gpu110")]);
        assert_eq!(d.fits(), ["gpu110"]);
        assert!(d.fits_now());
        assert!(d.fixes.is_empty());
        assert_eq!(queue_hint(&d), "fits now on gpu110");
        let nodes: Vec<NodeCapacity> = (0..3).map(|i| idle(&format!("gpu11{i}"))).collect();
        assert_eq!(queue_hint(&diag(&req(), &nodes)), "fits now on gpu110 +2");
        let plain = JobRequest::from_record(&plain_gpu_job());
        assert_eq!(
            queue_hint(&diag(&plain, &[idle("gpu105")])),
            "fits now · held for higher prio"
        );
    }

    // TestShortfall
    #[test]
    fn shortfall() {
        let d = diag(&req(), &live_a5000());
        let dims = [Dim::Gpus, Dim::Cpus, Dim::Mem];
        assert_eq!(dims.map(|dim| d.short_count(dim)), [3, 3, 3]);
        assert_eq!(dims.map(|dim| d.best_free(dim)), [2, 12, 36704]);
        assert_eq!(d.bottleneck(), Dim::Gpus);
        assert!(d.fixes.is_empty());
        assert_eq!(queue_hint(&d), "GPU-bound · max 2/4 free");
        let d = diag(
            &req(),
            &[a5000("gpu105", 2, 16, 5000), a5000("gpu110", 4, 8, 10080)],
        );
        assert!(d.fixes.is_empty());
        assert_eq!(d.bottleneck(), Dim::Mem);
        assert_eq!(queue_hint(&d), "mem-bound · max 9G/40G free");
        assert_eq!(diag(&req(), &live_a5000()).planned(), set(&["gpu106"]));
        let d = diag(
            &JobRequest::from_record(&cpu_job()),
            &[cpu_node(
                "cmp221",
                NodeSpec {
                    partitions: "big",
                    ..cpu_spec()
                },
            )],
        );
        assert_eq!(d.dims(), [Dim::Cpus, Dim::Mem]);
    }

    // TestNeverFits
    #[test]
    fn never_fits() {
        let r = request(cpu_job(), json!({"cpus": n(Some(20))}));
        let nodes = [
            cpu_node(
                "cmp221",
                NodeSpec {
                    partitions: "big",
                    cpus: 16,
                    ..cpu_spec()
                },
            ),
            cpu_node(
                "cmp222",
                NodeSpec {
                    partitions: "big",
                    cpus: 16,
                    ..cpu_spec()
                },
            ),
        ];
        let d = diag(&r, &nodes);
        assert!(d.never_fits());
        assert_eq!(queue_hint(&d), "never fits: max 16 CPUs/node");
        assert_eq!(d.fix_for(Dim::Cpus).unwrap().new, 16);
        let nodes = [
            node(
                "gpu061",
                NodeSpec {
                    reservation: "maint",
                    ..NodeSpec::default()
                },
            ),
            node(
                "gpu064",
                NodeSpec {
                    reservation: "maint",
                    ..NodeSpec::default()
                },
            ),
            node(
                "gpu067",
                NodeSpec {
                    state: "DOWN",
                    ..NodeSpec::default()
                },
            ),
        ];
        assert_eq!(
            queue_hint(&diag(&req(), &nodes)),
            "no usable nodes (2 reserved, 1 down)"
        );
        assert_eq!(queue_hint(&diag(&req(), &[])), "no usable nodes");
        let r = request(
            plain_gpu_job(),
            json!({"node_count": n(Some(3)), "cpus": n(Some(12))}),
        );
        let d = diag(&r, &[idle("a"), idle("b")]);
        assert_eq!(never_fits_reason(&d), "needs 3 nodes, 2 are big enough");
    }

    // TestFixes
    #[test]
    fn memory_fix_rounds_down() {
        let d = diag(
            &JobRequest::from_record(&plain_gpu_job()),
            &[live_a5000()[1].clone()],
        );
        let fix = d.fix_for(Dim::Mem).unwrap();
        assert_eq!(
            (fix.old, fix.new, fix.nodes.clone()),
            (12288, 9216, strings(&["gpu106"]))
        );
        assert_eq!(
            fix.command,
            Some(strings(&[
                "scontrol",
                "update",
                "JobId=65160",
                "MinMemoryNode=9216"
            ]))
        );
        assert_eq!(queue_hint(&d), "starts now w/ mem ≤ 9G");
    }

    #[test]
    fn coupled_gpu_fix() {
        let d = diag(&req(), &[a5000("gpu110", 2, 8, 20480)]);
        let fix = d.fix_for(Dim::Gpus).unwrap();
        assert_eq!(fix.new, 2);
        assert_eq!(fix.request.amount(), amount(2, 8, 20480));
        assert_eq!(
            fix.command,
            Some(strings(&[
                "scontrol",
                "update",
                "JobId=65177",
                "TresPerNode=gres/gpu:RTX-A5000:2"
            ]))
        );
        assert_eq!(d.lead_fix(), Some(fix));
        assert_eq!(queue_hint(&d), "starts now w/ ≤ 2 GPUs");
    }

    #[test]
    fn plain_cpu_fixes_are_advice_only() {
        let r = request(plain_gpu_job(), json!({"cpus_per_tres": ""}));
        let d = diag(&r, &[a5000("gpu110", 1, 2, 92000)]);
        let fix = d.fix_for(Dim::Cpus).unwrap();
        assert_eq!((fix.new, fix.command.clone()), (2, None));
        assert_eq!(queue_hint(&d), "starts now w/ ≤ 2 CPUs");
        let r = request(
            cpu_job(),
            json!({"minimum_cpus_per_node": n(Some(4)), "cpus_per_task": n(Some(4)), "tres_per_task": "cpu=4", "memory_per_node": n(Some(8192))}),
        );
        let cpu = cpu_node(
            "cmp221",
            NodeSpec {
                partitions: "big",
                cpus: 48,
                cpus_alloc: 46,
                ..cpu_spec()
            },
        );
        let fix = diag(&r, &[cpu]).fix_for(Dim::Cpus).cloned().unwrap();
        assert_eq!((fix.new, fix.command), (2, None));
        let r = request(
            plain_gpu_job(),
            json!({"cpus": n(Some(2)), "minimum_cpus_per_node": n(Some(2))}),
        );
        assert!(!diag(&r, &[a5000("gpu110", 1, 2, 92000)]).fits_now());
    }

    #[test]
    fn lead_fix_choice() {
        let r = request(
            plain_gpu_job(),
            json!({"cpus_per_tres": "", "cpus": n(Some(8))}),
        );
        let d = diag(
            &r,
            &[a5000("gpu110", 1, 8, 10080), a5000("gpu111", 1, 2, 92000)],
        );
        assert_eq!(d.lead_fix().unwrap().knob, Dim::Mem);
        let d = diag(
            &r,
            &[a5000("gpu110", 1, 8, 6144), a5000("gpu111", 1, 6, 92000)],
        );
        let (cpu_fix, lead) = (d.fix_for(Dim::Cpus).unwrap(), d.lead_fix().unwrap());
        assert_eq!(cpu_fix.command, None);
        assert_eq!(lead.knob, Dim::Mem);
        assert!(cpu_fix.cut() < lead.cut());
        assert!(!queue_hint(&d).contains("CPU"));
        let d = diag(
            &req(),
            &[a5000("gpu110", 4, 16, 30000), a5000("gpu111", 3, 12, 30720)],
        );
        let (gpu_fix, default) = (d.fix_for(Dim::Gpus).unwrap(), d.lead_fix().unwrap());
        assert!(gpu_fix.cut() < default.cut());
        assert_eq!(default.knob, Dim::Mem);
    }
}
