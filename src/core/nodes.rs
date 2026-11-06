//! Nodes: loading, the per-process node facts cache, GPU partition detection and per-node
//! free capacity. Mirrors `core.load_nodes`, `nodes_from_json`, `node_from_props`,
//! `remember_nodes`, `node_facts`, `is_gpu_partition`, `node_is_*`, `node_free`,
//! `node_gpu_type_counts`, `infer_node_gpu_type`. Owned by P2-C.
//!
//! # Node facts cache (plan amendment 6)
//!
//! Python keeps `_NODE_FACTS`, overwritten by EVERY `load_nodes()` call
//! (`remember_nodes`), and lazily filled by `node_facts()` (which calls
//! `load_nodes(strict=False)` if nothing was loaded yet). Styling helpers
//! (`partition_text`, `node_text`) and `is_gpu_partition(detect=False)` only read what is
//! loaded and never call Slurm. Here: a process-wide `RwLock<Option<Arc<NodeFacts>>>`;
//! [`remember_nodes`] replaces it, [`loaded_node_facts`] reads without loading,
//! [`node_facts`] loads on first use. `load_nodes` itself is NOT cached: Python calls Slurm
//! again on every `load_nodes()` and the call log must match. Unit tests keep the facts in
//! their `crate::hooks::testing::Ctx` instead.

use super::Record;
use super::gpu::GpuCounts;
use super::gpu::{parse_gpu_count, parse_gpu_type_counts};
use super::parse::{parse_int, parse_prop_line};
use crate::error::Result;
use crate::pyfmt;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

/// One node, in the shape `nodes_from_json` returns (all values strings, like Python).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Node {
    /// `NodeName`
    pub name: String,
    /// `State`, e.g. `MIXED+PLANNED` (JSON state list joined by `+`).
    pub state: String,
    /// `CPUTot`
    pub cpu_tot: String,
    /// `CPUAlloc`
    pub cpu_alloc: String,
    /// `RealMemory` (MB)
    pub real_memory: String,
    /// `AllocMem` (MB)
    pub alloc_mem: String,
    /// `Gres` (`(null)` when empty)
    pub gres: String,
    /// `GresUsed`
    pub gres_used: String,
    /// `Partitions`, comma-separated
    pub partitions: String,
    /// `ActiveFeatures`, comma-separated
    pub active_features: String,
    /// `Features`, comma-separated
    pub features: String,
    /// `Reservation`
    pub reservation: String,
    /// `SpecMem` (MB)
    pub spec_mem: String,
}

impl Node {
    /// `node.get(key)` by the Python dict key (`"NodeName"`, `"Gres"`, ...); None if unknown.
    pub fn get(&self, key: &str) -> Option<&str> {
        Some(match key {
            "NodeName" => &self.name,
            "State" => &self.state,
            "CPUTot" => &self.cpu_tot,
            "CPUAlloc" => &self.cpu_alloc,
            "RealMemory" => &self.real_memory,
            "AllocMem" => &self.alloc_mem,
            "Gres" => &self.gres,
            "GresUsed" => &self.gres_used,
            "Partitions" => &self.partitions,
            "ActiveFeatures" => &self.active_features,
            "Features" => &self.features,
            "Reservation" => &self.reservation,
            "SpecMem" => &self.spec_mem,
            _ => return None,
        })
    }

    /// Partition names (empty items skipped), in order.
    pub fn partition_list(&self) -> impl Iterator<Item = &str> {
        self.partitions.split(',').filter(|p| !p.is_empty())
    }
}

/// `NodeFacts`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeFacts {
    pub names: BTreeSet<String>,
    pub gpu_nodes: BTreeSet<String>,
    pub gpu_partitions: BTreeSet<String>,
}

/// `scontrol --json show nodes`, the query `ab` makes.
pub const JSON_ARGV: [&str; 4] = ["scontrol", "--json", "show", "nodes"];
/// The same information, ~3x cheaper ([`nodes_from_details_text`], [`super::fast`]).
pub const DETAILS_ARGV: [&str; 5] = ["scontrol", "--oneliner", "--details", "show", "node"];

/// The first node query [`load_nodes`] will make (for [`super::slurm::prefetch`]).
pub fn query_argv() -> Vec<String> {
    if super::fast::enabled() {
        super::slurm::argv(&DETAILS_ARGV)
    } else {
        super::slurm::argv(&JSON_ARGV)
    }
}

/// `load_nodes(strict=strict)`: `scontrol --json show nodes` (unchecked); if that fails or
/// its JSON is unusable, fall back to `scontrol --oneliner show node` (checked iff
/// `strict`). Always calls [`remember_nodes`]. Sequential: the fallback is conditional.
/// With [`super::fast::enabled`], [`DETAILS_ARGV`] first (unchecked); it is used when it
/// converts cleanly, else the Python sequence follows.
pub fn load_nodes(strict: bool) -> Result<Vec<Node>> {
    let nodes = fetch_nodes(strict)?;
    remember_nodes(&nodes);
    Ok(nodes)
}

/// [`load_nodes`] without [`remember_nodes`].
pub fn fetch_nodes(strict: bool) -> Result<Vec<Node>> {
    if super::fast::enabled() {
        let output = super::slurm::run(&DETAILS_ARGV, false)?;
        if output.success()
            && let Some(nodes) = nodes_from_details_text(&output.stdout)
        {
            return Ok(nodes);
        }
    }
    let output = super::slurm::run(&JSON_ARGV, false)?;
    let parsed = if output.success() {
        nodes_from_json(&output.stdout).ok()
    } else {
        None
    };
    match parsed {
        Some(nodes) => Ok(nodes),
        None => {
            let fallback = super::slurm::run(&["scontrol", "--oneliner", "show", "node"], strict)?;
            Ok(if fallback.success() {
                pyfmt::splitlines(&fallback.stdout)
                    .into_iter()
                    .filter(|line| !pyfmt::strip(line).is_empty())
                    .map(|line| node_from_props(&parse_prop_line(line)))
                    .collect()
            } else {
                Vec::new()
            })
        }
    }
}

/// Node states as `scontrol` prints them: base states, then flags in the order of
/// `node_state_flags[]` (`src/common/slurm_protocol_defs.c`).
const TEXT_BASE_STATES: [&str; 7] = [
    "DOWN",
    "IDLE",
    "ALLOCATED",
    "ERROR",
    "MIXED",
    "FUTURE",
    "UNKNOWN",
];
#[rustfmt::skip]
const TEXT_STATE_FLAGS: [&str; 20] = [
    "BLOCKED", "CLOUD", "COMPLETING", "DRAIN", "DYNAMIC_FUTURE", "DYNAMIC_NORM",
    "INVALID_REG", "FAIL", "MAINTENANCE", "POWER_DOWN", "POWER_UP", "POWERED_DOWN",
    "REBOOT_REQUESTED", "REBOOT_ISSUED", "RESERVED", "RESUME", "NOT_RESPONDING", "PLANNED",
    "POWERING_UP", "POWERING_DOWN",
];
/// The flags in the order of the JSON `state` list (data_parser `NODE_STATES`).
#[rustfmt::skip]
const JSON_STATE_FLAGS: [&str; 22] = [
    "RESERVED", "UNDRAIN", "CLOUD", "RESUME", "DRAIN", "COMPLETING", "NOT_RESPONDING",
    "POWERED_DOWN", "FAIL", "POWERING_UP", "MAINTENANCE", "REBOOT_REQUESTED",
    "REBOOT_CANCELED", "POWERING_DOWN", "DYNAMIC_FUTURE", "REBOOT_ISSUED", "PLANNED",
    "INVALID_REG", "POWER_DOWN", "POWER_UP", "POWER_DRAIN", "DYNAMIC_NORM",
];

/// `MIXED+MAINTENANCE+RESERVED` (scontrol) as the JSON list joins: `MIXED+RESERVED+MAINTENANCE`.
/// None for anything scontrol would not print or JSON would show differently (`BLOCKED`
/// is not in the JSON, flags JSON knows but scontrol prints as `?`, and `POWER_DOWN` with
/// `POWERED_DOWN`, which 26.05's JSON also lists as `POWER_DOWN_FORCE`).
fn json_state_from_text(state: &str) -> Option<String> {
    let has = |flag: &str| state.split('+').skip(1).any(|f| f == flag);
    if has("POWER_DOWN") && has("POWERED_DOWN") {
        return None;
    }
    let mut parts = state.split('+');
    let base = parts.next().filter(|b| TEXT_BASE_STATES.contains(b))?;
    let mut previous = None;
    let mut flags = Vec::new();
    for flag in parts {
        let text_index = TEXT_STATE_FLAGS.iter().position(|f| *f == flag)?;
        if previous.is_some_and(|p| p >= text_index) {
            return None;
        }
        previous = Some(text_index);
        flags.push(JSON_STATE_FLAGS.iter().position(|f| *f == flag)?);
    }
    flags.sort_unstable();
    Some(
        std::iter::once(base)
            .chain(flags.into_iter().map(|i| JSON_STATE_FLAGS[i]))
            .collect::<Vec<_>>()
            .join("+"),
    )
}

/// The value of ` KEY=` (or `KEY=` at the very start) in `line[*from..until]`, up to the
/// next space; moves `*from` past it.
fn text_field<'a>(line: &'a str, from: &mut usize, until: usize, key: &str) -> Option<&'a str> {
    let window = &line[*from..until];
    let start = if *from == 0 && window.starts_with(key) && window[key.len()..].starts_with('=') {
        key.len() + 1
    } else {
        let needle = format!(" {key}=");
        window.find(&needle)? + needle.len()
    };
    let value = &window[start..];
    let value = &value[..value.find(' ').unwrap_or(value.len())];
    *from += start + value.len();
    Some(value)
}

/// A count printed with `%u`, as the JSON prints it; None for what JSON would not print
/// as a number (`NO_VAL64`, `INFINITE64`: null).
fn text_number(value: &str) -> Option<String> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value
        .parse::<u64>()
        .ok()
        .filter(|n| *n < u64::MAX - 1)
        .map(|n| n.to_string())
}

/// A comma list as the JSON list joins (`(null)` = none, empty items dropped).
fn text_list(value: &str) -> String {
    if value == "(null)" {
        return String::new();
    }
    value
        .split(',')
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>()
        .join(",")
}

/// One line of `scontrol --oneliner --details show node` (Slurm 24.05 to 26.05
/// `slurm_sprint_node_table`) as [`nodes_from_json`] would have made it from the same
/// node's JSON. None when the line is not exactly in that shape.
fn node_from_details_line(line: &str) -> Option<Node> {
    let line = line.trim_end_matches(' ');
    let mut at = 0;
    let end = line.len();
    let name = text_field(line, &mut at, end, "NodeName")?;
    let cpu_alloc = text_number(text_field(line, &mut at, end, "CPUAlloc")?)?;
    let cpu_tot = text_number(text_field(line, &mut at, end, "CPUTot")?)?;
    let features = text_list(text_field(line, &mut at, end, "AvailableFeatures")?);
    let active_features = text_list(text_field(line, &mut at, end, "ActiveFeatures")?);
    let gres = text_field(line, &mut at, end, "Gres")?;
    let memory_at = at + line[at..].find(" RealMemory=")?;
    let gres_used = text_field(line, &mut at, memory_at, "GresUsed").unwrap_or("");
    at = memory_at;
    let real_memory = text_number(text_field(line, &mut at, end, "RealMemory")?)?;
    let alloc_mem = text_number(text_field(line, &mut at, end, "AllocMem")?)?;
    let state_at = at + line[at..].find(" State=")?;
    let spec_mem = match text_field(line, &mut at, state_at, "MemSpecLimit") {
        Some(value) => text_number(value)?,
        None => "0".to_owned(),
    };
    at = state_at;
    let state = json_state_from_text(text_field(line, &mut at, end, "State")?)?;
    let boot_at = at + line[at..].find(" BootTime=")?;
    let partitions = text_list(text_field(line, &mut at, boot_at, "Partitions").unwrap_or(""));
    at = boot_at;
    text_field(line, &mut at, end, "CurrentWatts")?;
    text_field(line, &mut at, end, "AveWatts")?;
    // Free text follows (Reason=, Comment=, Extra=, ...), then ReservationName= last.
    let tail = &line[at..];
    if tail.contains("Comment=") || tail.contains("Extra=") {
        return None;
    }
    let reservation = match tail.matches("ReservationName=").count() {
        0 => "",
        1 => {
            let (_, value) = tail.rsplit_once(" ReservationName=")?;
            if value.contains(' ') {
                return None;
            }
            value
        }
        _ => return None,
    };
    Some(Node {
        name: name.to_owned(),
        state,
        cpu_tot,
        cpu_alloc,
        real_memory,
        alloc_mem,
        gres: if gres.is_empty() { "(null)" } else { gres }.to_owned(),
        gres_used: gres_used.to_owned(),
        partitions,
        active_features,
        features,
        reservation: reservation.to_owned(),
        spec_mem,
    })
}

/// [`nodes_from_json`] of `scontrol --json show nodes`, from the output of
/// [`DETAILS_ARGV`] instead; None unless every line converts.
pub fn nodes_from_details_text(stdout: &str) -> Option<Vec<Node>> {
    pyfmt::splitlines(stdout)
        .into_iter()
        .filter(|line| !pyfmt::strip(line).is_empty())
        .map(node_from_details_line)
        .collect()
}

/// `n.get(key) or default` for a string field, `str()` of anything else truthy.
fn json_text(node: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    node.get(key)
        .filter(|v| pyfmt::json_truthy(v))
        .map(pyfmt::json_str)
}

/// `sep.join(n.get(key) or [])`; Err when it is not a list of strings (Python TypeError).
fn json_joined(
    node: &serde_json::Map<String, Value>,
    key: &str,
    sep: &str,
) -> std::result::Result<String, String> {
    match node.get(key) {
        None => Ok(String::new()),
        Some(v) if !pyfmt::json_truthy(v) => Ok(String::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{key}: not a string"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(|items| items.join(sep)),
        Some(Value::String(s)) => Ok(s.chars().map(String::from).collect::<Vec<_>>().join(sep)),
        Some(_) => Err(format!("{key}: not a list")),
    }
}

/// `nodes_from_json(stdout)`; Err for invalid JSON / wrong shape (Python ValueError /
/// AttributeError -> fallback). Handles Slurm 22.05 string `state` + `state_flags`.
pub fn nodes_from_json(stdout: &str) -> std::result::Result<Vec<Node>, String> {
    let document: Value = serde_json::from_str(stdout).map_err(|e| e.to_string())?;
    let Value::Object(top) = document else {
        return Err("not an object".to_owned());
    };
    let raw_nodes = match top.get("nodes") {
        None => return Ok(Vec::new()),
        Some(Value::Array(nodes)) => nodes,
        Some(_) => return Err("nodes: not a list".to_owned()),
    };
    let mut result = Vec::with_capacity(raw_nodes.len());
    for raw in raw_nodes {
        let Value::Object(n) = raw else {
            return Err("node: not an object".to_owned());
        };
        let state = match n.get("state") {
            Some(Value::String(s)) if !s.is_empty() => {
                let mut parts = vec![s.to_uppercase()];
                if let Some(Value::Array(flags)) = n.get("state_flags") {
                    for flag in flags {
                        parts.push(flag.as_str().ok_or("state_flags: not a string")?.to_owned());
                    }
                }
                parts.join("+")
            }
            _ => json_joined(n, "state", "+")?,
        };
        let number = |key: &str| n.get(key).map_or_else(|| "0".to_owned(), pyfmt::json_str);
        result.push(Node {
            name: n.get("name").map_or_else(String::new, pyfmt::json_str),
            state,
            cpu_tot: number("cpus"),
            cpu_alloc: number("alloc_cpus"),
            real_memory: number("real_memory"),
            alloc_mem: number("alloc_memory"),
            gres: json_text(n, "gres").unwrap_or_else(|| "(null)".to_owned()),
            gres_used: json_text(n, "gres_used").unwrap_or_default(),
            partitions: json_joined(n, "partitions", ",")?,
            active_features: json_joined(n, "active_features", ",")?,
            features: json_joined(n, "features", ",")?,
            reservation: json_text(n, "reservation").unwrap_or_default(),
            spec_mem: json_text(n, "specialized_memory").unwrap_or_else(|| "0".to_owned()),
        });
    }
    Ok(result)
}

/// `node_from_props(props)` for the oneliner fallback.
pub fn node_from_props(props: &Record) -> Node {
    let get = |key: &str, default: &str| {
        props
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_owned())
    };
    let truthy = |key: &str| props.get(key).filter(|v| !v.is_empty()).cloned();
    Node {
        name: get("NodeName", ""),
        state: get("State", ""),
        cpu_tot: get("CPUTot", "0"),
        cpu_alloc: get("CPUAlloc", "0"),
        real_memory: get("RealMemory", "0"),
        alloc_mem: get("AllocMem", "0"),
        gres: truthy("Gres").unwrap_or_else(|| "(null)".to_owned()),
        gres_used: get("GresUsed", ""),
        partitions: get("Partitions", ""),
        active_features: get("ActiveFeatures", ""),
        features: truthy("AvailableFeatures")
            .or_else(|| truthy("Features"))
            .unwrap_or_default(),
        reservation: get("Reservation", ""),
        spec_mem: get("MemSpecLimit", "0"),
    }
}

type FactsSlot = RwLock<Option<Arc<NodeFacts>>>;

fn with_facts_slot<T>(f: impl FnOnce(&FactsSlot) -> T) -> T {
    #[cfg(test)]
    {
        f(&crate::hooks::testing::ctx().node_facts)
    }
    #[cfg(not(test))]
    {
        static FACTS: FactsSlot = RwLock::new(None);
        f(&FACTS)
    }
}

/// `remember_nodes(nodes)`: replace the facts cache.
pub fn remember_nodes(nodes: &[Node]) {
    let mut facts = NodeFacts::default();
    for node in nodes {
        facts.names.insert(node.name.clone());
        if parse_gpu_count(Some(&node.gres)) > 0 {
            facts.gpu_nodes.insert(node.name.clone());
            facts
                .gpu_partitions
                .extend(node.partition_list().map(str::to_owned));
        }
    }
    with_facts_slot(|slot| *slot.write().expect("node facts") = Some(Arc::new(facts)));
}

/// The facts of the last `load_nodes`, without calling Slurm (`_NODE_FACTS`).
pub fn loaded_node_facts() -> Option<Arc<NodeFacts>> {
    with_facts_slot(|slot| slot.read().expect("node facts").clone())
}

/// `node_facts()`: loads nodes (`strict=False`) if none were loaded yet. Only Ctrl-C
/// during that load is an error.
pub fn node_facts() -> Result<Arc<NodeFacts>> {
    if let Some(facts) = loaded_node_facts() {
        return Ok(facts);
    }
    load_nodes(false)?;
    Ok(loaded_node_facts().unwrap_or_default())
}

/// `is_gpu_partition(name, detect=detect)`: `[cluster] gpu-partitions` if configured,
/// else whether a loaded node with GPUs is in it; `detect=true` may load nodes
/// (via [`node_facts`]; only Ctrl-C makes that fail).
pub fn is_gpu_partition(name: &str, detect: bool) -> Result<bool> {
    let configured = crate::settings::get().get_list("cluster", "gpu_partitions");
    if !configured.is_empty() {
        return Ok(configured.iter().any(|p| p == name));
    }
    let facts = if detect {
        Some(node_facts()?)
    } else {
        loaded_node_facts()
    };
    Ok(facts.is_some_and(|f| f.gpu_partitions.contains(name)))
}

/// `node_is_usable(node)`: no [`super::BAD_NODE_FLAGS`] substring in the upper-cased state.
pub fn node_is_usable(node: &Node) -> bool {
    let state = node.state.to_uppercase();
    !super::BAD_NODE_FLAGS
        .iter()
        .any(|flag| state.contains(flag))
}

/// `node_is_idle(node)`: `IDLE` in state.
pub fn node_is_idle(node: &Node) -> bool {
    node.state.to_uppercase().contains("IDLE")
}

/// `node_is_mixed(node)`: `MIXED` in state.
pub fn node_is_mixed(node: &Node) -> bool {
    node.state.to_uppercase().contains("MIXED")
}

/// `infer_node_gpu_type(node)`: from ActiveFeatures, AvailableFeatures (never set in our
/// node shape, always None), Features.
pub fn infer_node_gpu_type(node: &Node) -> Option<String> {
    use super::gpu::infer_gpu_type_from_features;
    infer_gpu_type_from_features(Some(&node.active_features))
        .or_else(|| infer_gpu_type_from_features(Some(&node.features)))
}

/// `node_gpu_type_counts(node)`: `Gres` counts, generic GPUs re-typed by features.
pub fn node_gpu_type_counts(node: &Node) -> GpuCounts {
    let mut counts = parse_gpu_type_counts(Some(&node.gres));
    let generic = counts.get("generic").copied().unwrap_or(0);
    if generic <= 0 {
        return counts;
    }
    let Some(inferred) = infer_node_gpu_type(node) else {
        return counts;
    };
    counts.remove("generic");
    *counts.entry(inferred).or_insert(0) += generic;
    counts
}

/// `NodeFree`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeFree {
    pub cpu_total: i64,
    pub cpu_free: i64,
    pub mem_total_mb: i64,
    pub mem_free_mb: i64,
    pub gpu_total_by_type: GpuCounts,
    pub gpu_free_by_type: GpuCounts,
}

/// `node_free(node)`: the one place the free-capacity math lives.
pub fn node_free(node: &Node) -> NodeFree {
    let cpu_total = parse_int(Some(&node.cpu_tot), 0);
    let mem_total = parse_int(Some(&node.real_memory), 0) - parse_int(Some(&node.spec_mem), 0);
    let total = node_gpu_type_counts(node);
    let mut used = parse_gpu_type_counts(Some(&node.gres_used));
    let generic_used = if total.contains_key("generic") {
        0
    } else {
        used.remove("generic").unwrap_or(0)
    };
    if generic_used != 0 && total.len() == 1 {
        let only_kind = total.keys().next().expect("one kind").clone();
        *used.entry(only_kind).or_insert(0) += generic_used;
    }
    NodeFree {
        cpu_total,
        cpu_free: (cpu_total - parse_int(Some(&node.cpu_alloc), 0)).max(0),
        mem_total_mb: mem_total,
        mem_free_mb: (mem_total - parse_int(Some(&node.alloc_mem), 0)).max(0),
        gpu_free_by_type: total
            .iter()
            .map(|(kind, count)| {
                (
                    kind.clone(),
                    (count - used.get(kind).copied().unwrap_or(0)).max(0),
                )
            })
            .collect(),
        gpu_total_by_type: total,
    }
}

/// Test builders mirroring `tests/slurm_records.py`.
#[cfg(test)]
pub mod testing {
    use super::Node;

    /// `node_dict(...)` options (defaults: an empty A5000 node).
    #[derive(Clone)]
    pub struct NodeSpec {
        pub state: &'static str,
        pub partitions: &'static str,
        pub gres: String,
        pub gpus_used: i64,
        pub gres_used: Option<String>,
        pub cpus: i64,
        pub cpus_alloc: i64,
        pub mem: i64,
        pub mem_alloc: i64,
        pub spec_mem: i64,
        pub features: &'static str,
        pub reservation: &'static str,
    }

    impl Default for NodeSpec {
        fn default() -> NodeSpec {
            NodeSpec {
                state: "MIXED",
                partitions: "gpu",
                gres: "gpu:RTX-A5000:4".to_owned(),
                gpus_used: 0,
                gres_used: None,
                cpus: 16,
                cpus_alloc: 0,
                mem: 92000,
                mem_alloc: 0,
                spec_mem: 0,
                features: "Intel,X4112,RTX-A5000",
                reservation: "",
            }
        }
    }

    /// `node_dict(name, **spec)`.
    pub fn node_dict(name: &str, spec: NodeSpec) -> Node {
        let gres_used = spec.gres_used.clone().unwrap_or_else(|| {
            if spec.gres == "(null)" {
                String::new()
            } else {
                let kind = spec
                    .gres
                    .rsplit_once(':')
                    .map_or(spec.gres.as_str(), |(k, _)| k);
                format!("{kind}:{}(IDX:N/A)", spec.gpus_used)
            }
        });
        Node {
            name: name.to_owned(),
            state: spec.state.to_owned(),
            cpu_tot: spec.cpus.to_string(),
            cpu_alloc: spec.cpus_alloc.to_string(),
            real_memory: spec.mem.to_string(),
            alloc_mem: spec.mem_alloc.to_string(),
            spec_mem: spec.spec_mem.to_string(),
            gres: spec.gres.clone(),
            gres_used,
            partitions: spec.partitions.to_owned(),
            active_features: spec.features.to_owned(),
            features: spec.features.to_owned(),
            reservation: spec.reservation.to_owned(),
        }
    }

    /// `a5000_dict(name, gpus_free=, cpus_free=, mem_free=, state=, gpus=4)`.
    pub fn a5000_dict(
        name: &str,
        gpus_free: i64,
        cpus_free: i64,
        mem_free: i64,
        state: &'static str,
    ) -> Node {
        node_dict(
            name,
            NodeSpec {
                state,
                gpus_used: 4 - gpus_free,
                cpus_alloc: 16 - cpus_free,
                mem_alloc: 92000 - mem_free,
                ..NodeSpec::default()
            },
        )
    }

    /// A node with just a state (`{'State': state}`).
    pub fn with_state(state: &str) -> Node {
        Node {
            state: state.to_owned(),
            ..Node::default()
        }
    }
}

#[cfg(test)]
mod tests {
    //! Port of the node parts of `tests/test_slurm.py`, `test_gpu.py` and `test_stats.py`.
    use super::testing::with_state;
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, ok};
    use crate::hooks::testing::Ctx;

    /// Three Allegro nodes (Slurm 24.11.5): `scontrol --oneliner --details show node` and
    /// the same nodes' `scontrol --json show nodes` (the fields bab reads).
    const DETAILS_TEXT: &str = "\
NodeName=gpu061 Arch=x86_64 CoresPerSocket=6  CPUAlloc=0 CPUEfctv=24 CPUTot=24 CPULoad=0.00 AvailableFeatures=Intel,GTX-1080ti ActiveFeatures=Intel,GTX-1080ti Gres=gpu:GTX-1080ti:4 GresDrain=N/A GresUsed=gpu:GTX-1080ti:0(IDX:N/A) NodeAddr=gpu061 NodeHostName=gpu061 Version=24.11.5 OS=Linux 6.12.96+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.96-1 (2026-07-20)  RealMemory=32000 AllocMem=0 FreeMem=24170 Sockets=2 Boards=1 State=IDLE+MAINTENANCE+RESERVED ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A Partitions=gpu,gpushort,gpusmall,gpumicro  BootTime=2026-07-29T11:06:39 SlurmdStartTime=2026-10-13T15:17:17 LastBusyTime=2026-10-13T15:17:16 ResumeAfterTime=None CfgTRES=cpu=24,mem=32000M,billing=24 AllocTRES= CurrentWatts=0 AveWatts=0  ReservationName=gpu-switch
NodeName=gpu062 Arch=x86_64 CoresPerSocket=6  CPUAlloc=1 CPUEfctv=12 CPUTot=12 CPULoad=1.04 AvailableFeatures=Intel,GTX-1080ti ActiveFeatures=Intel,GTX-1080ti Gres=gpu:GTX-1080ti:4 GresDrain=N/A GresUsed=gpu:GTX-1080ti:1(IDX:0) NodeAddr=gpu062 NodeHostName=gpu062 Version=24.11.5 OS=Linux 6.12.96+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.96-1 (2026-07-20)  RealMemory=64000 AllocMem=49152 FreeMem=11703 Sockets=2 Boards=1 MemSpecLimit=2048 State=MIXED+PLANNED ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A Partitions=gpu,gpushort,gpusmall,gpumicro  BootTime=2026-07-29T11:06:51 SlurmdStartTime=2026-10-13T15:17:17 LastBusyTime=2026-11-04T14:23:13 ResumeAfterTime=None CfgTRES=cpu=12,mem=62.50G,billing=12 AllocTRES=cpu=1,mem=48G CurrentWatts=0 AveWatts=0 
NodeName=gpu074 CoresPerSocket=6  CPUAlloc=0 CPUEfctv=12 CPUTot=12 CPULoad=0.00 AvailableFeatures=(null) ActiveFeatures=(null) Gres=(null) NodeAddr=gpu074 NodeHostName=gpu074  RealMemory=64000 AllocMem=0 FreeMem=N/A Sockets=2 Boards=1 State=DOWN+DRAIN+NOT_RESPONDING ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A BootTime=None SlurmdStartTime=None LastBusyTime=Unknown ResumeAfterTime=None CfgTRES=cpu=12,mem=62.50G,billing=12 AllocTRES= CurrentWatts=0 AveWatts=0  Reason=Not responding : reboot timed out [slurm@2026-07-01T10:13:49]
";
    const DETAILS_JSON: &str = r#"{"nodes": [
      {"name": "gpu061", "state": ["IDLE", "RESERVED", "MAINTENANCE"], "cpus": 24,
       "alloc_cpus": 0, "real_memory": 32000, "alloc_memory": 0,
       "gres": "gpu:GTX-1080ti:4", "gres_used": "gpu:GTX-1080ti:0(IDX:N/A)",
       "partitions": ["gpu", "gpushort", "gpusmall", "gpumicro"],
       "active_features": ["Intel", "GTX-1080ti"], "features": ["Intel", "GTX-1080ti"],
       "reservation": "gpu-switch", "specialized_memory": 0},
      {"name": "gpu062", "state": ["MIXED", "PLANNED"], "cpus": 12, "alloc_cpus": 1,
       "real_memory": 64000, "alloc_memory": 49152, "gres": "gpu:GTX-1080ti:4",
       "gres_used": "gpu:GTX-1080ti:1(IDX:0)",
       "partitions": ["gpu", "gpushort", "gpusmall", "gpumicro"],
       "active_features": ["Intel", "GTX-1080ti"], "features": ["Intel", "GTX-1080ti"],
       "reservation": "", "specialized_memory": 2048},
      {"name": "gpu074", "state": ["DOWN", "DRAIN", "NOT_RESPONDING"], "cpus": 12,
       "alloc_cpus": 0, "real_memory": 64000, "alloc_memory": 0, "gres": "",
       "gres_used": "", "partitions": [], "active_features": [], "features": [],
       "reservation": "", "specialized_memory": 0}
    ]}"#;

    #[test]
    fn details_text_converts_to_what_the_json_says() {
        assert_eq!(
            nodes_from_details_text(DETAILS_TEXT),
            Some(nodes_from_json(DETAILS_JSON).unwrap())
        );
        let line = |from: &str, to: &str| DETAILS_TEXT.lines().next().unwrap().replace(from, to);
        // anything that scontrol and the JSON could show differently: the JSON instead
        for odd in [
            line("State=IDLE+MAINTENANCE", "State=IDLE+BLOCKED+MAINTENANCE"),
            line("State=IDLE+MAINTENANCE", "State=IDLE+?+MAINTENANCE"),
            // 25.05 prints EXTERNAL; its JSON does not list it
            line("State=IDLE+MAINTENANCE", "State=IDLE+EXTERNAL+MAINTENANCE"),
            // 26.05's JSON adds POWER_DOWN_FORCE
            line(
                "State=IDLE+MAINTENANCE",
                "State=IDLE+POWERED_DOWN+POWER_DOWN",
            ),
            line("State=IDLE+MAINTENANCE+RESERVED", "State=MAINT+RESERVED"),
            line(
                "State=IDLE+MAINTENANCE+RESERVED",
                "State=IDLE+RESERVED+MAINTENANCE",
            ),
            line("AveWatts=0 ", "AveWatts=0  Comment=x"),
            line(
                "AveWatts=0 ",
                "AveWatts=0  Reason=see ReservationName=y [root@now]",
            ),
            line("RealMemory=32000", "RealMemory=18446744073709551614"),
            line("NodeName=", "Node="),
        ] {
            assert_eq!(nodes_from_details_text(&odd), None, "{odd}");
        }
    }

    const SCONTROL_NODE_OUTPUT: &str = r#"{
  "nodes": [
    {
      "name": "gpu001", "cpus": 48, "alloc_cpus": 0,
      "state": ["IDLE"],
      "gres": "gpu:rtx_a5000:4", "gres_used": "gpu:rtx_a5000:0(IDX:N/A)",
      "real_memory": 256000, "alloc_memory": 0,
      "partitions": ["gpumicro", "gpushort", "gpusmall", "gpu"],
      "active_features": ["a5000", "rtx", "nvlink", "gpu4"],
      "features": ["a5000", "rtx", "nvlink", "gpu4"]
    },
    {
      "name": "gpu002", "cpus": 48, "alloc_cpus": 16,
      "state": ["MIXED"],
      "gres": "gpu:rtx_a5000:4", "gres_used": "gpu:rtx_a5000:2(IDX:0,1)",
      "real_memory": 256000, "alloc_memory": 65536,
      "partitions": ["gpusmall", "gpu"],
      "active_features": ["a5000", "rtx", "nvlink"],
      "features": ["a5000", "rtx", "nvlink"]
    },
    {
      "name": "cmp001", "cpus": 64, "alloc_cpus": 0,
      "state": ["IDLE"],
      "gres": "", "gres_used": "",
      "real_memory": 512000, "alloc_memory": 0,
      "partitions": ["small", "big"],
      "active_features": ["amd", "zen3"],
      "features": ["amd", "zen3"]
    }
  ]
}"#;

    fn load_with(output: crate::core::slurm::Output) -> Vec<Node> {
        let _ctx = Ctx::default().slurm(FakeSlurm::always(output)).install();
        load_nodes(true).unwrap()
    }

    #[test]
    fn parses_scontrol_json() {
        let nodes = load_with(ok(SCONTROL_NODE_OUTPUT));
        assert_eq!(nodes.len(), 3);
        let names: Vec<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["gpu001", "gpu002", "cmp001"]);
        let gpu001 = &nodes[0];
        assert_eq!(gpu001.cpu_tot, "48");
        assert_eq!(gpu001.cpu_alloc, "0");
        assert_eq!(gpu001.state, "IDLE");
        assert_eq!(gpu001.gres, "gpu:rtx_a5000:4");
        assert_eq!(gpu001.gres_used, "gpu:rtx_a5000:0(IDX:N/A)");
        assert_eq!(gpu001.partitions, "gpumicro,gpushort,gpusmall,gpu");
        assert_eq!(gpu001.reservation, "");
        assert_eq!(gpu001.spec_mem, "0");
        let gpu002 = &nodes[1];
        assert_eq!(gpu002.state, "MIXED");
        assert_eq!(gpu002.cpu_alloc, "16");
        assert_eq!(gpu002.gres_used, "gpu:rtx_a5000:2(IDX:0,1)");
        assert_eq!(nodes[2].gres, "(null)");
        assert!(load_with(ok(r#"{"nodes": []}"#)).is_empty());
    }

    #[test]
    fn oneliner_fallback() {
        let fake = FakeSlurm::new(|argv| {
            Some(if argv[1] == "--json" {
                ok("scontrol: error: no data_parser plugin\n")
            } else {
                ok(
                    "NodeName=gpu1 State=MIXED CPUTot=8 Gres= AvailableFeatures=a5000 Features=x MemSpecLimit=512 Partitions=gpu\n\n",
                )
            })
        });
        let _ctx = Ctx::default().slurm(fake).install();
        let nodes = load_nodes(true).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].gres, "(null)");
        assert_eq!(nodes[0].features, "a5000");
        assert_eq!(nodes[0].spec_mem, "512");
        assert_eq!(nodes[0].cpu_alloc, "0");
    }

    #[test]
    fn bad_json_is_an_error() {
        for raw in [
            "",
            "[]",
            r#"{"nodes": [1]}"#,
            r#"{"nodes": [{"partitions": [1]}]}"#,
        ] {
            assert!(nodes_from_json(raw).is_err(), "{raw}");
        }
        assert!(nodes_from_json("{}").unwrap().is_empty());
    }

    #[test]
    fn facts_and_gpu_partitions() {
        let _ctx = Ctx::default().install();
        assert!(loaded_node_facts().is_none());
        assert!(!is_gpu_partition("gpu", false).unwrap());
        let mut gpu = with_state("IDLE");
        gpu.name = "g1".into();
        gpu.gres = "gpu:4".into();
        gpu.partitions = "gpu,gpu-a100".into();
        let mut cpu = with_state("IDLE");
        cpu.name = "c1".into();
        cpu.gres = "(null)".into();
        cpu.partitions = "small".into();
        remember_nodes(&[gpu, cpu]);
        let facts = loaded_node_facts().unwrap();
        assert_eq!(facts.gpu_nodes.iter().collect::<Vec<_>>(), ["g1"]);
        assert_eq!(facts.names.len(), 2);
        assert!(is_gpu_partition("gpu-a100", false).unwrap());
        assert!(!is_gpu_partition("small", true).unwrap());
        let configured = crate::settings::testing::with_preset(
            "allegro",
            &["[cluster]", "gpu-partitions = small"],
        );
        let _ctx = Ctx::default().settings(configured).install();
        assert!(is_gpu_partition("small", true).unwrap());
        assert!(!is_gpu_partition("gpu", true).unwrap());
    }

    #[test]
    fn states() {
        for state in ["IDLE", "MIXED", "ALLOCATED", "COMPLETING", ""] {
            assert!(node_is_usable(&with_state(state)), "{state}");
        }
        for state in [
            "DOWN",
            "IDLE+DRAIN",
            "FAIL",
            "MAINT",
            "REBOOT",
            "INVAL",
            "MIXED+DRAINING",
        ] {
            assert!(!node_is_usable(&with_state(state)), "{state}");
        }
        assert!(node_is_idle(&with_state("IDLE")));
        assert!(node_is_idle(&with_state("IDLE+COMPLETING")));
        for state in ["MIXED", "ALLOCATED", "DOWN"] {
            assert!(!node_is_idle(&with_state(state)));
        }
        assert!(node_is_mixed(&with_state("MIXED")));
        assert!(node_is_mixed(&with_state("MIXED+DRAINING")));
        for state in ["IDLE", "ALLOCATED"] {
            assert!(!node_is_mixed(&with_state(state)));
        }
    }

    fn gres_node(gres: &str, features: &str) -> Node {
        Node {
            gres: gres.to_owned(),
            active_features: features.to_owned(),
            ..Node::default()
        }
    }

    #[test]
    fn gpu_type_inference() {
        let n = |active: &str, features: &str| Node {
            active_features: active.into(),
            features: features.into(),
            ..Node::default()
        };
        assert_eq!(
            infer_node_gpu_type(&n("a5000,rtx,nvlink", "")).as_deref(),
            Some("rtx_a5000")
        );
        assert_eq!(
            infer_node_gpu_type(&n("infiniband", "a100,nvlink")).as_deref(),
            Some("a100")
        );
        assert_eq!(infer_node_gpu_type(&n("infiniband", "storage")), None);
        assert_eq!(infer_node_gpu_type(&Node::default()), None);
    }

    #[test]
    fn gpu_type_counts() {
        let c = |node: &Node| node_gpu_type_counts(node).into_iter().collect::<Vec<_>>();
        assert_eq!(
            c(&gres_node("gpu:rtx_a5000:4", "")),
            [("rtx_a5000".to_owned(), 4)]
        );
        assert_eq!(
            c(&gres_node("gpu:4", "a5000,rtx,nvlink")),
            [("rtx_a5000".to_owned(), 4)]
        );
        assert!(c(&gres_node("(null)", "")).is_empty());
        assert!(c(&Node::default()).is_empty());
        assert_eq!(
            c(&gres_node("gpu:a100:2,gpu:h100:2", "")),
            [("a100".to_owned(), 2), ("h100".to_owned(), 2)]
        );
    }
}
