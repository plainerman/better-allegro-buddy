//! Partitions: loading, access, ordering, per-partition stats, dev partition choice.
//! Mirrors `core.load_partitions`, `partition_allowed`, `sort_partitions`,
//! `get_accessible_partitions`, `choose_partition`, `PartitionStats`, `_summarize`,
//! `summarize_partitions`, `summarize_selected_partitions`. Owned by P2-C.

use super::gpu::GpuCounts;
use super::gpu::add_gpu_type_counts;
use super::nodes::Node;
use super::nodes::{is_gpu_partition, node_free, node_is_idle, node_is_mixed, node_is_usable};
use super::parse::{is_nullish, parse_prop_line};
use super::timefmt::parse_slurm_time;
use super::{Record, RecordExt};
use crate::error::Result;
use crate::error::die;
use crate::hooks::GroupSet;
use crate::pyfmt;
use std::collections::{BTreeSet, HashMap};

/// A partition: the `scontrol --oneliner show partition` record (`PartitionName`,
/// `MaxTime`, `AllowGroups`, `DenyGroups`, `Default`, `MaxMemPerCPU`, ...).
pub type Partition = Record;

/// `scontrol --oneliner show partition`.
pub const ARGV: [&str; 4] = ["scontrol", "--oneliner", "show", "partition"];

/// `load_partitions(strict=strict)`: [`ARGV`] (checked iff strict); on failure
/// (unchecked) `[]`.
pub fn load_partitions(strict: bool) -> Result<Vec<Partition>> {
    let output = super::slurm::run(&ARGV, strict)?;
    if !output.success() {
        return Ok(Vec::new());
    }
    Ok(pyfmt::splitlines(&output.stdout)
        .into_iter()
        .filter(|line| !pyfmt::strip(line).is_empty())
        .map(parse_prop_line)
        .collect())
}

fn group_list(value: &str) -> impl Iterator<Item = &str> {
    value.split(',').filter(|item| !item.is_empty())
}

/// `partition_allowed(partition, groups)` (DenyGroups first, then AllowGroups; `ALL`
/// and nullish mean everyone).
pub fn partition_allowed(partition: &Partition, groups: &(impl GroupSet + ?Sized)) -> bool {
    let deny = partition.get_str("DenyGroups");
    if !is_nullish(Some(deny))
        && pyfmt::strip(deny).to_uppercase() != "ALL"
        && group_list(deny).any(|g| groups.contains(g))
    {
        return false;
    }
    let allow = partition.get_opt("AllowGroups").unwrap_or("ALL");
    if is_nullish(Some(allow)) || pyfmt::strip(allow).to_uppercase() == "ALL" {
        return true;
    }
    group_list(allow).any(|g| groups.contains(g))
}

/// `_max_time(partition)`: MaxTime in seconds, None/0 -> infinity.
fn max_time(partition: &Partition) -> f64 {
    match parse_slurm_time(partition.get_opt("MaxTime").unwrap_or("UNLIMITED")) {
        Some(seconds) if seconds != 0 => seconds as f64,
        _ => f64::INFINITY,
    }
}

/// `sort_partitions(partitions)`: GPU partitions first, families (`gpu-a100`,
/// `gpu-a100:test`) together, shortest family MaxTime first, then family name, MaxTime,
/// name. MaxTime None/0 counts as infinity (`parse_slurm_time(...) or inf`). Uses
/// `is_gpu_partition(name)` with detection (may load nodes, like Python; only Ctrl-C
/// during that load is an error).
pub fn sort_partitions(partitions: Vec<Partition>) -> Result<Vec<Partition>> {
    fn family(p: &Partition) -> &str {
        p.get_str("PartitionName").split(':').next().unwrap_or("")
    }
    let mut shortest: HashMap<String, f64> = HashMap::new();
    for p in &partitions {
        let entry = shortest
            .entry(family(p).to_owned())
            .or_insert(f64::INFINITY);
        *entry = entry.min(max_time(p));
    }
    let mut keyed = Vec::with_capacity(partitions.len());
    for p in partitions {
        let not_gpu = !is_gpu_partition(p.get_str("PartitionName"), true)?;
        let key = (
            not_gpu,
            shortest[family(&p)],
            family(&p).to_owned(),
            max_time(&p),
            p.get_str("PartitionName").to_owned(),
        );
        keyed.push((key, p));
    }
    keyed.sort_by(|(a, _), (b, _)| {
        a.0.cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
            .then(a.3.total_cmp(&b.3))
            .then_with(|| a.4.cmp(&b.4))
    });
    Ok(keyed.into_iter().map(|(_, p)| p).collect())
}

/// `get_accessible_partitions(include_all=...)`: load (strict), filter by group access
/// unless `include_all`, sort.
pub fn get_accessible_partitions(include_all: bool) -> Result<Vec<Partition>> {
    let groups = super::users::current_groups();
    let mut partitions = load_partitions(true)?;
    if !include_all {
        partitions.retain(|p| partition_allowed(p, groups));
    }
    sort_partitions(partitions)
}

/// `choose_partition(partition_name, gpu_count, cpus, time_limit)` for `dev start`:
/// resolves `auto` to the `Default=YES` partition, validates access and MaxTime; every
/// failure is a `die()` with Python's exact message.
pub fn choose_partition(partition_name: &str, time_limit: &str) -> Result<String> {
    let mut partitions: Vec<(String, Partition)> = Vec::new();
    for item in get_accessible_partitions(true)? {
        let name = item.get_str("PartitionName").to_owned();
        match partitions.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = item,
            None => partitions.push((name, item)),
        }
    }
    let groups = super::users::current_groups();
    let mut selected = partition_name.to_owned();
    if partition_name == "auto" {
        selected = partitions
            .iter()
            .find(|(_, item)| item.get_opt("Default") == Some("YES"))
            .map(|(name, _)| name.clone())
            .unwrap_or_default();
        if selected.is_empty() {
            let accessible: Vec<&str> = partitions
                .iter()
                .filter(|(_, item)| partition_allowed(item, groups))
                .map(|(name, _)| name.as_str())
                .collect();
            let listed = if accessible.is_empty() {
                "none".to_owned()
            } else {
                accessible.join(", ")
            };
            return Err(die(format!(
                "this cluster has no default partition; pass --partition or set `partition` in [dev] of {} (you can use: {listed})",
                crate::settings::user_config_path().display()
            )));
        }
    }
    let Some((_, partition)) = partitions.iter().find(|(name, _)| *name == selected) else {
        return Err(die(format!("unknown partition: {selected}")));
    };
    if !partition_allowed(partition, groups) {
        return Err(die(format!(
            "you do not appear to have access to partition {selected}"
        )));
    }
    let requested = parse_slurm_time(time_limit);
    let max = parse_slurm_time(partition.get_opt("MaxTime").unwrap_or("UNLIMITED"));
    if let (Some(requested), Some(max)) = (requested, max)
        && requested > max
    {
        return Err(die(format!(
            "requested time {time_limit} exceeds the {selected} partition limit; set --partition explicitly"
        )));
    }
    Ok(selected)
}

/// `PartitionStats`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PartitionStats {
    pub nodes: i64,
    pub idle_nodes: i64,
    pub cpu_total: i64,
    pub cpu_free: i64,
    pub mem_total_gb: i64,
    pub mem_free_gb: i64,
    pub gpu_total: i64,
    pub gpu_free: i64,
    pub gpu_usable: i64,
    pub gpu_total_by_type: GpuCounts,
    pub gpu_free_by_type: GpuCounts,
    pub gpu_usable_by_type: GpuCounts,
}

/// `_summarize(names, nodes, merged=...)`; the merged bucket is keyed `""`.
fn summarize(names: &[String], nodes: &[Node], merged: bool) -> HashMap<String, PartitionStats> {
    let wanted: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let mut summary: HashMap<String, PartitionStats> = if merged {
        HashMap::from([(String::new(), PartitionStats::default())])
    } else {
        names
            .iter()
            .map(|n| (n.clone(), PartitionStats::default()))
            .collect()
    };
    for node in nodes {
        if !node_is_usable(node) {
            continue;
        }
        let here: Vec<&str> = node
            .partitions
            .split(',')
            .filter(|p| wanted.contains(p))
            .collect();
        if here.is_empty() {
            continue;
        }
        let free = node_free(node);
        let gpu_free: i64 = free.gpu_free_by_type.values().sum();
        let gpu_total: i64 = free.gpu_total_by_type.values().sum();
        let is_idle = node_is_idle(node);
        let is_schedulable = is_idle || node_is_mixed(node);
        let buckets: Vec<&str> = if merged { vec![""] } else { here };
        for name in buckets {
            let bucket = summary.get_mut(name).expect("bucket per name");
            bucket.nodes += 1;
            bucket.idle_nodes += i64::from(is_idle);
            bucket.cpu_total += free.cpu_total;
            bucket.cpu_free += free.cpu_free;
            bucket.mem_total_gb += pyfmt::round_div(free.mem_total_mb, 1024);
            bucket.mem_free_gb += pyfmt::round_div(free.mem_free_mb, 1024);
            bucket.gpu_total += gpu_total;
            bucket.gpu_free += gpu_free;
            add_gpu_type_counts(&mut bucket.gpu_total_by_type, &free.gpu_total_by_type);
            add_gpu_type_counts(&mut bucket.gpu_free_by_type, &free.gpu_free_by_type);
            if is_schedulable {
                bucket.gpu_usable += gpu_free;
                add_gpu_type_counts(&mut bucket.gpu_usable_by_type, &free.gpu_free_by_type);
            }
        }
    }
    summary
}

/// `summarize_partitions(names, nodes=nodes)`: stats per partition name (every name
/// present, even without nodes). Memory per node is `round(mb / 1024)` (half to even)
/// BEFORE summing, like Python. Callers pass `load_nodes()` for Python's `nodes=None`.
pub fn summarize_partitions(names: &[String], nodes: &[Node]) -> HashMap<String, PartitionStats> {
    summarize(names, nodes, false)
}

/// `summarize_selected_partitions(partition_names, nodes=nodes)`: one merged bucket
/// (each node counted once); empty set -> empty stats.
pub fn summarize_selected_partitions(names: &BTreeSet<String>, nodes: &[Node]) -> PartitionStats {
    if names.is_empty() {
        return PartitionStats::default();
    }
    let names: Vec<String> = names.iter().cloned().collect();
    summarize(&names, nodes, true)
        .remove("")
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    //! Port of `tests/test_gpu_counts.py`, the partition parts of `test_stats.py`,
    //! `test_slurm.py` and `test_parsing.py`.
    use super::*;
    use crate::core::slurm::fake::{FakeSlurm, ok};
    use crate::hooks::testing::Ctx;

    fn rec(pairs: &[(&str, &str)]) -> Record {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn groups(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|g| g.to_string()).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn node(name: &str, state: &str, gres: &str, partitions: &str, gres_used: &str) -> Node {
        Node {
            name: name.into(),
            state: state.into(),
            cpu_tot: "48".into(),
            cpu_alloc: "0".into(),
            real_memory: "256000".into(),
            alloc_mem: "0".into(),
            gres: gres.into(),
            gres_used: gres_used.into(),
            partitions: partitions.into(),
            ..Node::default()
        }
    }

    const A5000: &str = "gpu:RTX-A5000:4";

    fn stats(names: &[&str], nodes: &[Node]) -> HashMap<String, PartitionStats> {
        let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        summarize_partitions(&names, nodes)
    }

    #[test]
    fn free_gpus_per_partition() {
        let s = &stats(
            &["gpusmall"],
            &[node(
                "gpu001",
                "IDLE",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:0(IDX:N/A)",
            )],
        )["gpusmall"];
        assert_eq!((s.gpu_total, s.gpu_free, s.gpu_usable), (4, 4, 4));
        let s = &stats(
            &["gpusmall"],
            &[node(
                "gpu001",
                "MIXED",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:2(IDX:0-1)",
            )],
        )["gpusmall"];
        assert_eq!((s.gpu_total, s.gpu_free, s.gpu_usable), (4, 2, 2));
        assert_eq!(s.gpu_free_by_type.get("rtx_a5000"), Some(&2));
        let s = &stats(
            &["gpusmall"],
            &[node(
                "gpu001",
                "ALLOCATED",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:4(IDX:0-3)",
            )],
        )["gpusmall"];
        assert_eq!((s.gpu_free, s.gpu_usable), (0, 0));
        let s = &stats(
            &["gpusmall"],
            &[node("gpu001", "IDLE+DRAIN", A5000, "gpusmall", "")],
        )["gpusmall"];
        assert_eq!(s.gpu_total, 0);
        let s = &stats(
            &["gpusmall"],
            &[node(
                "gpu001",
                "ALLOCATED",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:2(IDX:0,1)",
            )],
        )["gpusmall"];
        assert_eq!((s.gpu_free, s.gpu_usable), (2, 0));
        let s = &stats(
            &["gpusmall"],
            &[node(
                "gpu001",
                "MIXED",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:6",
            )],
        )["gpusmall"];
        assert_eq!((s.gpu_free, s.gpu_usable), (0, 0));
    }

    #[test]
    fn selected_partitions() {
        assert_eq!(
            summarize_selected_partitions(&BTreeSet::new(), &[]),
            PartitionStats::default()
        );
        let nodes = [
            node("gpu001", "IDLE", A5000, "gpusmall,gpu", ""),
            node("cmp001", "IDLE", "(null)", "small", ""),
        ];
        let r = summarize_selected_partitions(&groups(&["gpusmall"]), &nodes);
        assert_eq!((r.nodes, r.gpu_total), (1, 4));
        let r = summarize_selected_partitions(
            &groups(&["gpusmall"]),
            &[node(
                "gpu001",
                "MIXED",
                A5000,
                "gpusmall",
                "gpu:RTX-A5000:1(IDX:0)",
            )],
        );
        assert_eq!((r.gpu_free, r.gpu_usable), (3, 3));
        let nodes = [
            node("gpu001", "IDLE", A5000, "gpusmall", ""),
            node("gpu002", "IDLE", A5000, "gpu", ""),
        ];
        let r = summarize_selected_partitions(&groups(&["gpusmall", "gpu"]), &nodes);
        assert_eq!(r.gpu_total, 8);
        let r = summarize_selected_partitions(
            &groups(&["gpusmall", "gpu"]),
            &[node("g", "IDLE", A5000, "gpusmall,gpu", "")],
        );
        assert_eq!(r.nodes, 1);
    }

    #[test]
    fn access_control() {
        assert!(partition_allowed(
            &rec(&[("AllowGroups", "ALL")]),
            &groups(&["students", "users"])
        ));
        assert!(partition_allowed(
            &rec(&[("AllowGroups", "gpu_users,researchers")]),
            &groups(&["gpu_users", "students"])
        ));
        assert!(!partition_allowed(
            &rec(&[("AllowGroups", "gpu_users")]),
            &groups(&["students"])
        ));
        assert!(!partition_allowed(
            &rec(&[("AllowGroups", "gpu_users")]),
            &groups(&[])
        ));
        assert!(partition_allowed(&rec(&[]), &groups(&[])));
        assert!(partition_allowed(
            &rec(&[("AllowGroups", "groupA,groupB,groupC")]),
            &groups(&["groupB"])
        ));
        let denied = rec(&[("AllowGroups", "ALL"), ("DenyGroups", "students")]);
        assert!(!partition_allowed(&denied, &groups(&["students"])));
        assert!(partition_allowed(&denied, &groups(&["staff"])));
        assert!(partition_allowed(
            &rec(&[("DenyGroups", "all")]),
            &groups(&["x"])
        ));
        assert!(partition_allowed(
            &rec(&[("AllowGroups", "(null)")]),
            &groups(&[])
        ));
    }

    fn with_partitions(groups_: &[&str], stdout: &'static str) -> crate::hooks::testing::Guard {
        let fake = FakeSlurm::new(move |argv| {
            Some(if argv[1] == "--json" {
                ok(r#"{"nodes": []}"#)
            } else {
                ok(stdout)
            })
        });
        Ctx::default().groups(groups_).slurm(fake).install()
    }

    #[test]
    fn choosing_a_partition() {
        let _ctx = with_partitions(
            &["users"],
            "PartitionName=small Default=YES MaxTime=1-00:00:00\nPartitionName=gpu AllowGroups=gpu MaxTime=UNLIMITED\n",
        );
        assert_eq!(choose_partition("auto", "08:00:00").unwrap(), "small");
        assert_eq!(
            choose_partition("auto", "2-00:00:00").unwrap_err(),
            crate::error::AbError::Exit(1)
        );
        assert_eq!(
            choose_partition("gpu", "1:00").unwrap_err(),
            crate::error::AbError::Exit(1)
        );
        assert_eq!(
            choose_partition("nope", "1:00").unwrap_err(),
            crate::error::AbError::Exit(1)
        );
        assert_eq!(choose_partition("small", "nonsense").unwrap(), "small");
    }
}
