//! GPU types, labels and GRES parsing from `core.py`. Owned by P2-C.
//!
//! GPU counts per type are [`GpuCounts`] (`BTreeMap`, deterministic). Python uses
//! insertion-ordered dicts, but no output depends on that order: display order always goes
//! through [`collect_display_gpu_types`], and `fit` sorts.

use crate::settings;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

/// Canonical GPU type -> count.
pub type GpuCounts = BTreeMap<String, i64>;

/// `GPU_TYPE_SYNONYMS` (cluster-neutral; settings `[gpu-aliases]` win).
pub const GPU_TYPE_SYNONYMS: [(&str, &str); 4] = [
    ("tesla_v100", "v100"),
    ("tesla_a100", "a100"),
    ("nvidia_a100", "a100"),
    ("nvidia_h100", "h100"),
];

/// `GPU_TYPE_LABELS` (settings `[gpu-labels]` win).
pub const GPU_TYPE_LABELS: [(&str, &str); 5] = [
    ("a100", "A100"),
    ("h100", "H100"),
    ("h200", "H200"),
    ("v100", "V100"),
    ("generic", "GPU"),
];

pub const GPU_FEATURE_TYPE_HINTS: [&str; 8] = [
    "a5000", "a100", "h100", "h200", "v100", "gtx", "tesla", "titan",
];

/// `GPU_RESOURCE_CHUNK`: groups `type`, `count`, `meta`.
pub static GPU_RESOURCE_CHUNK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?:^|[\s,])(?:gres/)?gpu(?::(?P<type>[A-Za-z0-9._-]+))?[:=](?P<count>\d+)(?:\((?P<meta>[^)]*)\))?",
    )
    .expect("GPU_RESOURCE_CHUNK")
});

static IDX_BLOB: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"IDX:([0-9,-]+)").expect("IDX"));

/// `int()` of a `\d+` match (Unicode digits included, like Python).
fn match_int(text: &str) -> i64 {
    text.parse().unwrap_or_else(|_| {
        text.chars()
            .filter_map(|c| c.to_digit(10).or_else(|| unicode_digit(c)))
            .fold(0i64, |acc, d| {
                acc.saturating_mul(10).saturating_add(i64::from(d))
            })
    })
}

/// The value of a non-ASCII Unicode decimal digit (blocks of ten starting at `…0`).
fn unicode_digit(c: char) -> Option<u32> {
    let u = c as u32;
    const ZEROS: [u32; 12] = [
        0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xFF10, 0x1D7CE,
    ];
    ZEROS
        .iter()
        .find(|&&zero| (zero..zero + 10).contains(&u))
        .map(|zero| u - zero)
}

/// `canonical_gpu_type(gpu_type)`: normalize; empty -> `generic`; then settings aliases,
/// then [`GPU_TYPE_SYNONYMS`], else the normalized name. Reads settings.
pub fn canonical_gpu_type(gpu_type: Option<&str>) -> String {
    try_canonical_gpu_type(gpu_type)
        .unwrap_or_else(|e| panic!("settings::get() before a successful load: {e}"))
}

/// [`canonical_gpu_type`] that reports a broken config (Python's `settings.load()` raising
/// `ConfigError`) instead of assuming settings were loaded; e.g. for click callbacks run
/// during completion.
pub fn try_canonical_gpu_type(gpu_type: Option<&str>) -> Result<String, settings::ConfigError> {
    let normalized = settings::normalize_gpu_type(gpu_type.unwrap_or(""));
    if normalized.is_empty() {
        return Ok("generic".to_owned());
    }
    if let Some(alias) = settings::load()?
        .gpu_alias(&normalized)
        .filter(|a| !a.is_empty())
    {
        return Ok(alias.to_owned());
    }
    Ok(GPU_TYPE_SYNONYMS
        .iter()
        .find(|(from, _)| *from == normalized)
        .map_or(normalized, |(_, to)| (*to).to_owned()))
}

/// `gpu_type_label(gpu_type)`: settings label, [`GPU_TYPE_LABELS`], else
/// `upper().replace("_", "")`.
pub fn gpu_type_label(gpu_type: &str) -> String {
    if let Some(label) = settings::get()
        .gpu_label(gpu_type)
        .filter(|l| !l.is_empty())
    {
        return label.to_owned();
    }
    GPU_TYPE_LABELS
        .iter()
        .find(|(kind, _)| *kind == gpu_type)
        .map_or_else(
            || gpu_type.to_uppercase().replace('_', ""),
            |(_, label)| (*label).to_owned(),
        )
}

/// `parse_gpu_type_counts(spec)` (summed per canonical type, finditer order).
pub fn parse_gpu_type_counts(spec: Option<&str>) -> GpuCounts {
    let mut counts = GpuCounts::new();
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return counts;
    };
    for chunk in GPU_RESOURCE_CHUNK.captures_iter(spec) {
        let kind = canonical_gpu_type(chunk.name("type").map(|m| m.as_str()));
        *counts.entry(kind).or_insert(0) += match_int(&chunk["count"]);
    }
    counts
}

/// `infer_gpu_type_from_features(feature_blob)`.
pub fn infer_gpu_type_from_features(feature_blob: Option<&str>) -> Option<String> {
    let blob = feature_blob.filter(|b| !matches!(*b, "" | "(null)" | "N/A"))?;
    for raw_token in blob.split(',') {
        let token = crate::pyfmt::strip(raw_token);
        if token.is_empty() {
            continue;
        }
        let lower = token.to_lowercase();
        if !GPU_FEATURE_TYPE_HINTS
            .iter()
            .any(|hint| lower.contains(hint))
        {
            continue;
        }
        let kind = canonical_gpu_type(Some(token));
        if kind != "generic" {
            return Some(kind);
        }
    }
    None
}

/// `parse_gpu_count(spec)`: total of [`parse_gpu_type_counts`].
pub fn parse_gpu_count(spec: Option<&str>) -> i64 {
    parse_gpu_type_counts(spec).values().sum()
}

/// `add_gpu_type_counts(target, source)`.
pub fn add_gpu_type_counts(target: &mut GpuCounts, source: &GpuCounts) {
    for (kind, count) in source {
        *target.entry(kind.clone()).or_insert(0) += count;
    }
}

/// `parse_gpu_ids(spec)`: sorted unique ids from every `IDX:...` in the spec.
pub fn parse_gpu_ids(spec: Option<&str>) -> Vec<i64> {
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return Vec::new();
    };
    let ids: std::collections::BTreeSet<i64> = IDX_BLOB
        .captures_iter(spec)
        .flat_map(|c| super::parse::parse_index_blob(&c[1]))
        .collect();
    ids.into_iter().collect()
}

/// `parse_node_gpu_types(spec)`: GPU index -> canonical type (IDX metadata first, then
/// remaining indices filled in chunk order).
pub fn parse_node_gpu_types(spec: Option<&str>) -> BTreeMap<i64, String> {
    let mut mapping = BTreeMap::new();
    let Some(spec) = spec.filter(|s| !s.is_empty()) else {
        return mapping;
    };
    let mut pending: Vec<(String, i64)> = Vec::new();
    let mut total = 0i64;
    for chunk in GPU_RESOURCE_CHUNK.captures_iter(spec) {
        let kind = canonical_gpu_type(chunk.name("type").map(|m| m.as_str()));
        let count = match_int(&chunk["count"]);
        total += count;
        let meta = chunk.name("meta").map_or("", |m| m.as_str());
        if let Some(index) = IDX_BLOB.captures(meta) {
            for gpu_id in super::parse::parse_index_blob(&index[1]) {
                mapping.insert(gpu_id, kind.clone());
            }
            continue;
        }
        pending.push((kind, count));
    }
    let remaining: Vec<i64> = (0..total).filter(|id| !mapping.contains_key(id)).collect();
    let mut cursor = 0;
    for (kind, count) in pending {
        for _ in 0..count {
            if cursor >= remaining.len() {
                break;
            }
            mapping.insert(remaining[cursor], kind.clone());
            cursor += 1;
        }
    }
    mapping
}

/// `collect_display_gpu_types(count_maps)`: types with count > 0, ordered by settings
/// `gpu_labels` order, then `generic`, then the rest sorted.
pub fn collect_display_gpu_types(count_maps: &[&GpuCounts]) -> Vec<String> {
    let discovered: std::collections::BTreeSet<&str> = count_maps
        .iter()
        .flat_map(|map| map.iter())
        .filter(|(_, count)| **count > 0)
        .map(|(kind, _)| kind.as_str())
        .collect();
    let mut ordered: Vec<String> = Vec::new();
    let labels = &settings::get().gpu_labels;
    for kind in labels.iter().map(|(k, _)| k.as_str()).chain(["generic"]) {
        if discovered.contains(kind) && !ordered.iter().any(|o| o == kind) {
            ordered.push(kind.to_owned());
        }
    }
    let rest: Vec<String> = discovered
        .iter()
        .filter(|kind| !ordered.iter().any(|o| o == *kind))
        .map(|kind| (*kind).to_owned())
        .collect();
    ordered.extend(rest);
    ordered
}

/// `format_gpu_type_header(prefix, gpu_type)`.
pub fn format_gpu_type_header(prefix: &str, gpu_type: &str) -> String {
    format!("{prefix}{}", gpu_type_label(gpu_type))
}

#[cfg(test)]
mod tests {
    //! Port of `tests/test_gpu.py` (core parts; the handler tests live with the handlers).
    use super::*;

    fn counts(pairs: &[(&str, i64)]) -> GpuCounts {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn canonical_types() {
        for variant in ["a5000", "rtx_a5000", "RTX-A5000"] {
            assert_eq!(canonical_gpu_type(Some(variant)), "rtx_a5000", "{variant}");
        }
        for variant in ["gtx1080", "gtx_1080ti", "GTX-1080ti", "GTX1080ti"] {
            assert_eq!(canonical_gpu_type(Some(variant)), "gtx_1080ti", "{variant}");
        }
        assert_eq!(canonical_gpu_type(Some("tesla_v100")), "v100");
        assert_eq!(canonical_gpu_type(Some("tesla_a100")), "a100");
        assert_eq!(canonical_gpu_type(Some("nvidia_a100")), "a100");
        assert_eq!(canonical_gpu_type(Some("nvidia_h100")), "h100");
        assert_eq!(canonical_gpu_type(Some("H200")), "h200");
        assert_eq!(canonical_gpu_type(Some("L40S")), "l40s");
        assert_eq!(canonical_gpu_type(None), "generic");
        assert_eq!(canonical_gpu_type(Some("")), "generic");
        assert_eq!(canonical_gpu_type(Some("RTX A5000")), "rtx_a5000");
    }

    #[test]
    fn type_counts() {
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:2")),
            counts(&[("generic", 2)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:rtx_a5000:4")),
            counts(&[("rtx_a5000", 4)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:a5000:4")),
            counts(&[("rtx_a5000", 4)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:a100:2,gpu:h100:1")),
            counts(&[("a100", 2), ("h100", 1)])
        );
        assert!(parse_gpu_type_counts(None).is_empty());
        assert!(parse_gpu_type_counts(Some("")).is_empty());
        assert!(parse_gpu_type_counts(Some("cpu:8")).is_empty());
        assert_eq!(
            parse_gpu_type_counts(Some("gres/gpu:rtx_a5000:4")),
            counts(&[("rtx_a5000", 4)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:a5000:4(IDX:0-3)")),
            counts(&[("rtx_a5000", 4)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gres/gpu:2")),
            counts(&[("generic", 2)])
        );
        assert_eq!(
            parse_gpu_type_counts(Some("gpu:a100:2 gpu:a100:1")),
            counts(&[("a100", 3)])
        );
        assert_eq!(parse_gpu_count(Some("gpu:a100:2,gpu:h100:1,gres/gpu=3")), 6);
        assert_eq!(parse_gpu_count(Some("xgpu:2,mygpu:3")), 0);
    }

    #[test]
    fn gpu_ids() {
        assert_eq!(parse_gpu_ids(Some("gpu:a5000:2(IDX:0,1)")), [0, 1]);
        assert_eq!(parse_gpu_ids(Some("gpu:a5000:4(IDX:0-3)")), [0, 1, 2, 3]);
        assert!(parse_gpu_ids(Some("gpu:a5000:4")).is_empty());
        assert!(parse_gpu_ids(None).is_empty());
        assert!(parse_gpu_ids(Some("")).is_empty());
        assert_eq!(parse_gpu_ids(Some("gpu:a5000:2(IDX:1,0)")), [0, 1]);
    }

    #[test]
    fn node_gpu_types() {
        let result = parse_node_gpu_types(Some("gpu:4"));
        assert_eq!(result.len(), 4);
        assert!(result.values().all(|v| v == "generic"));
        let all = |kind: &str, n: i64| -> BTreeMap<i64, String> {
            (0..n).map(|i| (i, kind.to_owned())).collect()
        };
        assert_eq!(
            parse_node_gpu_types(Some("gpu:a5000:4(IDX:0-3)")),
            all("rtx_a5000", 4)
        );
        let mixed = parse_node_gpu_types(Some("gpu:a100:2(IDX:0-1),gpu:h100:2(IDX:2-3)"));
        assert_eq!(
            mixed.values().collect::<Vec<_>>(),
            ["a100", "a100", "h100", "h100"]
        );
        assert_eq!(
            parse_node_gpu_types(Some("gpu:a100:4(S:0-1)")),
            all("a100", 4)
        );
        assert!(parse_node_gpu_types(None).is_empty());
        assert!(parse_node_gpu_types(Some("")).is_empty());
    }

    #[test]
    fn feature_inference() {
        let f = |s: Option<&str>| infer_gpu_type_from_features(s);
        assert_eq!(
            f(Some("a5000,rtx,nvlink,gpu4")).as_deref(),
            Some("rtx_a5000")
        );
        assert_eq!(f(Some("gtx1080,cuda,pcie")).as_deref(), Some("gtx_1080ti"));
        assert_eq!(f(Some("a100,nvlink,hbm2e")).as_deref(), Some("a100"));
        assert_eq!(f(Some("h100,nvlink,hbm3")).as_deref(), Some("h100"));
        assert_eq!(f(Some("infiniband,ib,opa")), None);
        assert_eq!(f(None), None);
        assert_eq!(f(Some("")), None);
        assert_eq!(f(Some("(null)")), None);
        assert_eq!(f(Some("N/A")), None);
    }
}
