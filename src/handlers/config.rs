//! `bab config show`. Port of `handlers/config.py`. Owned by P3-F.

use crate::core::RecordExt;
use crate::core::gpu::{collect_display_gpu_types, gpu_type_label};
use crate::core::nodes::{is_gpu_partition, load_nodes, node_gpu_type_counts};
use crate::core::partitions::load_partitions;
use crate::core::slurm::join2;
use crate::core::ui::{TableRow, muted, render_table};
use crate::error::Result;
use crate::render::console;
use crate::render::table::{Cell, Justify};
use crate::render::text::Text;
use crate::settings;

/// `display_key(section, key)`: option keys as typed (`log-dir`), lookup keys verbatim.
pub fn display_key(section: &str, key: &str) -> String {
    let options = settings::SECTIONS
        .iter()
        .find(|(name, _)| *name == section)
        .and_then(|(_, keys)| *keys);
    match options {
        None => key.to_owned(),
        Some(_) => key.replace('_', "-"),
    }
}

/// `cmd_config_show()`.
pub fn cmd_config_show() -> Result<()> {
    let _prefetch = crate::core::slurm::prefetch(vec![
        crate::core::nodes::query_argv(),
        crate::core::slurm::argv(&crate::core::partitions::ARGV),
    ]);
    let current = settings::get();
    let out = console::stdout();
    let mut head = Text::new("");
    head.append("Preset: ", Some("dim"));
    head.append(current.preset.as_deref().unwrap_or("none"), Some("bold"));
    head.append(&format!(" ({})", current.preset_reason), Some("dim"));
    for (label, path) in [
        ("Site file", settings::site_config_path()),
        ("User file", settings::user_config_path()),
    ] {
        head.append(&format!("\n{label}: "), Some("dim"));
        head.append(&path.display().to_string(), None);
        head.append(
            if path.exists() { "" } else { " (not present)" },
            Some("dim"),
        );
    }
    out.print(&head);
    out.newline();

    let mut items: Vec<&settings::Entry> = current.values.iter().collect();
    items.sort_by(|a, b| (&a.section, &a.key).cmp(&(&b.section, &b.key)));
    let rows: Vec<TableRow> = items
        .iter()
        .map(|e| {
            vec![
                Cell::Text(Text::new(format!(
                    "[{}] {}",
                    e.section,
                    display_key(&e.section, &e.key)
                ))),
                Cell::Text(Text::new(e.value.clone())),
                Cell::Text(muted(&e.source)),
            ]
        })
        .collect();
    let breaks: Vec<usize> = (0..items.len().saturating_sub(1))
        .filter(|&i| items[i].section != items[i + 1].section)
        .collect();
    render_table(
        &[
            ("Setting", Justify::Left),
            ("Value", Justify::Left),
            ("Source", Justify::Left),
        ],
        rows,
        Some("Settings"),
        None,
        &breaks,
        None,
    );
    out.newline();

    // Both queries are read-only, unchecked and always made: run them together.
    let (nodes, partitions) = join2(|| load_nodes(false), || load_partitions(false));
    let (nodes, partitions) = (nodes?, partitions?);
    let default = partitions
        .iter()
        .find(|p| p.get_opt("Default") == Some("YES"))
        .map(|p| p.get_str("PartitionName").to_owned());
    let counts: Vec<_> = nodes.iter().map(node_gpu_type_counts).collect();
    let gpu_types = collect_display_gpu_types(&counts.iter().collect::<Vec<_>>());
    let mut gpu_partitions = Vec::new();
    for p in &partitions {
        let name = p.get_str("PartitionName");
        if is_gpu_partition(name, true)? {
            gpu_partitions.push(name);
        }
    }
    let or_none = |value: String| {
        if value.is_empty() {
            Cell::Text(muted("none"))
        } else {
            Cell::Str(value)
        }
    };
    let detected: Vec<TableRow> = vec![
        vec![
            Cell::Str("GPU partitions".to_owned()),
            or_none(gpu_partitions.join(", ")),
        ],
        vec![
            Cell::Str("GPU types".to_owned()),
            or_none(
                gpu_types
                    .iter()
                    .map(|t| format!("{} ({t})", gpu_type_label(t)))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
        ],
        vec![
            Cell::Str("Default partition".to_owned()),
            or_none(default.unwrap_or_default()),
        ],
    ];
    if nodes.is_empty() {
        out.print(&muted("Slurm did not answer; nothing detected."));
        return Ok(());
    }
    render_table(
        &[("Setting", Justify::Left), ("Value", Justify::Left)],
        detected,
        Some("Detected from Slurm"),
        None,
        &[],
        None,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_keys() {
        assert_eq!(display_key("dev", "log_dir"), "log-dir");
        assert_eq!(display_key("cluster", "gpu_partitions"), "gpu-partitions");
        assert_eq!(display_key("gpu-aliases", "rtx_a5000"), "rtx_a5000");
        assert_eq!(display_key("other", "a_b"), "a_b");
    }
}
