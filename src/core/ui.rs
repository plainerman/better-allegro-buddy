//! Shared output helpers from `core.py`: `warn`, `hint`, `render_table`, `as_cell` and the
//! `*_text` styling helpers. Owned by P2-C (uses P2-A's render API).
//!
//! Styling never calls Slurm: `partition_text`/`node_text` only read settings and the
//! already-loaded node facts ([`super::nodes::loaded_node_facts`]).

use super::nodes::{is_gpu_partition, loaded_node_facts};
use super::parse::{format_mem_value, is_nullish, normalize_sacct_state, parse_int, truncate};
use super::{Record, RecordExt, STATE_STYLES};
use crate::render::console::{self, PrintOptions};
use crate::render::table::{Cell, Justify, Table};
use crate::render::text::Text;

/// One table row (`TableRow`): cells as Python passes them (`str` or `Text`; Python
/// `None`/`""` -> pass `Cell::Str(String::new())`, rendered as a dim `-` by [`as_cell`]).
pub type TableRow = Vec<Cell>;

/// `warn(message)`: `Text.assemble(("warning: ", "bold yellow"), message)` on stderr.
pub fn warn(message: impl Into<Text>) {
    let line = Text::assemble([("warning: ", "bold yellow").into(), message.into().into()]);
    console::stderr().print(&line);
}

/// `hint(text)`: `Text(f"ⓘ {text}", style="dim")` on stdout.
pub fn hint(text: &str) {
    console::stdout().print(&Text::styled(format!("ⓘ {text}"), "dim"));
}

/// `as_cell(value)`: Text as is; empty str -> `Text("-", style="dim")`; str as is.
pub fn as_cell(cell: Cell) -> Cell {
    match cell {
        Cell::Str(s) if s.is_empty() => Cell::Text(muted("-")),
        other => other,
    }
}

/// `render_table(columns, rows, title=, caption=, section_breaks=, width=)`:
/// no rows -> `muted("Nothing to show.")`; else [`crate::render::Table::simple`] with
/// `end_section` on the rows in `section_breaks`, printed with `width`, then the caption
/// as `muted(caption)` on its own line (not Rich's caption).
pub fn render_table(
    columns: &[(&str, Justify)],
    rows: Vec<TableRow>,
    title: Option<&str>,
    caption: Option<&str>,
    section_breaks: &[usize],
    width: Option<usize>,
) {
    let out = console::stdout();
    if rows.is_empty() {
        out.print(&muted("Nothing to show."));
        return;
    }
    let mut table = Table::simple(title);
    for (header, justify) in columns {
        table.add_column(header, *justify);
    }
    for (i, row) in rows.into_iter().enumerate() {
        let cells = row.into_iter().map(as_cell).collect();
        table.add_row(cells, section_breaks.contains(&i));
    }
    out.print_with(
        &table,
        PrintOptions {
            width,
            ..PrintOptions::default()
        },
    );
    if let Some(caption) = caption.filter(|c| !c.is_empty()) {
        out.print(&muted(caption));
    }
}

/// `state_text(state)`: [`super::STATE_STYLES`] or `white`.
pub fn state_text(state: &str) -> Text {
    let style = STATE_STYLES
        .iter()
        .find(|(name, _)| *name == state)
        .map_or("white", |(_, style)| style);
    Text::styled(state, style)
}

fn gpu_partition_no_detect(name: &str) -> bool {
    is_gpu_partition(name, false).unwrap_or(false)
}

/// `partition_text(name)`: `bold green` for GPU partitions (no detection), `bold blue`
/// when nodes are loaded or `gpu_partitions` configured, else `bold`.
pub fn partition_text(name: &str) -> Text {
    if gpu_partition_no_detect(name) {
        return Text::styled(name, "bold green");
    }
    let known = loaded_node_facts().is_some()
        || !crate::settings::get()
            .get_list("cluster", "gpu_partitions")
            .is_empty();
    Text::styled(name, if known { "bold blue" } else { "bold" })
}

/// `node_text(name, partition)`: `(pending)`/empty -> yellow; green for GPU nodes (loaded
/// facts or a GPU partition of the job), else blue.
pub fn node_text(name: &str, partition: &str) -> Text {
    if matches!(name, "" | "(pending)" | "(null)") {
        return Text::styled(if name.is_empty() { "(pending)" } else { name }, "yellow");
    }
    let gpu = loaded_node_facts().is_some_and(|facts| facts.gpu_nodes.contains(name))
        || partition
            .split(',')
            .filter(|p| !p.is_empty())
            .any(gpu_partition_no_detect);
    Text::styled(name, if gpu { "green" } else { "blue" })
}

/// `job_id_text(job_id)`: cyan.
pub fn job_id_text(job_id: &str) -> Text {
    Text::styled(job_id, "cyan")
}

/// `ratio_text(free, total, unit=, free_style=)`: `free/total[unit]` (free in
/// `free_style` or dim when 0, rest dim); total <= 0 -> dim `-`.
pub fn ratio_text(free: i64, total: i64, unit: &str, free_style: &str) -> Text {
    if total <= 0 {
        return muted("-");
    }
    let mut text = Text::new("");
    text.append(
        &free.to_string(),
        Some(if free > 0 { free_style } else { "dim" }),
    );
    text.append("/", Some("dim"));
    text.append(&total.to_string(), Some("dim"));
    if !unit.is_empty() {
        text.append(unit, Some("dim"));
    }
    text
}

/// `muted(value)`: `Text(value, style="dim")` (Python default `"-"`).
pub fn muted(value: &str) -> Text {
    Text::styled(value, "dim")
}

/// `gpu_free_text(free, usable, total)`: `usable[+n]/total`.
pub fn gpu_free_text(free: i64, usable: i64, total: i64) -> Text {
    if total <= 0 {
        return muted("-");
    }
    let mut text = Text::new("");
    text.append(
        &usable.to_string(),
        Some(if usable > 0 { "green" } else { "dim" }),
    );
    if usable < free {
        text.append(&format!("+{}", free - usable), Some("yellow"));
    }
    text.append("/", Some("dim"));
    text.append(&total.to_string(), Some("dim"));
    text
}

/// `exit_code_text(exit_code)`.
pub fn exit_code_text(exit_code: &str) -> Text {
    if is_nullish(Some(exit_code)) {
        return muted("-");
    }
    if super::parse::is_success_exit_code(exit_code) {
        return muted(exit_code);
    }
    Text::styled(exit_code, "red")
}

/// The first non-empty of `keys` (Python `a.get(x) or b.get(y) or ...`).
fn first_set<'a>(job: &'a Record, keys: &[&str]) -> &'a str {
    keys.iter()
        .map(|key| job.get_str(key))
        .find(|value| !value.is_empty())
        .unwrap_or("")
}

/// `build_job_history_rows(jobs)` (uses `format_history_end` with `hooks::now()`).
pub fn build_job_history_rows(jobs: &[Record]) -> Vec<TableRow> {
    let now = crate::hooks::now();
    jobs.iter()
        .map(|job| {
            let job_id = first_set(job, &["JobID", "JobId", "JobIDRaw"]);
            let token = job.get_str("StateToken");
            let state = if token.is_empty() {
                normalize_sacct_state(job.get_str("State"))
            } else {
                token.to_owned()
            };
            vec![
                Cell::Text(job_id_text(job_id)),
                Cell::Str(truncate(job.get_str("JobName"), 26)),
                Cell::Text(state_text(&state)),
                Cell::Text(exit_code_text(job.get_str("ExitCode"))),
                Cell::Str(super::timefmt::format_history_end(job.get_str("End"), now)),
            ]
        })
        .collect()
}

/// `build_dev_job_rows(jobs)`.
pub fn build_dev_job_rows(jobs: &[Record]) -> Vec<TableRow> {
    use super::gpu::parse_gpu_count;
    jobs.iter()
        .map(|job| {
            let state = job.get_str("JobState");
            let partition = job.get_str("Partition");
            let node = job.get_str("NodeList");
            let mem = job
                .get_opt("Mem")
                .unwrap_or_else(|| job.get_str("MinMemoryNode"));
            let gpus = [
                job.get_opt("JOB_GRES"),
                job.get_opt("TresPerNode"),
                job.get_opt("GRES"),
            ]
            .into_iter()
            .map(parse_gpu_count)
            .find(|&count| count != 0)
            .unwrap_or(0);
            let detail = if state == "RUNNING" {
                Cell::Str(super::timefmt::format_eta_now(job.get_str("EndTime")))
            } else {
                let reason = job.get_str("Reason");
                Cell::Text(muted(if reason.is_empty() { "-" } else { reason }))
            };
            let node_cell = if is_nullish(Some(node)) {
                muted("-")
            } else {
                node_text(&truncate(node, 18), partition)
            };
            vec![
                Cell::Text(job_id_text(&super::jobs::display_job_id(job))),
                Cell::Text(state_text(state)),
                Cell::Text(partition_text(partition)),
                Cell::Text(node_cell),
                Cell::Str(parse_int(job.get_opt("NumCPUs"), 0).to_string()),
                Cell::Str(format_mem_value(Some(mem))),
                Cell::Str(gpus.to_string()),
                detail,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Port of the row builders in `tests/test_stats.py` and `test_parsing.py`.
    use super::*;
    use crate::hooks::testing::Ctx;

    #[test]
    fn styled_helpers() {
        let _ctx = Ctx::default().install();
        assert_eq!(ratio_text(3, 0, "", "green").plain(), "-");
        assert_eq!(ratio_text(3, 8, "G", "green").plain(), "3/8G");
        assert_eq!(gpu_free_text(5, 3, 8).plain(), "3+2/8");
        assert_eq!(gpu_free_text(0, 0, 0).plain(), "-");
        assert_eq!(exit_code_text("(null)").plain(), "-");
        assert_eq!(exit_code_text("0:0").plain(), "0:0");
        assert_eq!(node_text("", "gpu").plain(), "(pending)");
        assert_eq!(node_text("(null)", "gpu").plain(), "(null)");
        assert_eq!(state_text("WEIRD").plain(), "WEIRD");
        assert_eq!(partition_text("gpu").plain(), "gpu");
        assert!(matches!(as_cell(Cell::Str(String::new())), Cell::Text(t) if t.plain() == "-"));
        assert!(matches!(as_cell(Cell::Str("x".into())), Cell::Str(s) if s == "x"));
    }
}
