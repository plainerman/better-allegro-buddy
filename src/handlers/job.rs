//! `bab job history` and `bab job output`. Port of `handlers/job.py`. Owned by P3-F.

use crate::core::jobs::{OutputStream, get_job_detail, get_job_output, get_user_job_history};
use crate::core::parse::{base_job_id, user_from_userid};
use crate::core::ui::{build_job_history_rows, muted, render_table};
use crate::core::users::current_user;
use crate::core::{DEFAULT_JOB_OUTPUT_TAIL_LINES, RecordExt};
use crate::error::{Result, die};
use crate::pyfmt;
use crate::render::console;
use crate::render::table::Justify;
use crate::render::text::Text;

/// `cmd_job_history(limit=)` (0 or negative: all).
pub fn cmd_job_history(limit: i64) -> Result<()> {
    let jobs = get_user_job_history(None, true, true, "now-30days")?;
    if jobs.is_empty() {
        console::stdout().print(&muted("No finished jobs in your history."));
        return Ok(());
    }
    let shown = match usize::try_from(limit) {
        Ok(limit) if limit > 0 => &jobs[..limit.min(jobs.len())],
        _ => &jobs[..],
    };
    render_table(
        &[
            ("JobID", Justify::Left),
            ("Name", Justify::Left),
            ("State", Justify::Left),
            ("Exit code", Justify::Left),
            ("Ended", Justify::Left),
        ],
        build_job_history_rows(shown),
        Some("Finished jobs"),
        None,
        &[],
        None,
    );
    Ok(())
}

/// `cmd_job_output(jobid=, stderr=)`.
pub fn cmd_job_output(jobid: &str, stderr: bool) -> Result<()> {
    let stream = if stderr {
        OutputStream::Stderr
    } else {
        OutputStream::Stdout
    };

    // Pre-check ownership for a clearer error than get_job_output's "could not resolve".
    // Python: `get_job_detail(jobid) or get_job_detail(base_job_id(jobid))`.
    let detail = match get_job_detail(jobid)? {
        Some(detail) if !detail.is_empty() => Some(detail),
        _ => get_job_detail(base_job_id(jobid))?,
    };
    if let Some(detail) = detail {
        let owner = user_from_userid(detail.get_str("UserId"));
        if !owner.is_empty() && owner != current_user() {
            return Err(die(format!(
                "job {jobid} belongs to {owner}; you can only read your own output"
            )));
        }
    }

    let (output_path, output_text) =
        get_job_output(jobid, stream, DEFAULT_JOB_OUTPUT_TAIL_LINES, None)?;

    let out = console::stdout();
    let mut header = muted(&output_path.display().to_string());
    if pyfmt::splitlines(&output_text).len() >= DEFAULT_JOB_OUTPUT_TAIL_LINES {
        header.append(
            &format!(" · last {DEFAULT_JOB_OUTPUT_TAIL_LINES} lines"),
            None,
        );
    }
    out.print(&header);

    if pyfmt::strip(&output_text).is_empty() {
        out.print(&muted("The file is empty."));
        return Ok(());
    }
    out.print(&Text::new(output_text.trim_end_matches('\n')));
    Ok(())
}
