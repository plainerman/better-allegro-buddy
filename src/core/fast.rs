//! Cheaper Slurm queries that carry the same information as the ones `ab` makes.
//!
//! `scontrol --json` spends most of its time formatting JSON on the client: on Allegro
//! (Slurm 24.11, ~200 jobs, ~90 nodes) `scontrol --json show nodes` takes ~50 ms and
//! `scontrol --json show job` ~130 ms, while `scontrol --oneliner --details show node`
//! takes ~14 ms and `squeue --json` filtered to the jobs needed ~15 ms. bab uses those
//! instead and converts them to exactly what the JSON would have said
//! ([`super::nodes::nodes_from_details_text`]; `squeue --json` prints the same job records
//! as `scontrol --json show job`, in the same order, just fewer of them).
//!
//! The conversion follows the Slurm source of the versions in [`VERIFIED`], so it is only
//! used there; with any other client version, or when a fast query fails or prints
//! something unexpected, bab makes `ab`'s queries and handles them like `ab`.
//!
//! The client version (`scontrol --version`, no RPC) is cached in
//! `~/.cache/allegro-buddy/slurm-version.<host>` together with the path, size and mtime of
//! the `scontrol` on `PATH`, so it is asked again only after Slurm changes.

/// Slurm client versions (major, minor) whose `scontrol` text, JSON and `squeue --json`
/// output the conversion was checked against (24.05: data_parser v0.0.41, 24.11: v0.0.42,
/// 25.05: v0.0.43, 25.11: v0.0.44, 26.05: v0.0.45), each newer one also on a live cluster
/// (25.05.8, 25.11.8, 26.05.2). Node flags the text and JSON show differently make the
/// node text fall back: `EXTERNAL` (text from 25.05, JSON from 25.11), `BLOCKED`, and
/// `POWER_DOWN`+`POWERED_DOWN` (26.05's JSON adds `POWER_DOWN_FORCE`).
const VERIFIED: [(u32, u32); 5] = [(24, 5), (24, 11), (25, 5), (25, 11), (26, 5)];

/// Whether the fast queries may be used (see the module docs).
pub fn enabled() -> bool {
    #[cfg(test)]
    {
        crate::hooks::testing::ctx()
            .slurm_version
            .is_some_and(|v| VERIFIED.contains(&v))
    }
    #[cfg(not(test))]
    {
        static VERSION: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
        VERSION
            .get_or_init(client_version)
            .is_some_and(|v| VERIFIED.contains(&v))
    }
}

/// `24.11` of `slurm 24.11.5` / `slurm-wlm 24.11.5`.
fn parse_version(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.split_whitespace().last()?.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// The `scontrol` that `PATH` resolves to, and what identifies that build of it.
#[cfg(not(test))]
fn locate_scontrol() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join("scontrol");
        if let Ok(meta) = std::fs::metadata(&candidate)
            && meta.is_file()
            && meta.mode() & 0o111 != 0
        {
            return Some(format!(
                "{}\t{}:{}:{}:{}.{}",
                candidate.display(),
                meta.dev(),
                meta.ino(),
                meta.size(),
                meta.mtime(),
                meta.mtime_nsec()
            ));
        }
    }
    None
}

#[cfg(not(test))]
fn client_version() -> Option<(u32, u32)> {
    let key = locate_scontrol()?;
    let cache = crate::settings::bab_cache_path("slurm-version");
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Some((cached_key, version)) = text.split_once('\n')
        && cached_key == key
    {
        return parse_version(version);
    }
    let output = super::slurm::try_run(&["scontrol", "--version"], false).ok()?;
    if !output.success() {
        return None;
    }
    let version = crate::pyfmt::strip(&output.stdout);
    if let Some(parent) = cache.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&cache, format!("{key}\n{version}\n"));
    parse_version(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(parse_version("slurm-wlm 24.11.5\n"), Some((24, 11)));
        assert_eq!(parse_version("slurm 24.05.8"), Some((24, 5)));
        assert_eq!(parse_version("slurm 25.11.0-0rc1"), Some((25, 11)));
        assert_eq!(parse_version(""), None);
    }
}
