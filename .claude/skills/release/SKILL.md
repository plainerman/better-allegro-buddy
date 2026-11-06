---
name: release
description: Cut a release of bab (better-allegro-buddy): bump the version, tag vX.Y.Z, and follow the release workflow that builds the binaries and wheels, creates the GitHub release and optionally publishes to PyPI. Use when asked to release, publish, bump the version or check on a release.
---

# Releasing bab

The authoritative steps are in `README.md`, section "Releasing"; the workflow is
`.github/workflows/release.yml` (its header comment says what runs when), packaging is
`pyproject.toml` (maturin, binary-only wheels). Read them first; this is the checklist.

1. Ask before tagging or pushing to `main`: a tag publishes a GitHub release and,
   with PyPI enabled, a PyPI upload that can never be replaced.
2. Bump `version` in `Cargo.toml`, run `cargo check` (updates `Cargo.lock`). Whether
   `VERSION_DISPLAY` in `src/lib.rs` (what `bab --version` prints, currently the
   reference's string) changes too is the maintainer's call: it changes output that
   the harness compares, so say so.
3. Run the checks from `AGENTS.md` (fmt, both clippy runs, `cargo test --locked`).
   Optional local packaging check: `uvx maturin build --release`, `uvx maturin sdist`.
4. Merge into `main`; pushes there run CI and a release dry run (build and smoke-test,
   no release).
5. `git tag vX.Y.Z && git push origin vX.Y.Z` on that `main` commit. The tag must equal
   the `Cargo.toml` version (the workflow checks).

## Following and checking it

- `gh run list --workflow release.yml` / `gh run watch <id>`.
- `gh release view vX.Y.Z`: the `bab-<arch>-<OS>.tar.gz` files, wheels, sdist,
  `SHA256SUMS`. The workflow already installs the release with the README one-liner on
  Linux and macOS; locally, run that one-liner from the README "Install" section.
- PyPI uploads happen only if the repository variable `PYPI_PUBLISH` is `true`
  (`gh variable list`) and trusted publishing is set up (README, one-time setup).
  Publishing an already released tag later: re-run that tag's workflow run.
