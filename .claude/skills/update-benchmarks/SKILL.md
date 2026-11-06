---
name: update-benchmarks
description: Re-run the live ab vs bab benchmark on the Allegro cluster after performance work, regenerate bench/results/live.json and the plots, and update every number quoted in README.md and bench/RESULTS.md. Use when asked to update, refresh or rerun the benchmarks, plots or performance numbers.
---

# Updating the benchmark numbers

The procedure (one-time cluster setup, build, run, fetch, plot) is in
`bench/RESULTS.md`, section "Updating the numbers". Follow it; this skill adds the
checks around it and the list of places to update.

## Before the run

1. Only benchmark a build that behaves identically: `cargo test --locked` passes and,
   after changes to anything but docs, the `verify-equivalence` skill passes too.
2. Build without test hooks: `cargo build --release --locked`. The binary is in
   `$CARGO_TARGET_DIR/release/bab` if that is set, else `target/release/bab`. Its glibc
   must not be newer than the cluster's: compare `ldd --version` locally and via
   `ssh allegro`, else build on the cluster or use a release binary.
3. Print the current numbers so you can find them later:
   `uv run bench/plot.py` (redraws the plots from the committed `live.json`
   byte-identically and prints the table and summary lines).

## The run

- Cluster rules from `AGENTS.md` apply: `bench/live.sh` only runs read-only commands
  and the venv's `allegro-buddy`; don't add commands to it that change Slurm state.
- One run at a time, at a quiet time; note the time (it goes into the text).
- `uv run bench/plot.py --run bench/.work/live-<stamp>` rewrites
  `bench/results/live.json`, `bench/plots/live.png`, `bench/plots/hero-{light,dark}.svg`
  and prints the new table and summary lines. Machine details are in the run's
  `env.txt` (also stored in `live.json` under `env`).

## Where the numbers appear

Update each with the new values, keeping the current wording and rounding:

- `bench/RESULTS.md`
  - Summary, "Live on Allegro" bullet: date/time, Slurm version, jobs, nodes, runs,
    median and geometric mean, saved ms range, Slurm-bound range, `--version`, the
    "all read-only commands once" totals; drop or redo the "second pass" sentence.
  - "Live on Allegro" table: paste the table printed by `plot.py`.
  - "Where the speedup comes from": the live ms figures and the `why` speedup.
  - "Performance pass": add a row per new optimisation (what changed, measured effect).
- `README.md`
  - header chart `<img alt=...>`: command count and totals.
  - intro: "**N× faster**" (median, live).
  - Commands: the `-h` time ("comes back in ~N ms").
  - Performance: the setup line (jobs, nodes, runs), the short table (a subset of the
    RESULTS rows), the median/geometric mean/saved-ms line, and the `why` figures and
    speedup under "Where it stops".
- Local replay numbers (README "instant fake Slurm" sentence, RESULTS "Local replay"
  table and bullets) are not produced by `live.sh`. Rerun them with the tooling at the
  `equivalence-harness-3` tag (`bench/run_local.sh`, `bench/README.md` there) or leave
  them as they are; never mix numbers from different builds in one table.

## Verify

- `git grep -nF '<old value>' -- README.md bench/RESULTS.md` for each old figure from
  step 3 finds nothing that should have changed.
- `uv run bench/plot.py` leaves `git status` clean (plots match `live.json`).
- Commit `bench/results/live.json`, `bench/plots/*` and the docs, not `bench/.work/`.
