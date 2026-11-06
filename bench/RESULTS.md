# ab vs bab: performance

Python `allegro-buddy` 2.7 (`ab`) vs its byte-identical Rust rewrite `bab`, every
read-only command. The live numbers, the README plots and this file's live table come from
`bench/live.sh` (run on the cluster), summarised into `bench/results/live.json` and
plotted by `bench/plot.py`; see [Updating the numbers](#updating-the-numbers). The local
replay (anonymised Allegro recordings, replay fake, raw hyperfine exports) is kept at the
git tag
[`equivalence-harness-3`](https://github.com/plainerman/better-allegro-buddy/tree/equivalence-harness-3/bench).

## Summary

- **Live on Allegro** (2026-11-05, 14:13 CET, `bab` 2.7.1, Slurm 24.11.5, 164 jobs,
  93 nodes, 10 runs + 1 warmup each): `bab` is **11.4× faster** (median over 13
  commands; geometric mean 14.3×). Every call saves **130–500 ms**. Slurm-bound
  commands: 6.2× (`node`) to 19.9× (`why`); most are now one Slurm round trip (~15 ms)
  plus a few ms. Pure start-up: `--version` 145 → 1.7 ms. All 12 read-only commands once
  each: `ab` **2.95 s**, `bab` **0.24 s** (the README header chart).
- **Second performance pass** (Slurm side, see [below](#performance-pass-2-the-slurm-side)):
  against the previous `bab` median 3.3× (geometric mean 2.6×), all 12 commands 0.88 →
  0.23 s; `why` 209 → 18 ms, `q --all` 181 → 32 ms, `info` 88 → 19 ms (118 → 19 ms for
  a user with 9 jobs).
- **Third pass** (big clusters and newer Slurm, see [below](#performance-pass-3-big-clusters)):
  on charite, lise and hydra (Slurm 25.05/25.11/26.05, up to ~2200 jobs) `why` 1.4–5.4 s
  → 0.03–0.06 s, `node` up to 4.5 s → ~0.1 s, `q --all` up to 5.6 s → 0.3–0.7 s. On
  Allegro `q --all` pays one more round trip (32 → 48 ms), `node` 52 → 42 ms.
- **Local, instant fake Slurm** (pure tool overhead, 30 runs per command; first pass,
  `ab`'s queries): median **19×**, from 8.7× (`why`) to 154× (`--version`); `ab` needs
  85–222 ms per command, `bab` 0.6–15 ms. Peak RSS `ab` 23–32 MiB, `bab` 3.6–10 MiB.
- **Local, replayed Slurm latencies** (each fake call sleeps as long as it took on
  Allegro; first pass, the recording only has `ab`'s queries): median **2.7×** (1.5×
  `why` … 151× `--version`).

![bab vs ab, live on Allegro](plots/live.png)

## Live on Allegro

Login node (2× Xeon E5-2690 v4), early afternoon (load average 1.5). `ab` = console script of a private venv
(Python 3.13.5, allegro-buddy 2.7, click 8.3.2, rich 15.0.0, rich-click 1.9.9); `bab` =
release build without test hooks. Real clock, groups and host, no user config,
`COLUMNS=160`, output to /dev/null, hyperfine `-N`. Speedup = mean(ab) / mean(bab).

| command | ab mean ± σ (ms) | bab mean ± σ (ms) | speedup |
|---|---:|---:|---:|
| `--version` | 145.3 ± 3.9 | 1.74 ± 0.27 | **83.3×** |
| `-h` | 162.9 ± 2.6 | 3.73 ± 0.66 | **43.7×** |
| `info` | 262.9 ± 7.8 | 19.36 ± 1.50 | **13.6×** |
| `info` (cold cache) | 266.4 ± 8.7 | 32.39 ± 2.40 | **8.2×** |
| `free` | 227.6 ± 8.8 | 18.47 ± 1.83 | **12.3×** |
| `free --nodes` | 260.7 ± 8.6 | 20.14 ± 1.23 | **12.9×** |
| `q` | 146.8 ± 4.6 | 17.24 ± 0.79 | **8.5×** |
| `q --all` | 549.4 ± 14.9 | 48.01 ± 1.87 | **11.4×** |
| `why` | 367.6 ± 9.2 | 18.43 ± 2.39 | **19.9×** |
| `why <pending job>` | 211.9 ± 6.3 | 19.88 ± 2.10 | **10.7×** |
| `node gpu062` | 260.4 ± 15.7 | 41.70 ± 2.63 | **6.2×** |
| `job history` | 144.9 ± 2.0 | 12.80 ± 0.93 | **11.3×** |
| `config show` | 206.5 ± 6.1 | 18.45 ± 1.07 | **11.2×** |

## Local replay

First performance pass, before the Slurm-side changes: the recording only has `ab`'s
queries, which is what `bab` still makes on Slurm versions other than 24.05/24.11.
Workstation (Xeon W-1250P, `powersave` governor, both tools pinned to CPUs 4–7), the
anonymised Allegro snapshot (225 jobs, `scontrol --json show job` = 2.4 MB), served by a
`sh` + `cat` fake Slurm (~1 ms per call for both tools). 5 interleaved rounds × 6 runs.

| command | ab, no delay (ms) | bab, no delay (ms) | speedup | ab, realtime (ms) | bab, realtime (ms) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `--version` | 91.9 | 0.60 | 154× | 91.1 | 0.60 | 151× |
| `-h` | 102.6 | 1.60 | 64× | 102.2 | 1.61 | 64× |
| `info` | 100.4 | 7.40 | 13.6× | 287.3 | 134.5 | 2.1× |
| `free` | 97.2 | 5.07 | 19.2× | 217.8 | 90.7 | 2.4× |
| `q` | 84.6 | 1.12 | 75× | 112.7 | 26.0 | 4.3× |
| `q --all` | 221.9 | 15.4 | 14.4× | 524.0 | 195.9 | 2.7× |
| `why` | 113.2 | 13.0 | 8.7× | 378.2 | 249.6 | 1.5× |
| `node gpu062` | 113.8 | 11.6 | 9.8× | 242.7 | 131.4 | 1.8× |
| `job history` | 85.0 | 1.13 | 75× | 107.1 | 21.4 | 5.0× |
| `config show` | 96.6 | 4.35 | 22× | 207.1 | 77.9 | 2.7× |

## Where the speedup comes from, and where it stops

1. **No interpreter start-up and imports.** `ab --version` is 91 ms locally (`import
   allegro_buddy.cli` alone 79 ms: click, rich, rich-click), ~145 ms on Allegro's login
   node; `bab --version` 0.6 ms.
2. **Parsing and rendering.** `q --all` (2.4 MB of job JSON, a 225-row table): `ab`
   ~130 ms of own work, `bab` 15 ms. `bab` decodes only the 28 job keys it reads.
3. **Cheaper Slurm queries with the same answer.** `scontrol --json` spends most of its
   time writing JSON on the client: live, `show job` takes ~130 ms and `show nodes`
   ~50 ms, the text `scontrol --oneliner --details show node` ~14 ms and a filtered
   `squeue --json --state=PD` ~15 ms. See [pass 2](#performance-pass-2-the-slurm-side).
4. **All read-only queries of a command at once**, evaluated in `ab`'s order, so errors
   and output stay `ab`'s even when an earlier call fails.
5. **The floor is Slurm.** Most commands are now one round trip (~15 ms) plus 2–5 ms of
   `bab`. `node` lists the node's jobs, then asks for each one's details (~40 ms);
   `q --all` needs the queue listing before it knows which jobs get a hint, then asks
   for those (~48 ms). Under heavier load the absolute savings stay, the ratios of
   Slurm-bound commands shrink.

## Performance pass

Measured with callgrind (instructions) and pinned hyperfine on the replay; output and
Slurm calls verified unchanged after every change (compat 846/846, replay 532/532).

| change | effect |
|---|---|
| ASCII twins of the `\w`/`\d` highlight regexes (Unicode originals only for non-ASCII text) | `bab -h`: 31.3M → 4.3M instructions, 7.8 → 2.2 ms; every help and usage-error panel likewise |
| Parse only the 28 job-record keys bab reads (a serde visitor; odd shapes fall back to the full parse) | `why` 26.5 → 14.2 ms, `q --all` 30.7 → 15.4 ms of own work |
| `free` runs the reservation and partition queries side by side | `free` realtime 110 → 95 ms |

Tried and rejected: more concurrent Slurm calls (would change the calls made when an
earlier one fails; done in pass 2 once only output had to match), mimalloc (+0.6 ms
start-up on every command), smaller wins under 2 % (pre-sized records, a reduced
node-JSON parse), `-C target-cpu=native` (not portable).

## Performance pass 2: the Slurm side

`bab`'s own work was 1–30 ms per command; the rest was waiting for Slurm. Measured live
on Allegro with hyperfine (10 runs, mean); "before" is the previous `bab`, both columns
from one back-to-back run. Output verified identical after every change: compat harness
846/846 (its fake taught the new query forms, answered from the same world), replay
parity 532/532 (that recording only has `ab`'s queries, so it checks the fallback path),
and live `ab` vs old vs new `bab` byte-identical on 17 read-only commands and, for
`info`, as each of the 8 users with jobs.

| change | effect (live, each row on top of the ones above) |
|---|---|
| Start all read-only queries of a command at once (`core::slurm::prefetch`), consume them in `ab`'s order | `info` 89 → 54 ms, `why` 210 → 151 ms, `node` 110 → 67 ms (alone) |
| Nodes from `scontrol --oneliner --details show node`, converted to what `--json show nodes` says (flag order, `(null)`, lists); Slurm 24.05/24.11 only | `free` 56 → 21 ms, `config show` 53 → 18 ms, `node` 67 → 47 ms; all 93 Allegro nodes identical to the JSON conversion |
| `why`: `squeue --json --state=PD --user=U` instead of every job's JSON; `q` hints: `squeue --json --state=PD [--user=U]` (same records, same order) | `why` 151 → 18 ms (210 before the pass), `q --all` 171 → 31 ms |
| `info`: your jobs' ETAs from one `squeue --format=%i\|%T\|%S\|%e` instead of `squeue` + one `scontrol` per job | `info` as a user with 9 / 54 jobs: 118 / 141 → 19 / 20 ms (1 + 9 calls → 1) |

The cheaper forms follow Slurm's source; `bab` uses them only with a client whose
version was checked (24.05/24.11 then; 25.05/25.11/26.05 since
[pass 3](#performance-pass-3-big-clusters)) (`scontrol --version`, cached in
`~/.cache/allegro-buddy/slurm-version.<host>` with the binary's size and mtime) and goes back to `ab`'s queries on any failure or unexpected
answer (unknown or `BLOCKED` node flags, `Comment=`/`Extra=` on a node, a
heterogeneous job, a hint job missing from the answer). Extra read-only calls are
cheap: at most a query an earlier failure leaves unused, and `scontrol --version` once
per Slurm install.

Tried and rejected: `node` from `squeue -w NODE` plus one `scontrol show job` per job
(saves ~10 ms on Allegro but more calls; adopted in pass 3, where it saves seconds),
node GRES/CPU blocks from `squeue --json` (lossy conversion), speculative hint queries
for `bab q` of your own jobs (three calls for the case without pending jobs), local CPU
work (now 2–14 ms per command; `-h`'s
remaining regex compile ~0.3 ms, `node`'s text parse ~10 ms: small gain for the code).

## Performance pass 3: big clusters

Allegro (~170 jobs) hid what other clusters showed: Slurm formats every job on the
client, and resolves its user and group, whenever a query asks for all of them. Measured
live with the same `bab` with and without the cheaper queries (the second mode is
`ab`'s queries), every read-only command byte-identical between the two, on charite
(Slurm 25.05.8, ~2200 jobs, sssd/AD), lise (25.11.8, 1152 nodes, ~1250 jobs) and hydra
(26.05.2, ~490 jobs, LDAP without a cache). Harness: tag `equivalence-harness-3`.

| change | effect (live) |
|---|---|
| Your groups looked up only when a partition or reservation names one (sssd/AD returns every member of every group) | charite: no group lookups in `free`/`info` (were 244 `getgrgid_r`, ~20 s with a cold sssd cache) |
| The cheaper queries also for Slurm 25.05, 25.11 and 26.05 (checked against live snapshots and Slurm's source; 26.05's `POWER_DOWN_FORCE` falls back) | `why` charite 1.7 s → 0.03 s, lise 1.4 s → 0.05 s, hydra 5.4 s → 0.06 s; `info`/`free` 2–5× |
| `q --all` hints: `squeue --json --state=PD --jobs=<hinted ids>` after the listing instead of every pending job beside it | charite 1.8 s → 0.3 s, hydra 5.6 s → 0.7 s; Allegro 32 → 48 ms (one more round trip in sequence) |
| `node`: `squeue -w NODE` and one `scontrol show job ID` per job (≤ 20, in parallel) instead of every job's details | charite 0.95 s → 0.04–0.16 s, hydra 4.5 s → 0.1 s, lise 0.26 s → 0.05–0.09 s; Allegro 52 → 42 ms |
| `info` ETAs: `squeue -r … %A\|%i\|%T\|%S\|%e`, one row per job record (a fix: plain squeue merged pending array tasks, so `info` could count fewer than `ab`) | same speed |

Not fixable in `bab`: on charite about one DNS lookup of the controller in 32 gets no
answer and waits the resolver's 5 s timeout, for every Slurm command (`ab`'s too).

## Methodology notes

- Same output: the measured builds pass replay parity 532/532 (every recorded command ×
  `COLUMNS` 80/160/200 × plain/`FORCE_COLOR`; stdout, stderr, exit code; Slurm calls
  equal up to the read-only changes of pass 2).
- hyperfine `-N` runs `venv/bin/ab` and `bab` directly (no shell, no `uv run`). The
  replay freezes clock, groups and host name for both tools (`BAB_TEST_*` hooks for
  `bab`, an equivalent patch for `ab`); measured cost within noise. Live runs use no hooks.
- Peak RSS live includes the Slurm clients (`scontrol --json` ~50 MiB for both tools);
  the local numbers are the tools' own.
- Python differs between hosts (3.12.11 locally, 3.13.5 on Allegro); comparisons are
  always within one host.

## Updating the numbers

Live, on the cluster (≈ 2 minutes, read-only commands only):

```sh
# once: the reference in a private venv (never /usr/bin/ab, that is Apache Bench)
B=.local/share/bab-bench
ssh allegro "python3 -m venv $B/venv && $B/venv/bin/pip install allegro-buddy==2.7 \
    click==8.3.2 rich==15.0.0 rich-click==1.9.9 markdown-it-py==4.0.0 pygments==2.20.0 \
    && mkdir -p $B/bin"
scp "$(command -v hyperfine)" allegro:$B/bin/

# each time: a release build WITHOUT test hooks (glibc no newer than the cluster's,
# e.g. built on the cluster or a release binary), then the run
cargo build --release
scp target/release/bab bench/live.sh allegro:$B/bin/
ssh allegro "cd $B && timeout 900 sh bin/live.sh out/live-\$(date +%Y%m%dT%H%M%S)"
mkdir -p bench/.work && scp -r allegro:$B/out/live-<stamp> bench/.work/
uv run bench/plot.py --run bench/.work/live-<stamp>
```

`plot.py --run` rewrites `bench/results/live.json` (per command and tool: n, mean, σ, min
in ms, plus the run's `env.txt`), redraws `bench/plots/live.png` and
`bench/plots/hero-{light,dark}.svg`, and prints the live table and the summary numbers
(median speedup, total of all read-only commands) for this file and the README.
`uv run bench/plot.py` alone redraws the plots from the committed summary. Commit
`live.json` and the plots, not the raw run (`bench/.work/` is ignored); update the numbers
quoted above and in the README (header chart alt text, intro, the `-h` time under
Commands, Performance). The local replay numbers do not come from this run (see the
`equivalence-harness-3` tag).
