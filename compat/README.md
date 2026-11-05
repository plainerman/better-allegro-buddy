# compat: differential tests, Python `ab` 2.7 vs Rust `bab`

Black-box harness that runs the Python reference (allegro-buddy 2.7 from PyPI, renamed
to `bab`) and a `bab` binary against the same fake Slurm and requires identical
stdout, stderr (bytes, ANSI included), exit code, Slurm calls and resulting files.

The binary under test must be built with the test hooks (`BAB_TEST_*`, see below), which
a normal build leaves out:

```sh
cargo build --release --features test-hooks
uv run --project compat compat/run.py check --bin target/release/bab      # vs goldens
uv run --project compat compat/run.py check --bin target/release/bab -k 'why|fix'
uv run --project compat compat/run.py diff  --bin target/release/bab      # both live
uv run --project compat compat/run.py golden          # regenerate goldens from ref-bab
uv run --project compat compat/run.py list --category dev
BAB_BIN=target/release/bab uv run --project compat pytest compat -q -n auto   # CI
```

`check`/`diff` print a unified diff per failing scenario (ANSI shown as `\x1b`), a
pass/fail table per category and exit non-zero on any mismatch. Scratch directories
live under `$BAB_COMPAT_WORK` (default `~/.cache/babc`; at most 30 characters, see below).

## Layout

| path | what |
|---|---|
| `pyproject.toml`, `uv.lock` | the pinned oracle: Python 3.12, allegro-buddy 2.7, click 8.3.2, rich 15.0.0, rich-click 1.9.9 (main's versions), pytest |
| `reference.py`, `ref-bab` | builds the renamed copy of the package (`.cache/ref-*`) and runs it with the test hooks |
| `fake_slurm/fake_slurm.py`, `fake_slurm/bin/*` | one standalone script; `squeue scontrol sinfo sacct sacctmgr sbatch scancel srun ssh salloc sstat` are symlinks to it |
| `worlds.py` | fake clusters ported from the Python tests: `plain` (FakeSlurm), `demo` (DemoSlurm), `busy`, `other`, `cpuonly`, `nodefault`, `lise` (bhlrn4), `nojson`, `empty`, `down`, `noname`, `groups`, plus `why_world(...)` (test_why records) |
| `scenarios.py` | the scenario matrix (846 scenarios) |
| `harness.py` | runs one scenario in a sealed scratch dir |
| `run.py`, `test_compat.py` | golden / check / diff / list, and the pytest entry point |
| `golden/<scenario>/` | `stdout`, `stderr`, `exit_code`, `calls.jsonl`, `files.json` + `files/` (captured files and `.tree`), `scenario.json` (for humans) |

## The environment of every run

Scratch dir `T` with a fixed path length of 40 characters (`$BAB_COMPAT_WORK/<random>___`),
so output that wraps around a path wraps identically on every machine; every `T` in
captured bytes becomes `@TMP@___...` (same length). Nothing else is normalised, except
the spinner frames of `Console.status` in the three pty scenarios that wait for a dev
job (`normalize` in the scenario: frame count depends on timing), and the install path
of a Python warning raised inside rich-click: on a line
`<prefix>rich_click/<module>.py:<line>: <Category>Warning: ...` the `<prefix>` (Python:
`<venv>/lib/python3.12/site-packages/`; bab prints none) becomes `@PYLIB@/` on both
sides (stdout and stderr). That is rich-click's `RichClickTheme '<name>' not found.`
`UserWarning` for an unknown `RICH_CLICK_THEME` (`theme-help-*bogus`); bab prints it at
the same points (import, then each help/error formatter after Python's warnings registry
was reset by a subprocess or the lazy `_compat_click` import).

Environment (nothing inherited): `PATH=<fake_slurm/bin>:/usr/bin:/bin`, `HOME=T/home`
(also the cwd), `USER=LOGNAME=alice`, `TZ=UTC`, `LANG=C.UTF-8`,
`XDG_CONFIG_HOME=T/config`, `XDG_CACHE_HOME=T/cache`, `AB_SITE_CONFIG=T/site/config.ini`,
`BAB_TEST_NOW=2026-11-03T10:00:00`, `BAB_TEST_GROUPS=users`, `BAB_TEST_HOSTNAME=login1`,
`FAKE_SLURM_STATE`, `FAKE_SLURM_LOG`, `FAKE_SLURM_PYTHON`; pty scenarios add
`TERM=xterm-256color`. Scenarios override or unset (`None`) any of these and add
`COLUMNS`, `FORCE_COLOR`, `NO_COLOR`, `TERM`, `RICH_CLICK_THEME`, `SHELL`, `_BAB_COMPLETE`...

`tty='pty'`: stdin, stdout and stderr on one pseudo-terminal (echo off, window
25 x `$COLUMNS` or 80, own session); stderr is merged into `stdout`. Input is written up
front; `\x04` is EOF. `sigint_after=3` sends SIGINT to the process (watch scenarios).
`then=[...]` runs further argvs in the same `T` (outputs joined with `=== run N ===`).

## Test hooks the Rust binary must implement

Only compiled in with `--features test-hooks` (a plain `cargo build` never reads these
variables, so every scenario that relies on them fails against such a binary). Unset
means real behaviour. The reference patches exactly these Python functions:

| variable | format | replaces in Python |
|---|---|---|
| `BAB_TEST_NOW` | naive local ISO datetime `YYYY-MM-DDTHH:MM:SS` | `core.datetime.now()` (every "now": ETAs, history "today", reservation windows, pending starts) and `time.strftime` in the `--watch` header (`every 10s · 10:00:00 · Ctrl-C to quit`). Not `time.time()`/`sleep` (polling and timeouts stay real) and not `datetime.fromtimestamp` (Slurm epochs, local time = `TZ`). |
| `BAB_TEST_GROUPS` | comma-separated names; empty string = no groups | `current_groups()` (`os.getgroups()` + `grp.getgrgid`) everywhere (partition AllowGroups/DenyGroups, reservation Groups) |
| `BAB_TEST_INTERACTIVE` | `1` / `0` | `why._interactive()` = `sys.stdin.isatty()`, the "may I ask?" check of `why --fix` without `-y`. The prompt itself (click.confirm) still reads stdin. |
| `BAB_TEST_HOSTNAME` | any string | `socket.gethostname()`: the cluster-name cache file `$XDG_CACHE_HOME/allegro-buddy/cluster-name.<host>` and `node` without an argument (tries `<host>` and `<host>` up to the first dot) |

The current user is `getpass.getuser()`: first non-empty of `LOGNAME`, `USER`, `LNAME`,
`USERNAME`, then the passwd entry; the harness sets `USER` and `LOGNAME`.

## Fake Slurm

State: `$FAKE_SLURM_STATE` (JSON world + mutations, under `flock`, safe for concurrent
callers); jobs are high-level records rendered as `squeue --format` fields,
`scontrol --oneliner --details show job` lines and `scontrol --json show job` records;
nodes as `--json show nodes` or `--oneliner show node`. bab's cheaper forms of the same
queries are answered from the same world: `scontrol --version` (24.11.5; older when the
world's JSON is: no data_parser, 22.05 `state_flags`; never fails, it is local),
`scontrol --oneliner --details show node` (Slurm 24.11 text of the same nodes),
`squeue --json --state=PD [--user=U | --jobs=A,B]` (the same `scontrol --json show job`
records, filtered by job, array or het job id), `squeue -h --user=U --format=%i|%T|%S|%e`
(%i as squeue prints it, JobState/StartTime/EndTime as the detail lines do),
`squeue -h -r --user=U --format=%A|%i|%T|%S|%e` (the same, one line per array task a
record holds, all with its job id as %A; an unknown time is `N/A`) and
`squeue -h --nodelist=NODE --states=R,CG,CF --format=%i` (the jobs whose detail line has
NODE in NodeList and JobState RUNNING/COMPLETING/CONFIGURING, array tasks as
`<ArrayJobId>_<ArrayTaskId>`, which `--details show job ID` also answers; an unknown
node fails); `fail` rules for the query of ab's a form stands in for break that form
too. `sbatch` can add a job that turns RUNNING (or FAILED) after N status polls,
`scancel` removes jobs, `scontrol update` applies a scripted result (`update.then`
replaces records/nodes) or changes `MinMemoryNode` of a demo job. `ssh`/`srun` print
`[fake ssh session]` and exit with the world's code. Every invocation appends `{"argv": [...], "exit": N}` to
`$FAKE_SLURM_LOG`; anything not recognised exactly exits 1 and is logged with
`"unexpected": true` (fail closed; goldens contain none).

Calls are compared as: mutating calls (`sbatch`, `scancel`, `scontrol update`, `ssh`,
`srun`) in exact order; all other (read-only) calls as a multiset, so the Rust binary
may run independent queries concurrently but must make the same set of calls, with two
exceptions for bab's Slurm-side speedups (`run.py`, `_readonly_problems`; off with
`--exact-calls`, which the reference passes):
- cheaper forms: bab may make the forms above, and skip a query of ab's only if the form
  standing in for it succeeded in that scenario (`--details show node` for
  `--json show nodes` / `--oneliner show node`; `squeue --json` for
  `scontrol --json show job [ID]`; `%i|%T|%S|%e` and `-r ... %A|%i|%T|%S|%e` for
  `--format=%F` of the same user and `--details show job [ID]`; `--nodelist=NODE` for
  `--details show job`), and after a successful `--nodelist=NODE` make
  `--details show job ID` (one per job listed there);
- prefetch: an extra read-only query is fine if the reference makes it for the same
  subcommand in some golden (it started early, then an error or the data made it
  unnecessary).
Unexpected calls never pass. In `.tree`, bab's version cache
`cache/allegro-buddy/slurm-version.<host>` is ignored (and the directories holding it, if
the reference has none of them and they hold nothing else).

## Scenario categories (846)

cli 90, completion 78, dev 76, settings 74, errors 67, color 59, help 53, fix 42,
demo 41, cluster 40, why 39, job 33, gpu 30, queue 25, nasty 24, theme 18, width 18,
groups 15, watch 13, node 11 (806 piped, 40 on a pty).

- demo / cluster: every `tests/demo_cluster.py` SCENARIO and `test_cluster_output.py`
  case at 160 and 80 columns (`demo-*-c160` equal the old `tests/snapshots/allegro-*.txt`
  with ab→bab, apart from CliRunner echoing prompt input and the scrubbed paths).
- color / width / theme: `FORCE_COLOR=1` variants of the main views, pty with
  `TERM`/`NO_COLOR`/`COLORTERM`, odd `COLUMNS`, `RICH_CLICK_THEME`, `GITHUB_ACTIONS`.
- help: `-h`/`--help` of every command and subgroup (incl. hidden `gpualloc`, `l`, `q`).
- cli: test_cli.py parsing cases (flag_value options, attached values, `--`, repeats,
  unknown options/commands, choices, FloatRange, int errors, missing arguments).
- why / fix / queue / gpu / job / dev: test_why, test_queue, test_gpu*, test_stats,
  test_dev behaviours end to end (fix accepted/declined/garbage/EOF on pipe and pty,
  `-y`, verification paths, dev start/login/stop incl. `--replace` order, exec of ssh/srun).
- completion: scripts, `$SHELL` detection, `--install` (twice, files captured),
  `_BAB_COMPLETE=*_complete` requests incl. dynamic job ids.
- settings: presets via `AB_CLUSTER`/ClusterName/files, layers, invalid configs, cluster
  name cache cold/warm/unwritable/unreadable.
- watch: one frame, then SIGINT after 3 s (pipe and pty).
- nasty: markup, emoji codes and wide characters in plain-string table cells.

## Reference quirks the goldens pin (Rust must match)

- `--version` prints `bab, version 2.7` (the package metadata version, not 2.7.0).
- Plain `str` table cells are Rich markup: `[bold]x[/bold]` in a job name renders bold,
  `:smile:` becomes an emoji, `\[` escapes. Plain strings printed outside tables
  (`Canceled 70110.`, `Left unchanged.`, the watch header) get Rich's ReprHighlighter
  (numbers bold cyan, the clock in the watch header).
- `why --fix` with EOF at the prompt: the prompt stays unterminated on stdout, `Aborted.`
  goes to stderr, exit 1.
- `completion --prog '---'` uses `_AB_COMPLETE` (fallback in `_completion_var`).
- Pipe mode prints no spinner; `CONSOLE.clear()` is a no-op off a terminal.

Not covered on purpose (the reference crashes with a Python traceback, which cannot be
compared): `q -w nan` / `-w inf` (time.sleep error), a job name containing a bare `[/]`
(Rich MarkupError), `completion zsh --install` with a non-existent `$ZDOTDIR`, an
unwritable `$HOME` for `--install`. `[link=...]` markup is avoided (random ids).

## Deliberate divergences (reference crashes, bab rejects)

- `--watch` / `--poll-interval` values click's `FloatRange` accepts but `time.sleep`
  cannot take (`nan`, `inf`, anything from 2**63 ns ≈ 9223372036.85 s up): Python renders a
  watch frame, or for `dev start` cancels (`--replace`) and submits, and then dies with a
  `ValueError`/`OverflowError` traceback. bab checks the value before it is first needed
  (before any rendering for `--watch`; before cancelling/submitting for `dev start`
  unless `--no-wait`; before waiting for `dev login --wait`) and fails with a click-style
  usage error, exit 2: `Invalid value for '--poll-interval': nan is not in the range
  1<=x<9223372036.854776.` Where the value is never slept on (`--no-wait`, `l --wait` on
  a running job) bab matches Python. Tests: tests/sleep_intervals.rs.

## Caveats

- Permission scenarios (`chmod 000` files, read-only cache dir) assume a non-root user.
- The scratch root must be at most 30 characters; set `BAB_COMPAT_WORK` otherwise.
- Verified in `apptainer exec --cleanenv --containall` with ubuntu-24.04.sif (which has
  no python3: the fake runs with `$FAKE_SLURM_PYTHON`, the venv's interpreter).
- PyPI allegro-buddy 2.7 is byte-identical to `main:src/allegro_buddy` (all 14 files).
