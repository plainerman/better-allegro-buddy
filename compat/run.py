#!/usr/bin/env python3
"""Differential harness: the Python reference (`ref-bab`) vs a `bab` binary.

    uv run --project compat compat/run.py golden [-k PATTERN]        # regenerate goldens
    uv run --project compat compat/run.py check --bin PATH [-k PATTERN]
    uv run --project compat compat/run.py diff --bin PATH [--ref compat/ref-bab]
    uv run --project compat compat/run.py list [-k PATTERN]

`check` compares against compat/golden/<scenario>/; `diff` runs both binaries live.
Compared: stdout and stderr (bytes), exit code, the fake Slurm call log (mutating
calls -- sbatch, scancel, scontrol update, ssh, srun -- in exact order, read-only
calls as a multiset) and the files the scenario captures, plus the tree of T.
Exits non-zero on any mismatch.

bab may start read-only queries early and replace them by cheaper ones carrying the
same information, and caches the Slurm client version (`_readonly_problems`,
`_without_version_cache`); `--exact-calls` turns that off (the reference passes with it).
"""

from __future__ import annotations

import argparse
import difflib
import functools
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMPAT = Path(__file__).resolve().parent
sys.path.insert(0, str(COMPAT))

import harness  # noqa: E402
from scenarios import SCENARIOS, Scenario  # noqa: E402

GOLDEN = COMPAT / 'golden'
REFERENCE = COMPAT / 'ref-bab'


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------


def write_golden(scenario: Scenario, result: harness.Result) -> None:
    folder = GOLDEN / scenario.name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / 'stdout').write_bytes(result.stdout)
    (folder / 'stderr').write_bytes(result.stderr)
    (folder / 'exit_code').write_text(result.exit_code + '\n')
    (folder / 'calls.jsonl').write_text(result.calls_jsonl())
    (folder / 'scenario.json').write_text(json.dumps(scenario.describe(), indent=1) + '\n')
    manifest = {}
    for rel, content in result.files.items():
        manifest[rel] = content is not None
        if content is not None:
            path = folder / 'files' / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    (folder / 'files.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def read_golden(scenario: Scenario) -> harness.Result | None:
    folder = GOLDEN / scenario.name
    if not (folder / 'exit_code').exists():
        return None
    manifest = json.loads((folder / 'files.json').read_text())
    files = {
        rel: (folder / 'files' / rel).read_bytes() if present else None
        for rel, present in manifest.items()
    }
    calls = [
        json.loads(line)
        for line in (folder / 'calls.jsonl').read_text().splitlines()
        if line.strip()
    ]
    return harness.Result(
        (folder / 'stdout').read_bytes(),
        (folder / 'stderr').read_bytes(),
        (folder / 'exit_code').read_text().strip(),
        calls,
        files,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _visible(data: bytes) -> list[str]:
    text = data.decode('utf-8', 'backslashreplace')
    text = text.replace('\x1b', '\\x1b').replace('\r', '\\r')
    return text.splitlines(keepends=True)


def _diff(expected: bytes, actual: bytes, label: str) -> str:
    lines = difflib.unified_diff(
        _visible(expected),
        _visible(actual),
        f'expected/{label}',
        f'actual/{label}',
        lineterm='',
    )
    out = ''.join(line if line.endswith('\n') else line + '\n' for line in lines)
    if not out:  # differ only in a missing final newline
        out = f'--- expected/{label}\n+++ actual/{label}\n {expected[-40:]!r}\n!{actual[-40:]!r}\n'
    return out


def _fast_keys(argv: list[str]) -> set[tuple[str, str]]:
    """What a cheaper read-only query of bab's (src/core/fast.rs on the `rust` branch)
    stands in for; empty for any other call."""
    if argv == ['scontrol', '--version']:  # local, no RPC; picks the forms below
        return {('version', '')}
    if argv == ['scontrol', '--oneliner', '--details', 'show', 'node']:
        return {('nodes', '')}
    if argv[:3] == ['squeue', '--json', '--state=PD'] and len(argv) <= 4:
        return {('job-records', '')}
    if (
        len(argv) == 4
        and argv[:2] == ['squeue', '-h']
        and argv[2].startswith('--user=')
        and argv[3] == '--format=%i|%T|%S|%e'
    ):
        return {('job-times', argv[2]), ('job-times', '*')}
    if (
        len(argv) == 5
        and argv[:3] == ['squeue', '-h', '-r']
        and argv[3].startswith('--user=')
        and argv[4] == '--format=%A|%i|%T|%S|%e'
    ):
        return {('job-times', argv[3]), ('job-times', '*')}
    if (
        len(argv) == 5
        and argv[:2] == ['squeue', '-h']
        and argv[2].startswith('--nodelist=')
        and argv[3:] == ['--states=R,CG,CF', '--format=%i']
    ):
        return {('node-jobs', '')}
    return set()


def _replaced_keys(argv: list[str]) -> set[tuple[str, str]]:
    """The `_fast_keys` entries that may each replace this query of `ab`'s."""
    if argv in (
        ['scontrol', '--json', 'show', 'nodes'],
        ['scontrol', '--oneliner', 'show', 'node'],
    ):
        return {('nodes', '')}
    if argv[:4] == ['scontrol', '--json', 'show', 'job'] and len(argv) <= 5:
        return {('job-records', '')}
    if argv == ['scontrol', '--oneliner', '--details', 'show', 'job']:
        return {('job-times', '*'), ('node-jobs', '')}
    if argv[:5] == ['scontrol', '--oneliner', '--details', 'show', 'job'] and len(argv) == 6:
        return {('job-times', '*')}
    if (
        len(argv) == 4
        and argv[:2] == ['squeue', '-h']
        and argv[2].startswith('--user=')
        and argv[3] == '--format=%F'
    ):
        return {('job-times', argv[2])}
    return set()


def _follow_up_key(argv: list[str]) -> tuple[str, str] | None:
    """The `_fast_keys` entry whose success lets bab make this query of ab's vocabulary
    though the reference does not here: `node NODE` asks `--details show job ID` for each
    job `squeue --nodelist=NODE` listed, instead of ab's full `--details show job`."""
    if argv[:5] == ['scontrol', '--oneliner', '--details', 'show', 'job'] and len(argv) == 6:
        return ('node-jobs', '')
    return None


def command_of(scenario: Scenario) -> str:
    """The subcommand a scenario runs (`q`, `node`, ...; '' for none)."""
    return next((arg for arg in scenario.argv if not arg.startswith('-')), '')


@functools.cache
def reference_queries() -> dict[str, frozenset[tuple[str, ...]]]:
    """Per subcommand, every read-only query the reference makes in some golden."""
    found: dict[str, set[tuple[str, ...]]] = {}
    for scenario in SCENARIOS:
        expected = read_golden(scenario)
        for call in expected.calls if expected else []:
            if not harness.is_mutating(call['argv']) and not call.get('unexpected'):
                found.setdefault(command_of(scenario), set()).add(tuple(call['argv']))
    return {command: frozenset(argvs) for command, argvs in found.items()}


def _readonly_problems(
    expected: list[dict], actual: list[dict], exact: bool, command: str | None
) -> str:
    """Read-only calls are compared as a multiset. Unless `exact`, bab may also (AGENTS.md
    on the `rust` branch: read-only queries may start early or be replaced by cheaper
    ones carrying the same information):
      - make its cheaper forms (`_fast_keys`), and skip a query of `ab`'s only when a
        cheaper form standing in for it succeeded in the same scenario (`_replaced_keys`);
      - make the per-job queries following a cheaper form that succeeded
        (`_follow_up_key`: `node NODE`'s `--details show job ID` per job listed);
      - prefetch: make a read-only query that the reference makes for the same
        subcommand in some scenario, though not in this one (an earlier call failed, or
        the data made it unnecessary: no pending job to explain, an unknown node, ...).
    Unexpected calls (the fake fails closed) are never accepted."""
    _, exp_ro = _split_calls(expected)
    _, act_ro = _split_calls(actual)
    missing, extra = exp_ro - act_ro, act_ro - exp_ro
    if not exact:
        made = set().union(*(_fast_keys(c['argv']) for c in actual if c.get('exit') == 0))
        missing = Counter(
            {
                c: n
                for c, n in missing.items()
                if not _replaced_keys(json.loads(c)['argv']) & made
            }
        )
        prefetchable = reference_queries().get(command or '', frozenset())
        extra = Counter(
            {
                c: n
                for c, n in extra.items()
                if json.loads(c).get('unexpected')
                or not (
                    _fast_keys(json.loads(c)['argv'])
                    or tuple(json.loads(c)['argv']) in prefetchable
                    or _follow_up_key(json.loads(c)['argv']) in made
                )
            }
        )
    if not missing and not extra:
        return ''
    text = 'read-only Slurm calls (multiset' + (')' if exact else ', cheaper forms allowed)')
    text += ':\n'
    text += ''.join(f'  missing {n}x {c}\n' for c, n in sorted(missing.items()))
    text += ''.join(f'  extra   {n}x {c}\n' for c, n in sorted(extra.items()))
    return text


def _without_version_cache(tree: bytes, expected: bytes | None) -> bytes:
    """bab's only extra file is its Slurm client version cache,
    <cache>/allegro-buddy/slurm-version.<host>: drop it, and the directories holding it
    when the reference has none of them and nothing else is in them."""
    lines = tree.splitlines(keepends=True)
    pattern = re.compile(rb'f (.*/)?allegro-buddy/slurm-version\.[^/]+\n')
    cache = [line for line in lines if pattern.fullmatch(line)]
    if not cache:
        return tree
    wanted = set((expected or b'').splitlines(keepends=True))
    drop = set(cache)
    parents = {p for line in cache for p in Path(line[2:-1].decode()).parents if str(p) != '.'}
    for parent in sorted(parents, key=lambda p: -len(p.parts)):
        line = f'd {parent}\n'.encode()
        inside = f'{parent}/'.encode()
        if line not in wanted and all(
            other in drop or not other[2:].startswith(inside) for other in lines
        ):
            drop.add(line)
    return b''.join(line for line in lines if line not in drop)


def _split_calls(calls: list[dict]) -> tuple[list[str], Counter]:
    mutating, readonly = [], Counter()
    for call in calls:
        text = json.dumps(call)
        if harness.is_mutating(call['argv']):
            mutating.append(text)
        else:
            readonly[text] += 1
    return mutating, readonly


def compare(
    expected: harness.Result,
    actual: harness.Result,
    exact_calls: bool = False,
    command: str | None = None,
) -> list[str]:
    """Human-readable differences; empty when the runs match."""
    problems = []
    if expected.exit_code != actual.exit_code:
        problems.append(f'exit code: expected {expected.exit_code}, got {actual.exit_code}\n')
    for label in ('stdout', 'stderr'):
        if getattr(expected, label) != getattr(actual, label):
            problems.append(_diff(getattr(expected, label), getattr(actual, label), label))
    exp_mut, _ = _split_calls(expected.calls)
    act_mut, _ = _split_calls(actual.calls)
    if exp_mut != act_mut:
        problems.append(
            'mutating Slurm calls (exact order):\n'
            + _diff(
                ''.join(c + '\n' for c in exp_mut).encode(),
                ''.join(c + '\n' for c in act_mut).encode(),
                'calls.mutating',
            )
        )
    readonly = _readonly_problems(expected.calls, actual.calls, exact_calls, command)
    if readonly:
        problems.append(readonly)
    for rel in sorted(set(expected.files) | set(actual.files)):
        exp, act = expected.files.get(rel), actual.files.get(rel)
        if rel == '.tree' and act is not None and not exact_calls:
            act = _without_version_cache(act, exp)
        if exp == act:
            continue
        if exp is None or act is None:
            state = lambda c: 'missing' if c is None else f'{len(c)} bytes'  # noqa: E731
            problems.append(f'file {rel}: expected {state(exp)}, got {state(act)}\n')
        else:
            problems.append(_diff(exp, act, f'files/{rel}'))
    return problems


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def select(pattern: str | None, category: str | None) -> list[Scenario]:
    chosen = [
        s
        for s in SCENARIOS
        if (pattern is None or re.search(pattern, s.name))
        and (category is None or s.category == category)
    ]
    if not chosen:
        raise SystemExit('no scenario matches')
    return chosen


def _parallel(fn, items, jobs: int):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        yield from pool.map(fn, items)


def cmd_golden(args: argparse.Namespace) -> int:
    chosen = select(args.k, args.category)
    if args.k is None and args.category is None:
        shutil.rmtree(GOLDEN, ignore_errors=True)
    bad = 0
    started = time.monotonic()
    for scenario, result in zip(
        chosen, _parallel(lambda s: harness.run(s, str(REFERENCE)), chosen, args.jobs)
    ):
        write_golden(scenario, result)
        unexpected = [c for c in result.calls if c.get('unexpected')]
        if unexpected or result.exit_code == 'timeout':
            bad += 1
            print(f'WARNING {scenario.name}: exit {result.exit_code}, unexpected calls '
                  f'{[c["argv"] for c in unexpected]}')
    print(f'wrote {len(chosen)} goldens in {time.monotonic() - started:.0f}s ({bad} warnings)')
    return 1 if bad else 0


def _report(results: list[tuple[Scenario, list[str]]], started: float, limit: int) -> int:
    failed = [(s, p) for s, p in results if p]
    for scenario, problems in failed:
        print(f'\n=== FAIL {scenario.name}  (bab {" ".join(scenario.argv)})')
        text = ''.join(problems)
        lines = text.splitlines(keepends=True)
        sys.stdout.write(''.join(lines[:limit]))
        if len(lines) > limit:
            print(f'... {len(lines) - limit} more lines')
    by_category: dict[str, list[int]] = {}
    for scenario, problems in results:
        counts = by_category.setdefault(scenario.category, [0, 0])
        counts[0 if not problems else 1] += 1
    print('\ncategory      pass  fail')
    for category, (ok, bad) in sorted(by_category.items()):
        print(f'{category:<12} {ok:>5} {bad:>5}')
    passed = len(results) - len(failed)
    print(
        f'\n{passed} passed, {len(failed)} failed of {len(results)} scenarios '
        f'in {time.monotonic() - started:.0f}s'
    )
    if failed:
        print('failed: ' + ' '.join(s.name for s, _ in failed))
    return 1 if failed else 0


def cmd_check(args: argparse.Namespace) -> int:
    chosen = select(args.k, args.category)
    started = time.monotonic()

    def one(scenario: Scenario) -> list[str]:
        expected = read_golden(scenario)
        if expected is None:
            return [f'no golden for {scenario.name}; run `run.py golden`\n']
        actual = harness.run(scenario, args.bin)
        return compare(expected, actual, args.exact_calls, command_of(scenario))

    results = list(zip(chosen, _parallel(one, chosen, args.jobs)))
    return _report(results, started, args.max_lines)


def cmd_diff(args: argparse.Namespace) -> int:
    chosen = select(args.k, args.category)
    started = time.monotonic()

    def one(scenario: Scenario) -> list[str]:
        expected = harness.run(scenario, args.ref)
        actual = harness.run(scenario, args.bin)
        return compare(expected, actual, args.exact_calls, command_of(scenario))

    results = list(zip(chosen, _parallel(one, chosen, args.jobs)))
    return _report(results, started, args.max_lines)


def cmd_list(args: argparse.Namespace) -> int:
    chosen = select(args.k, args.category)
    for scenario in chosen:
        print(f'{scenario.category:<11} {scenario.name:<48} bab {" ".join(scenario.argv)}')
    print(f'{len(chosen)} scenarios', Counter(s.category for s in chosen))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('golden', 'check', 'diff', 'list'):
        p = sub.add_parser(name)
        p.add_argument('-k', metavar='PATTERN', help='regex on scenario names')
        p.add_argument('--category')
        p.add_argument('-j', '--jobs', type=int, default=min(16, os.cpu_count() or 4))
        p.add_argument('--max-lines', type=int, default=80, help='diff lines per failure')
        if name in ('check', 'diff'):
            p.add_argument('--bin', required=True, help='the bab binary under test')
            p.add_argument(
                '--exact-calls',
                action='store_true',
                help="read-only calls and the file tree exactly as the reference's "
                "(no cheaper bab query forms, no version cache; for checking ref-bab)",
            )
        if name == 'diff':
            p.add_argument('--ref', default=str(REFERENCE), help='reference binary')
    args = parser.parse_args()
    return {'golden': cmd_golden, 'check': cmd_check, 'diff': cmd_diff, 'list': cmd_list}[
        args.command
    ](args)


if __name__ == '__main__':
    sys.exit(main())
