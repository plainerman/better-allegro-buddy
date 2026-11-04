#!/bin/sh
''''exec "${FAKE_SLURM_PYTHON:-python3}" "$0" "$@" #'''
"""A fake Slurm for black-box tests: one script behind every command name.

Symlinks in ``bin/`` (squeue, scontrol, sinfo, sacct, sacctmgr, sbatch, scancel, srun,
ssh) all point here; the command is the name it was called by. It answers from a
"world" in the JSON file $FAKE_SLURM_STATE (written by compat/worlds.py), changes that
state on sbatch / scancel / scontrol update (under an flock, so concurrent callers are
safe), and appends one JSON line per invocation to $FAKE_SLURM_LOG:

    {"argv": [...], "exit": 0}                       (plus "unexpected": true)

Any argv it does not recognise exactly fails closed: exit 1, an error on stderr and
"unexpected": true in the log. Standalone on purpose: no allegro_buddy import.

Ported from tests/test_cluster_output.py (FakeSlurm), tests/demo_cluster.py (DemoSlurm)
and tests/slurm_records.py of allegro-buddy 2.7. Times are relative to the world's
"now" (the scenario's BAB_TEST_NOW).
"""

import fcntl
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any

State = dict[str, Any]


class Unexpected(Exception):
    pass


# ---------------------------------------------------------------------------
# Rendering the world the way Slurm prints it
# ---------------------------------------------------------------------------


def number(value: int | None) -> dict[str, Any]:
    if value is None:
        return {'set': False, 'infinite': False, 'number': 0}
    return {'set': True, 'infinite': False, 'number': value}


def nodes_json(nodes: list[dict[str, str]]) -> str:
    """`scontrol --json show nodes`, from node dicts in the shape core.load_nodes returns."""
    return json.dumps(
        {
            'nodes': [
                {
                    'name': n['NodeName'],
                    'state': n['State'].split('+'),
                    'cpus': int(n['CPUTot']),
                    'alloc_cpus': int(n['CPUAlloc']),
                    'real_memory': int(n['RealMemory']),
                    'alloc_memory': int(n['AllocMem']),
                    'gres': '' if n['Gres'] == '(null)' else n['Gres'],
                    'gres_used': n['GresUsed'],
                    'partitions': n['Partitions'].split(','),
                    'active_features': n['ActiveFeatures'].split(','),
                    'features': n['Features'].split(','),
                    'reservation': n['Reservation'],
                    'specialized_memory': int(n['SpecMem']),
                }
                for n in nodes
            ]
        }
    )


def node_lines(nodes: list[dict[str, str]]) -> str:
    """`scontrol --oneliner show node`."""
    return ''.join(
        f'NodeName={n["NodeName"]} AvailableFeatures={n["Features"]} '
        f'ActiveFeatures={n["ActiveFeatures"]} Gres={n["Gres"]} CPUAlloc={n["CPUAlloc"]} '
        f'CPUTot={n["CPUTot"]} RealMemory={n["RealMemory"]} AllocMem={n["AllocMem"]} '
        f'State={n["State"]} Partitions={n["Partitions"]} GresUsed={n["GresUsed"]}\n'
        for n in nodes
    )


TEXT_BASES = ['DOWN', 'IDLE', 'ALLOCATED', 'ERROR', 'MIXED', 'FUTURE', 'UNKNOWN']
TEXT_FLAGS = [
    'BLOCKED', 'CLOUD', 'COMPLETING', 'DRAIN', 'DYNAMIC_FUTURE', 'DYNAMIC_NORM',
    'INVALID_REG', 'FAIL', 'MAINTENANCE', 'POWER_DOWN', 'POWER_UP', 'POWERED_DOWN',
    'REBOOT_REQUESTED', 'REBOOT_ISSUED', 'RESERVED', 'RESUME', 'NOT_RESPONDING', 'PLANNED',
    'POWERING_UP', 'POWERING_DOWN',
]


def text_state(state: str) -> str:
    """The JSON state list (joined by +) as Slurm 24.11 scontrol prints it."""
    base, *flags = state.split('+')
    if base not in TEXT_BASES or any(f not in TEXT_FLAGS for f in flags):
        return state  # not a state Slurm has; print it as given
    return '+'.join([base, *sorted(flags, key=TEXT_FLAGS.index)])


def node_details_lines(nodes: list[dict[str, str]]) -> str:
    """`scontrol --oneliner --details show node` (Slurm 24.11 layout): the nodes of
    nodes_json, for bab's cheaper query."""
    out = []
    for n in nodes:
        name, cpus = n['NodeName'], n['CPUTot']
        feats = ','.join(f for f in n['Features'].split(',') if f) or '(null)'
        act = ','.join(f for f in n['ActiveFeatures'].split(',') if f) or '(null)'
        gres = '(null)' if n['Gres'] in ('', '(null)') else n['Gres']
        line = (
            f'NodeName={name} Arch=x86_64 CoresPerSocket=1  CPUAlloc={n["CPUAlloc"]} '
            f'CPUEfctv={cpus} CPUTot={cpus} CPULoad=0.00 AvailableFeatures={feats} '
            f'ActiveFeatures={act} Gres={gres} '
        )
        if n['GresUsed']:
            line += f'GresUsed={n["GresUsed"]} '
        line += (
            f'NodeAddr={name} NodeHostName={name} Version=24.11.5 OS=Linux 6.1.0 #1 SMP  '
            f'RealMemory={n["RealMemory"]} AllocMem={n["AllocMem"]} FreeMem=N/A Sockets=1 '
            f'Boards=1 '
        )
        if int(n['SpecMem']):
            line += f'MemSpecLimit={n["SpecMem"]} '
        line += (
            f'State={text_state(n["State"])} ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A '
            'MCS_label=N/A '
        )
        parts = ','.join(p for p in n['Partitions'].split(',') if p)
        if parts:
            line += f'Partitions={parts}  '
        line += (
            'BootTime=None SlurmdStartTime=None LastBusyTime=Unknown ResumeAfterTime=None '
            f'CfgTRES=cpu={cpus} AllocTRES= CurrentWatts=0 AveWatts=0 '
        )
        if n['Reservation']:
            line += f' ReservationName={n["Reservation"]}'
        out.append(line + '\n')
    return ''.join(out)


def prop_line(props: dict[str, str]) -> str:
    return ' '.join(f'{key}={value}' for key, value in props.items())


class Clock:
    def __init__(self, state: State) -> None:
        self.now = datetime.fromisoformat(state['now'])

    def at(self, minutes: int) -> str:
        return (self.now + timedelta(minutes=minutes)).isoformat(timespec='seconds')

    def stamp(self, minutes: int) -> int:
        return int((self.now + timedelta(minutes=minutes)).timestamp())


def slurm_duration(minutes: int) -> str:
    """Minutes as squeue prints them: 1-02:03:00, 2:03:00 or 3:00."""
    days, rest = divmod(minutes * 60, 86400)
    hours, rest = divmod(rest, 3600)
    mins, secs = divmod(rest, 60)
    if days:
        return f'{days}-{hours:02d}:{mins:02d}:{secs:02d}'
    if hours:
        return f'{hours}:{mins:02d}:{secs:02d}'
    return f'{mins}:{secs:02d}'


JOB_DEFAULTS: dict[str, Any] = {
    'gpus': 0,
    'gpu_type': '',
    'node': '',
    'gpu_idx': '',
    'started': 0,
    'reason': 'None',
    'start_estimate': None,
    'stdout': '',
    'stderr': '',
}


def job(raw: dict[str, Any]) -> dict[str, Any]:
    return {**JOB_DEFAULTS, **raw}


def gres(j: dict[str, Any]) -> str:
    if not j['gpus']:
        return ''
    return f'gpu:{j["gpu_type"]}:{j["gpus"]}' if j['gpu_type'] else f'gpu:{j["gpus"]}'


def squeue_fields(j: dict[str, Any], clock: Clock) -> dict[str, str]:
    running = j['state'] == 'RUNNING'
    if running:
        start = clock.at(j['started'])
    elif j['start_estimate'] is not None:
        start = clock.at(j['start_estimate'])
    else:
        start = 'N/A'
    fields = {
        'i': j.get('display_id') or j['id'],
        'A': j['id'],
        'F': j.get('array_job_id') or j['id'],
        'u': j['user'],
        'j': j['name'],
        'T': j['state'],
        'P': j['partition'],
        'M': slurm_duration(-j['started']) if running else '0:00',
        'L': slurm_duration(j['limit'] + j['started'] if running else j['limit']),
        'l': slurm_duration(j['limit']),
        'S': start,
        'R': j['node'] if running else f'({j["reason"]})',
    }
    fields.update(j.get('squeue', {}))
    return fields


def detail_line(j: dict[str, Any], state: State, clock: Clock) -> str:
    """The job as `scontrol --oneliner --details show job` prints it."""
    if j.get('detail_line'):
        return j['detail_line']
    running = j['state'] == 'RUNNING'
    start = clock.at(j['started']) if running else 'Unknown'
    end = clock.at(j['started'] + j['limit']) if running else 'Unknown'
    uid = state.get('uids', {}).get(j['user'], 1000)
    fields = [f'JobId={j["id"]}']
    if j.get('array_job_id'):
        fields += [f'ArrayJobId={j["array_job_id"]}', f'ArrayTaskId={j["array_task_id"]}']
    fields += [
        f'JobName={j["name"]}',
        f'UserId={j["user"]}({uid})',
        f'JobState={j["state"]}',
        f'Reason={j["reason"]}',
        f'SubmitTime={clock.at(min(j["started"], 0) - j.get("submitted_before", 5))}',
        f'StartTime={start}',
        f'EndTime={end}',
        f'Partition={j["partition"]}',
        f'NodeList={j["node"] or "(null)"}',
        f'NumCPUs={j["cpus"]}',
        f'MinMemoryNode={j["mem"] // 1024}G',
    ]
    if j['gpus']:
        fields += [f'TresPerNode=gres/{gres(j)}', f'JOB_GRES={gres(j)}']
    if running:
        node_gres = f' GRES={gres(j)}(IDX:{j["gpu_idx"]})' if j['gpus'] else ''
        fields.append(
            f'Nodes={j["node"]} CPU_IDs=0-{j["cpus"] - 1} Mem={j["mem"]}{node_gres}'
        )
    out = j['stdout'] or f'/home/{j["user"]}/slurm-{j["id"]}.out'
    err = j['stderr'] or out
    fields += [f'StdErr={err}', f'StdOut={out}']
    return ' '.join(fields)


def record(j: dict[str, Any], state: State, clock: Clock) -> dict[str, Any]:
    """The job as `scontrol --json show job` reports it."""
    planned = j['start_estimate'] is not None
    return {
        **state.get('record_base', {}),
        'job_id': int(j['id']),
        'user_name': j['user'],
        'name': j['name'],
        'job_state': [j['state']],
        'state_reason': j['reason'],
        'partition': j['partition'],
        'cpus': number(j['cpus']),
        'tres_per_node': f'gres/{gres(j)}' if j['gpus'] else '',
        'cpus_per_tres': '',
        'memory_per_node': number(j['mem']),
        'tres_req_str': f'cpu={j["cpus"]},mem={j["mem"]}M,node=1,billing={j["cpus"]}',
        'features': '',
        'scheduled_nodes': 'gpu101' if planned else '',
        'start_time': number(clock.stamp(j['start_estimate']) if planned else 0),
        **j.get('record', {}),
    }


# ---------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------

SQUEUE_CODES = set('iAFujTPMLlSR')


def squeue_json(argv: list[str], state: State, clock: Clock) -> tuple[int, str, str]:
    """`squeue --json --state=PD (--user=U | --jobs=A,B)`: those `scontrol --json show job`
    records, in that order."""
    if not state.get('json_ok', True):
        return 1, '', 'squeue: error: No data_parser plugin\n'
    if len(argv) not in (1, 2) or argv[0] != '--state=PD':
        raise Unexpected
    records = state.get('records')
    if records is None:
        records = [record(job(raw), state, clock) for raw in state.get('jobs', [])]

    def pending(r: dict[str, Any]) -> bool:
        st = r.get('job_state') or []
        return 'PENDING' in st

    if len(argv) == 1:
        chosen = [r for r in records if pending(r)]
    elif argv[1].startswith('--user='):
        user = argv[1].split('=', 1)[1]
        chosen = [r for r in records if r.get('user_name') == user and pending(r)]
    elif argv[1].startswith('--jobs='):
        ids = argv[1].split('=', 1)[1].split(',')
        chosen = [
            r
            for r in records
            if pending(r)
            and (
                str(r.get('job_id')) in ids
                or str((r.get('array_job_id') or {}).get('number')) in ids
            )
        ]
    else:
        raise Unexpected
    return 0, json.dumps({'jobs': chosen}), ''


def cmd_squeue(argv: list[str], state: State, clock: Clock) -> tuple[int, str, str]:
    if argv[:1] == ['--json']:
        return squeue_json(argv[1:], state, clock)
    if argv[:1] != ['-h']:
        raise Unexpected
    user = partition = name = fmt = None
    pending_only = False
    rest = argv[1:]
    while rest:
        arg = rest.pop(0)
        if arg.startswith('--user='):
            user = arg.split('=', 1)[1]
        elif arg == '-u' and rest:
            user = rest.pop(0)
        elif arg == '-p' and rest:
            partition = rest.pop(0)
        elif arg == '--state=PD':
            pending_only = True
        elif arg.startswith('--name='):
            name = arg.split('=', 1)[1]
        elif arg.startswith('--format='):
            fmt = arg.split('=', 1)[1]
        else:
            raise Unexpected
    if fmt is None:
        raise Unexpected
    if fmt == '%i|%T|%S|%e':
        # bab's one-call form of `squeue --format=%F` + `scontrol --oneliner --details show
        # job ID` per id (real Slurm prints both from the same job records): %i as this
        # world's squeue prints it, JobState/StartTime/EndTime as its detail_line does
        if user is None or partition or name is not None or pending_only:
            raise Unexpected
        rows = []
        for raw in state.get('jobs', []):
            j = job(raw)
            if j['user'] != user:
                continue
            line = detail_line(j, state, clock)
            props = dict(f.split('=', 1) for f in line.split(' ') if '=' in f)
            rows.append('|'.join([
                squeue_fields(j, clock)['i'], props.get('JobState', ''),
                props.get('StartTime', ''), props.get('EndTime', ''),
            ]) + '\n')
        return 0, ''.join(rows), ''
    codes = [field.lstrip('%') for field in fmt.split('|')]
    if not all(len(code) == 1 and code in SQUEUE_CODES for code in codes):
        raise Unexpected
    rows = []
    for raw in state.get('jobs', []):
        j = job(raw)
        if user and j['user'] != user:
            continue
        if partition and j['partition'] not in partition.split(','):
            continue
        if pending_only and j['state'] != 'PENDING':
            continue
        if name is not None and j['name'] != name:
            continue
        fields = squeue_fields(j, clock)
        rows.append('|'.join(fields[code] for code in codes) + '\n')
    return 0, ''.join(rows), ''


def _matches_id(raw: dict[str, Any], wanted: str) -> bool:
    return wanted in {raw['id'], raw.get('array_job_id'), raw.get('display_id')}


def _advance(raw: dict[str, Any]) -> None:
    """A submitted job that turns RUNNING (or ends) after a number of status polls."""
    becomes = raw.get('becomes')
    if not becomes:
        return
    if becomes.get('after_polls', 0) > 0:
        becomes['after_polls'] -= 1
        return
    raw.pop('becomes')
    raw['state'] = becomes.get('state', 'RUNNING')
    raw['reason'] = becomes.get('reason', 'None')
    if raw['state'] == 'RUNNING':
        raw['node'] = becomes.get('node', 'gpu101')
        raw['gpu_idx'] = becomes.get('gpu_idx', '0')
        raw['started'] = 0


def cmd_scontrol(argv: list[str], state: State, clock: Clock) -> tuple[int, str, str]:
    jobs = state.get('jobs', [])
    known = jobs + state.get('finished', [])
    if argv == ['show', 'config']:
        name = state.get('cluster_name')
        text = 'Configuration data as of 2026-11-03T10:00:00\n'
        text += 'AccountingStorageType   = accounting_storage/slurmdbd\n'
        if name:
            text += f'ClusterName             = {name}\n'
        text += 'SlurmctldPort           = 6817\n'
        return state.get('config_exit', 0), text, ''
    if argv == ['--json', 'show', 'nodes']:
        if not state.get('json_ok', True):
            return 1, '', 'scontrol: error: No data_parser plugin\n'
        return 0, nodes_json(state.get('nodes', [])), ''
    if argv == ['--oneliner', '--details', 'show', 'node']:
        return 0, node_details_lines(state.get('nodes', [])), ''
    if argv == ['--oneliner', 'show', 'node']:
        return 0, node_lines(state.get('nodes', [])), ''
    if argv == ['--oneliner', 'show', 'partition']:
        lines = ''.join(prop_line(p) + '\n' for p in state.get('partitions', []))
        return 0, lines, ''
    if argv == ['--oneliner', 'show', 'reservation']:
        resvs = state.get('reservations', [])
        if not resvs:
            return 0, 'No reservations in the system\n', ''
        return 0, ''.join(line + '\n' for line in resvs), ''
    if argv[:3] == ['--json', 'show', 'job'] and len(argv) <= 4:
        records = state.get('records')
        if records is None:
            records = [record(job(raw), state, clock) for raw in jobs]
        if len(argv) == 4:
            wanted = argv[3]
            if state.get('records_filter', True):
                records = [
                    r
                    for r in records
                    if str(r.get('job_id')) == wanted
                    or str((r.get('array_job_id') or {}).get('number')) == wanted
                ]
            if not records:
                return 1, '', 'slurm_load_jobs error: Invalid job id specified\n'
        return 0, json.dumps({'jobs': records}), ''
    if argv[:4] == ['--oneliner', '--details', 'show', 'job'] and len(argv) <= 5:
        if len(argv) == 4:
            return 0, ''.join(detail_line(job(r), state, clock) + '\n' for r in known), ''
        wanted = argv[4]
        found = [raw for raw in known if _matches_id(raw, wanted)]
        for raw in found:
            _advance(raw)
        if not found:
            return 1, '', 'slurm_load_jobs error: Invalid job id specified\n'
        return 0, ''.join(detail_line(job(r), state, clock) + '\n' for r in found), ''
    if argv[:1] == ['update'] and len(argv) == 3 and argv[1].startswith('JobId='):
        return update(argv[1].split('=', 1)[1], argv[2], state)
    raise Unexpected


def update(job_id: str, field: str, state: State) -> tuple[int, str, str]:
    rule = state.get('update')
    if rule is not None:
        state.update(rule.get('then', {}))
        return rule.get('exit', 0), rule.get('stdout', ''), rule.get('stderr', '')
    key, _, value = field.partition('=')
    for raw in state.get('jobs', []):
        if raw['id'] == job_id and key == 'MinMemoryNode' and value.isdigit():
            raw['mem'] = int(value)
            return 0, '', ''
    raise Unexpected


HISTORY_FORMAT = '--format=JobIDRaw,JobID,JobName,State,ExitCode,End,Reason'
OUTPUT_FORMAT = '--format=User,StdOut,StdErr,JobIDRaw,JobName,JobID'


def cmd_sacct(argv: list[str], state: State) -> tuple[int, str, str]:
    if (
        len(argv) == 6
        and argv[:3] == ['--parsable2', '--noheader', '-X']
        and argv[3].startswith('--user=')
        and argv[4].startswith('--starttime=')
        and argv[5] == HISTORY_FORMAT
    ):
        rule = state.get('sacct_fail')
        if rule:
            return rule.get('exit', 1), rule.get('stdout', ''), rule.get('stderr', '')
        if argv[3] != f'--user={state.get("user", "alice")}':
            return 0, '', ''
        return 0, ''.join(row + '\n' for row in state.get('sacct_rows', [])), ''
    if len(argv) == 6 and argv[:4] == ['-X', '-n', '-P', '-j'] and argv[5] == OUTPUT_FORMAT:
        row = state.get('sacct_jobs', {}).get(argv[4])
        return 0, (row + '\n') if row else '', ''
    raise Unexpected


def cmd_sacctmgr(argv: list[str], state: State) -> tuple[int, str, str]:
    if len(argv) < 6 or argv[:3] != ['-nP', 'show', 'user']:
        raise Unexpected
    if argv[-1] != 'format=DefaultAccount,Account,QOS':
        raise Unexpected
    middle = argv[4:-1]
    if middle not in (['withassoc'], ['withassoc', f'cluster={state.get("cluster_name")}']):
        raise Unexpected
    rule = state.get('sacctmgr_fail')
    if rule:
        return rule.get('exit', 1), rule.get('stdout', ''), rule.get('stderr', '')
    if argv[3] != state.get('user', 'alice'):
        return 0, '', ''
    return 0, state.get('sacctmgr', ''), ''


SBATCH_FLAGS = {
    '--job-name', '--partition', '--time', '--mem', '--output', '--wrap',
    '--cpus-per-gpu', '--gres', '--cpus-per-task', '--account', '--constraint',
    '--exclude', '--nodelist',
}  # fmt: skip


def cmd_sbatch(argv: list[str], state: State) -> tuple[int, str, str]:
    if argv[:1] != ['--parsable']:
        raise Unexpected
    flags: dict[str, str] = {}
    for arg in argv[1:]:
        key, sep, value = arg.partition('=')
        if not sep or key not in SBATCH_FLAGS or key in flags:
            raise Unexpected
        flags[key] = value
    rule = state.setdefault('sbatch', {})
    if rule.get('exit', 0):
        stderr = rule.get('stderr', 'sbatch: error: Batch job submission failed\n')
        return rule['exit'], '', stderr
    job_id = str(rule.get('next_id', '4242'))
    rule['next_id'] = str(int(job_id) + 1)
    if rule.get('add', False):
        gres_flag = flags.get('--gres', '')
        gpus = int(gres_flag.rsplit(':', 1)[1]) if gres_flag else 0
        cpus = int(flags.get('--cpus-per-task', 0)) or int(
            flags.get('--cpus-per-gpu', 1)
        ) * max(gpus, 1)
        mem_flag = flags.get('--mem', '8G')
        mem = int(mem_flag.rstrip('G')) * 1024 if mem_flag.rstrip('G').isdigit() else 8192
        new = {
            'id': job_id,
            'user': state.get('user', 'alice'),
            'name': flags['--job-name'],
            'state': 'PENDING',
            'partition': flags['--partition'],
            'limit': 480,
            'cpus': cpus,
            'mem': mem,
            'gpus': gpus,
            'gpu_type': gres_flag.split(':')[1] if gres_flag.count(':') == 2 else '',
            'reason': 'Priority',
        }
        if 'becomes' in rule:
            new['becomes'] = dict(rule['becomes'])
        state.setdefault('jobs', []).append(new)
    return 0, f'{job_id}{rule.get("suffix", "")}\n', ''


def cmd_scancel(argv: list[str], state: State) -> tuple[int, str, str]:
    if not argv or any(arg.startswith('-') for arg in argv):
        raise Unexpected
    state['jobs'] = [raw for raw in state.get('jobs', []) if raw['id'] not in argv]
    rule = state.get('scancel', {})
    return rule.get('exit', 0), rule.get('stdout', ''), rule.get('stderr', '')


def cmd_exec(name: str, argv: list[str], state: State) -> tuple[int, str, str]:
    """ssh / srun: the process `dev login` replaces itself with."""
    if name == 'ssh' and (argv[:1] != ['-t'] or len(argv) not in (2, 3)):
        raise Unexpected
    if name == 'srun' and (argv[:1] != ['--jobid'] or '--overlap' not in argv):
        raise Unexpected
    rule = state.get(name, {})
    return (
        rule.get('exit', 0),
        rule.get('stdout', f'[fake {name} session]\n'),
        rule.get('stderr', ''),
    )


def slurm_version(state: State) -> str:
    """`scontrol --version`: 24.11 (the layout this fake prints) unless the world's JSON
    says otherwise: no data_parser plugin, or Slurm 22.05's `state_flags`."""
    if not state.get('json_ok', True):
        return 'slurm 20.11.9\n'
    if any('"state_flags"' in r.get('stdout', '') for r in state.get('fail', [])):
        return 'slurm 22.05.11\n'
    return 'slurm 24.11.5\n'


# bab's cheaper queries (src/core/fast.rs on the `rust` branch) -> the queries of ab's
# whose records they print: a `fail` rule for those breaks them the same way. (Rules on
# `scontrol --json show nodes` model broken JSON and so do not reach the text form
# `scontrol --oneliner --details show node`; rules on all of `scontrol` do, by prefix.)
STANDS_IN_FOR = {
    ('squeue', '--json'): ['scontrol', '--json', 'show', 'job'],
    ('squeue', '-h'): ['scontrol', '--oneliner', '--details', 'show', 'job'],
}


def answer(name: str, argv: list[str], state: State) -> tuple[int, str, str]:
    rules = state.get('fail', [])
    if [name, *argv] == ['scontrol', '--version']:  # local: no slurmctld, no fail rule
        return 0, slurm_version(state), ''
    ab_query = STANDS_IN_FOR.get((name, *argv[:1]))
    if ab_query and (argv[:1] == ['--json'] or argv[-1:] == ['--format=%i|%T|%S|%e']):
        for rule in rules:
            if ab_query[: len(rule['argv'])] == rule['argv']:
                return rule.get('exit', 1), rule.get('stdout', ''), rule.get('stderr', '')
    for rule in rules:
        prefix = rule['argv']
        if [name, *argv][: len(prefix)] == prefix:
            return rule.get('exit', 1), rule.get('stdout', ''), rule.get('stderr', '')
    clock = Clock(state)
    if name == 'squeue':
        return cmd_squeue(argv, state, clock)
    if name == 'scontrol':
        return cmd_scontrol(argv, state, clock)
    if name == 'sacct':
        return cmd_sacct(argv, state)
    if name == 'sacctmgr':
        return cmd_sacctmgr(argv, state)
    if name == 'sbatch':
        return cmd_sbatch(argv, state)
    if name == 'scancel':
        return cmd_scancel(argv, state)
    if name in ('ssh', 'srun'):
        return cmd_exec(name, argv, state)
    raise Unexpected


def main() -> int:
    name = os.path.basename(sys.argv[0])
    argv = sys.argv[1:]
    entry: dict[str, Any] = {'argv': [name, *argv]}
    with open(os.environ['FAKE_SLURM_STATE'], 'r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.load(handle)
        before = json.dumps(state, sort_keys=True)
        try:
            code, out, err = answer(name, argv, state)
        except Unexpected:
            code, out, err = 1, '', f'fake_slurm: unexpected call: {[name, *argv]}\n'
            entry['unexpected'] = True
        if json.dumps(state, sort_keys=True) != before:
            handle.seek(0)
            handle.truncate()
            json.dump(state, handle, indent=1)
            handle.flush()
        entry['exit'] = code
        log_path = os.environ.get('FAKE_SLURM_LOG')
        if log_path:
            with open(log_path, 'a') as log:
                log.write(json.dumps(entry) + '\n')
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.stdout.flush()
    sys.stderr.flush()
    return code


if __name__ == '__main__':
    sys.exit(main())
