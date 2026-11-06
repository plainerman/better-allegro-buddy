# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib==3.10.*"]
# ///
"""The README's benchmark plots from the live ab vs bab comparison (bench/live.sh).

    uv run bench/plot.py --run DIR   # summarise a live.sh output dir into bench/results/live.json
    uv run bench/plot.py             # plots from bench/results/live.json

Writes bench/plots/live.png (per command) and bench/plots/hero-{light,dark}.svg (README
header: every read-only command once) and prints the table and summary for
bench/RESULTS.md. Means with one standard deviation; speedup = mean(ab) / mean(bab).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402  (a matplotlib dependency)

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / 'results' / 'live.json'
PLOTS = HERE / 'plots'
AB_COLOR = '#3776ab'  # Python blue
BAB_COLOR = '#ce422b'  # Rust orange
TEXT = '#333333'
GRID = '#d0d0d0'

# live.sh benchmark name -> label; order = plot order (top to bottom)
COMMANDS = {
    'version': '--version', 'help': '-h', 'info': 'info', 'info-cold': 'info (cold cache)',
    'free': 'free', 'free-nodes': 'free --nodes', 'q': 'q', 'q-all': 'q --all', 'why': 'why',
    'why-pending': 'why <pending job>', 'node': 'node gpu062', 'job-history': 'job history',
    'config-show': 'config show',
}


def summarise(run: Path) -> dict:
    """hyperfine exports + env.txt of one live.sh run -> the committed summary."""
    env = dict(line.split('=', 1) for line in (run / 'env.txt').read_text().splitlines())
    commands = {}
    for key, label in COMMANDS.items():
        path = run / f'{key}.json'
        if not path.exists():
            continue
        row = {'label': label}
        for result in json.loads(path.read_text())['results']:
            ms = [m['time_wall_clock']['value'] * 1000 for m in result['measurements']]
            row[result['name']] = {
                'n': len(ms), 'mean': round(statistics.fmean(ms), 3),
                'stddev': round(statistics.stdev(ms), 3), 'min': round(min(ms), 3)}
        commands[key] = row
    return {'env': env, 'commands': commands}


def speedup(row: dict) -> float:
    return row['ab']['mean'] / row['bab']['mean']


def fmt_ms(v: float) -> str:
    return f'{v:.0f}' if v >= 100 else f'{v:.1f}' if v >= 1 else f'{v:.2f}'


def live_plot(summary: dict, out: Path) -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'axes.edgecolor': GRID, 'xtick.color': TEXT,
        'ytick.color': TEXT, 'axes.titleweight': 'bold', 'axes.titlesize': 13,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    env, rows = summary['env'], list(summary['commands'].values())
    labels = [row['label'] for row in rows]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, 0.55 * n + 1.7))
    h = 0.38
    for off, impl, color, name in ((-h / 2, 'ab', AB_COLOR, 'ab: allegro-buddy 2.7 (Python)'),
                                   (h / 2, 'bab', BAB_COLOR, 'bab: the Rust rewrite')):
        means = [row[impl]['mean'] for row in rows]
        errs = [row[impl]['stddev'] for row in rows]
        ax.barh([y + off for y in range(n)], means, height=h, color=color, label=name,
                xerr=errs, error_kw={'ecolor': TEXT, 'elinewidth': 0.8, 'capsize': 2})
        for y, m, e in zip(range(n), means, errs):
            ax.text(m + e, y + off, f' {fmt_ms(m)}', va='center', fontsize=7.5, color=TEXT)
    xmax = max(row['ab']['mean'] + row['ab']['stddev'] for row in rows)
    ax.set_xlim(0, xmax * 1.32)
    for y, row in enumerate(rows):
        ax.text(xmax * 1.13, y, f'{speedup(row):.1f}×', va='center', ha='left',
                fontsize=10, fontweight='bold', color=BAB_COLOR)
    ax.text(xmax * 1.13, -0.85, 'speedup', ha='left', va='center', fontsize=8, color=TEXT)
    ax.set_yticks(range(n), labels, fontfamily='DejaVu Sans Mono', fontsize=9)
    ax.set_ylim(n - 0.5, -1.3)
    ax.set_xlabel('wall time per invocation, ms — mean ± σ')
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_title('Rewrite it in Rust: actually worth it', loc='left', pad=22)
    when = env['date_start'][:16].replace('T', ' ')
    ax.text(0, 1.015, f"live on Allegro (Slurm {env['slurm'].split()[-1]}, "
            f"{env['jobs_total']} jobs, {env['nodes']} nodes), {when} CET, login node; "
            f"{env['runs']} runs + 1 warmup each",
            transform=ax.transAxes, fontsize=8.5, color='#666666')
    ax.legend(loc='upper left', ncol=2, frameon=False, fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    # 64 colours keep it a small file
    Image.open(buf).convert('RGB').quantize(64).save(out, optimize=True)
    print(f'wrote {out}')


def hero_totals(summary: dict) -> dict[str, float]:
    """Seconds to run every distinct read-only command once (sum of means)."""
    rows = [row for key, row in summary['commands'].items() if key != 'info-cold']
    return {impl: sum(row[impl]['mean'] for row in rows) / 1000 for impl in ('bab', 'ab')}


def hero(totals: dict[str, float], out_dir: Path) -> None:
    """Small uv-README-style chart: one bar per tool, light and dark variant."""
    w, h, left, top, row, bar, axis = 496, 76, 60, 2, 26, 14, 20
    plot_w, plot_h = 400, row * len(totals)
    font = ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica', 'Arial',
            'sans-serif']
    css_font = ', '.join(f"'{f}'" if ' ' in f else f for f in font)
    xmax = max(totals.values()) * 1.18
    for variant, text in (('light', '#333333'), ('dark', '#C9D1D9')):
        with plt.rc_context({'svg.fonttype': 'none', 'svg.hashsalt': 'bab', 'font.family': font,
                             'font.size': 12, 'text.color': text, 'xtick.color': text,
                             'ytick.color': text, 'figure.facecolor': 'none',
                             'axes.facecolor': 'none', 'savefig.facecolor': 'none'}):
            fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)
            ax = fig.add_axes((left / w, (h - top - plot_h) / h, plot_w / w, plot_h / h))
            names = list(totals)
            ax.barh(range(len(names)), [totals[n] for n in names], height=bar / row,
                    color=BAB_COLOR)
            # baselines 4.3 px below the bar centre: cap height of 12 px text, centred
            for y, n in enumerate(names):
                weight = 'bold' if n == 'bab' else 'normal'
                ax.annotate(f'{totals[n]:.2f}s', (totals[n], y), xytext=(4, -4.3),
                            textcoords='offset points', va='baseline', ha='left',
                            fontweight=weight)
                ax.annotate(n, (0, y), xytext=(-10, -4.3), textcoords='offset points',
                            va='baseline', ha='right', fontweight=weight)
            ax.set_xlim(0, xmax)
            ax.set_ylim(len(names) - 0.5, -0.5)
            ax.set_xticks(range(int(xmax) + 1), [f'{t}s' for t in range(int(xmax) + 1)])
            ax.set_yticks([])
            ax.tick_params(length=0, pad=axis - 15)
            ax.grid(axis='x', color='#7f7f7f', alpha=0.25, linewidth=1)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            path = out_dir / f'hero-{variant}.svg'
            fig.savefig(path, transparent=True, metadata={
                'Date': None, 'Creator': None, 'Format': None, 'Type': None})
            plt.close(fig)
            svg = path.read_text()
            # 1 pt = 1 px at 72 dpi; px makes GitHub render it at exactly w×h like uv's
            svg = svg.replace(f'width="{w}pt" height="{h}pt"', f'width="{w}" height="{h}"', 1)
            # matplotlib expands sans-serif into its own fallback list; keep uv's stack
            svg = re.sub(r"font-family: [^;]*;", f"font-family: {css_font};", svg)
            path.write_text(svg)
            print(f'wrote {path}')


def report(summary: dict, totals: dict[str, float]) -> None:
    """The table and summary lines of bench/RESULTS.md."""
    print('\n| command | ab mean ± σ (ms) | bab mean ± σ (ms) | speedup |\n|---|---:|---:|---:|')
    for row in summary['commands'].values():
        a, b = row['ab'], row['bab']
        print(f"| `{row['label']}` | {a['mean']:.1f} ± {a['stddev']:.1f} | "
              f"{b['mean']:.2f} ± {b['stddev']:.2f} | **{speedup(row):.1f}×** |")
    sp = [speedup(row) for row in summary['commands'].values()]
    print(f'\nmedian speedup {statistics.median(sp):.1f}× '
          f'(geometric mean {statistics.geometric_mean(sp):.1f}×) over {len(sp)} commands')
    print(f"every read-only command once: ab {totals['ab']:.2f} s, bab {totals['bab']:.2f} s "
          f"({totals['ab'] / totals['bab']:.1f}×)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, help='a live.sh output dir: rewrite the summary first')
    args = ap.parse_args()
    if args.run:
        SUMMARY.parent.mkdir(exist_ok=True)
        SUMMARY.write_text(json.dumps(summarise(args.run), indent=1) + '\n')
        print(f'wrote {SUMMARY}')
    summary = json.loads(SUMMARY.read_text())
    PLOTS.mkdir(exist_ok=True)
    live_plot(summary, PLOTS / 'live.png')
    totals = hero_totals(summary)
    hero(totals, PLOTS)
    report(summary, totals)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
