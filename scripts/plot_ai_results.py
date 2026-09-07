#!/usr/bin/env python3
"""Render recorded AI evidence as PNG/SVG figures without training or evaluating."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator, PercentFormatter, FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
BG = '#f4f6fa'
INK = '#18253e'
MUTED = '#53627a'
GRID = '#e1e6ef'
TEAL = '#007e68'
BLUE = '#315fcd'
PURPLE = '#7650bb'
RED = '#ad344c'
SOURCES = [
    'ai/runs/hardened_pretrain/training.jsonl',
    'ai/runs/hardened_sft/training.jsonl',
    'ai/runs/hardened_dpo/training.jsonl',
    'reports/ai/framework-tests.json',
    'reports/ai/application-tests.txt',
    'reports/ai/environment-validation.json',
    'reports/ai/cpp-quality.json',
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(root):
    training = {}
    for phase in ('pretrain', 'sft', 'dpo'):
        path = root / f'ai/runs/hardened_{phase}/training.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or any(not math.isfinite(row['loss']) for row in rows):
            raise ValueError(f'{phase}: missing or nonfinite losses')
        training[phase] = rows
    framework = json.loads((root / SOURCES[3]).read_text())
    application_text = (root / SOURCES[4]).read_text()
    result = re.search(r'Ran (\d+) tests in ([\d.]+)s\s+OK\s*$', application_text)
    if result is None:
        raise ValueError('The application log must end with an unqualified unittest OK')
    application = {'tests_run': int(result[1]), 'seconds': float(result[2])}
    environment = json.loads((root / SOURCES[5]).read_text())
    quality = json.loads((root / SOURCES[6]).read_text())
    if framework['failures'] or framework['errors'] or framework['skipped']:
        raise ValueError('Framework results changed; update the pass-only figure layout')
    if not framework['passed'] or environment['mechanics_status'] != 'passed':
        raise ValueError('Execution status changed; update the pass-only figure layout')
    if not all(item['passed'] for item in environment['mechanics_checks']):
        raise ValueError('At least one environment check failed')
    if quality['gate']['status'] != 'fail':
        raise ValueError('Model quality changed; update the failed-gate figure narrative')
    if environment['summary']['neural_compile_status'] != 'compile_error':
        raise ValueError('Neural compilation changed; update the observed-output narrative')
    if quality['reproducibility']['checkpoint_payload_sha256'] != environment['model']['checkpoint_payload_sha256']:
        raise ValueError('The environment and quality reports use different checkpoints')
    return training, framework, application, environment, quality


def canvas(title, subtitle):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.text(.055, .955, 'QUESTLINE  /  RECORDED AI EVIDENCE', color=MUTED,
             fontsize=11, weight='bold', va='top')
    fig.text(.055, .905, title, color=INK, fontsize=28, weight='bold', va='top')
    fig.text(.055, .842, subtitle, color=MUTED, fontsize=13, va='top')
    return fig


def card(fig, x, y, width, height, color='white'):
    patch = FancyBboxPatch((x, y), width, height, transform=fig.transFigure,
                          boxstyle='round,pad=0.008,rounding_size=0.014',
                          linewidth=0, facecolor=color, zorder=-1)
    fig.add_artist(patch)


def style_axes(ax):
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=12, length=0, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=.8)


def save(fig, output, name):
    outputs = []
    for extension in ('png', 'svg'):
        path = output / f'{name}.{extension}'
        metadata = {'Software': 'Questline evidence renderer'} if extension == 'png' else {
            'Date': None, 'Creator': 'Questline evidence renderer',
        }
        fig.savefig(path, dpi=160, facecolor=BG, metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def training_figure(training, environment, output):
    count = environment['model']['parameter_count']
    updates = [rows[-1]['updates'] for rows in training.values()]
    fig = canvas('Training loss, exactly as recorded',
                 f'{count:,} parameters  |  {sum(updates)} optimizer updates across three phases  |  No validation-loss series recorded')
    labels = [
        ('pretrain', '01  Pretraining', 'Next-token cross-entropy', TEAL),
        ('sft', '02  Supervised fine-tuning', 'Response-token cross-entropy', BLUE),
        ('dpo', '03  Preference alignment', 'Offline DPO objective', PURPLE),
    ]
    for index, (phase, title, objective, color) in enumerate(labels):
        left = .055 + index * .31
        card(fig, left, .262, .288, .51)
        fig.text(left + .018, .738, title, color=color, fontsize=16, weight='bold')
        fig.text(left + .018, .704, objective, color=MUTED, fontsize=11.5)
        ax = fig.add_axes([left + .054, .407, .211, .234])
        style_axes(ax)
        rows = training[phase]
        x = [row['updates'] for row in rows]
        y = [row['loss'] for row in rows]
        ax.plot(x, y, color=color, linewidth=2.5, marker='o', markersize=7,
                markeredgecolor='white', markeredgewidth=1.5)
        extent = max(y) - min(y)
        margin = max(extent * .38, .005)
        ax.set_ylim(min(y) - margin, max(y) + margin)
        ax.set_xlim(min(x) - .23, max(x) + .23)
        ax.set_xticks(x)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlabel('Optimizer update within phase', color=MUTED, fontsize=11, labelpad=11)
        for step, loss in zip(x, y):
            ax.annotate(f'{loss:.4f}', (step, loss), xytext=(0, 12),
                        textcoords='offset points', ha='center', color=INK, fontsize=12,
                        weight='bold')
        fig.text(left + .018, .286, f'{len(rows)} logged points  /  {rows[-1]["skipped_updates"]} skipped updates',
                 color=MUTED, fontsize=11)
    card(fig, .055, .105, .908, .106, '#e8eef8')
    fig.text(.073, .174, 'Read the scales separately.', color=INK, fontsize=13, weight='bold')
    fig.text(.073, .138, 'Different objectives and minibatches; three observations per phase do not establish convergence or generalization.',
             color=MUTED, fontsize=11.5)
    fig.text(.055, .048, 'Source: ai/runs/hardened_{pretrain,sft,dpo}/training.jsonl  |  Raw points and straight joins; no smoothing.',
             fontsize=10.5, color=MUTED)
    return save(fig, output, 'training-loss')


def execution_figure(framework, application, environment, output):
    when = datetime.fromisoformat(environment['started_at_utc']).strftime('%d %b %Y').lstrip('0')
    machine = environment['environment']
    fig = canvas('The software runs. The neural output fails.',
                 f'Recorded {when}  |  {machine["cpu_model"]}, {machine["physical_memory_bytes"] / 2**30:.0f} GiB RAM  |  Real NumPy + HTTP + restricted C++ compiler')
    card(fig, .055, .27, .538, .498)
    fig.text(.076, .726, 'Execution checks passed', color=INK, fontsize=17, weight='bold')
    fig.text(.076, .687, 'Counts of checks; these are not model-accuracy scores.', color=MUTED, fontsize=11.5)
    ax = fig.add_axes([.215, .415, .332, .218])
    style_axes(ax)
    ax.grid(False)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    counts = [framework['tests_run'], application['tests_run'], len(environment['mechanics_checks'])]
    ax.barh([2, 1, 0], counts, color=[TEAL, BLUE, PURPLE], height=.47)
    ax.set_yticks([2, 1, 0], ['Framework tests', 'Application tests', 'Environment checks'])
    ax.set_xlim(0, max(counts) * 1.25)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.set_xlabel('Passed checks', fontsize=11.5, color=MUTED, labelpad=11)
    ax.spines[['left', 'bottom']].set_visible(False)
    for row, count in zip([2, 1, 0], counts):
        ax.text(count + max(counts) * .03, row, f'{count}/{count}', ha='left', va='center',
                fontsize=13, weight='bold', color=INK)
    fig.text(.076, .295, 'Framework and application suites: zero skips.', color=MUTED, fontsize=11.5)
    summary = environment['summary']
    card(fig, .618, .514, .345, .254, '#fbecef')
    fig.text(.64, .728, 'ACTUAL NEURAL PROGRAM', color=RED, fontsize=11, weight='bold')
    fig.text(.64, .674, 'Compilation failed', color=RED, fontsize=21, weight='bold')
    fig.text(.64, .628, environment['neural_outputs'][0]['generated_text'],
             color=INK, fontsize=17, family='DejaVu Sans Mono')
    fig.text(.64, .587, 'One generated output submitted unchanged.\nThe compiler rejected it before tests ran.',
             color=MUTED, fontsize=11.5, linespacing=1.5, va='top')
    card(fig, .618, .27, .345, .212, '#e2f1ed')
    fig.text(.64, .44, 'SEPARATE CURATED REFERENCE', color=TEAL, fontsize=11, weight='bold')
    fig.text(.64, .384, f'{summary["reference_passed"]}/{summary["reference_total"]} C++ cases passed',
             color=TEAL, fontsize=21, weight='bold')
    fig.text(.64, .345, 'Authored curriculum solution, explicitly requested.\nNot generated by the neural model.',
             color=MUTED, fontsize=11.5, linespacing=1.5, va='top')
    card(fig, .055, .105, .908, .106, '#e8eef8')
    fig.text(.073, .174, 'Execution correctness and teaching competence are different results.',
             color=INK, fontsize=13, weight='bold')
    fig.text(.073, .138, 'The model remains experimental. A passing API response or curated program does not validate its C++ knowledge.',
             color=MUTED, fontsize=11.5)
    fig.text(.055, .048, 'Sources: reports/ai/framework-tests.json, application-tests.txt, environment-validation.json  |  No Docker.',
             fontsize=10.5, color=MUTED)
    return save(fig, output, 'environment-tests')


def quality_figure(quality, output):
    overall = quality['overall']
    count = overall['examples']
    gate = quality['gate']['criteria']
    threshold = gate['minimum_preference_accuracy'] * 100
    lower_required = gate['minimum_bootstrap_lower_bound'] * 100
    fig = canvas('C++ quality screen: FAILED',
                 f'{count} original paired questions  |  Final DPO checkpoint  |  Supplied-answer likelihood; no freeform accuracy score')
    card(fig, .055, .262, .908, .51)
    fig.text(.077, .732, 'Correct answer preferred', color=INK, fontsize=17, weight='bold')
    fig.text(.077, .697, 'Dots = measured score   •   Lines = 95% bootstrap interval', color=MUTED, fontsize=11.5)
    ax = fig.add_axes([.238, .425, .578, .211])
    style_axes(ax)
    ax.grid(False)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.axvspan(threshold, 100, color='#eaf2ef')
    ax.axvline(threshold, color=TEAL, linestyle=(0, (4, 4)), linewidth=1.5)
    ax.text(threshold, 1.51, f'{threshold:.0f}% score threshold', color=TEAL, fontsize=11.5,
            ha='center', va='bottom')
    for index, (prefix, color) in enumerate([('mean', BLUE), ('sum', PURPLE)]):
        row = 1 - index
        value = overall[f'{prefix}_preference_accuracy'] * 100
        interval = overall[f'{prefix}_preference_ci95']
        lo, hi = interval['lower'] * 100, interval['upper'] * 100
        ax.errorbar(value, row, xerr=[[value - lo], [hi - value]], fmt='o', color=color,
                    markersize=10, capsize=7, capthick=2, elinewidth=3)
        credit = overall[f'{prefix}_preference_accuracy'] * count
        ax.text(value, row + .24, f'{credit:g}/{count}  ·  {value:.1f}%',
                color=color, ha='center', fontsize=14, weight='bold')
        ax.text(100, row, f'{lo:.1f}–{hi:.1f}%', color=MUTED, fontsize=11.5,
                ha='left', va='center', clip_on=False)
    ax.set_yticks([1, 0], ['Mean token log-prob.', 'Summed log-prob.'])
    ax.set_ylim(-.43, 1.48)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel('Paired-answer preference rate', color=MUTED, fontsize=11.5, labelpad=11)
    ax.spines[['left', 'bottom']].set_visible(False)
    fig.text(.077, .29, f'Pass requires both scores ≥ {threshold:.0f}% AND both interval lower bounds > {lower_required:.0f}%. Neither metric qualifies.',
             color=RED, fontsize=12, weight='bold')
    card(fig, .055, .105, .908, .106, '#fbecef')
    fig.text(.073, .174, 'Small diagnostic probe; no broad competence claim.', color=INK, fontsize=13, weight='bold')
    fig.text(.073, .138, f'{quality["limits"]["bootstrap_samples"]:,} bootstrap resamples show sensitivity within this set. Intervals do not establish performance on all C++ tasks.',
             color=MUTED, fontsize=11.5)
    fig.text(.055, .048, 'Source: reports/ai/cpp-quality.json  |  All 18 examples scored; no ties or rejected examples in this recorded run.',
             fontsize=10.5, color=MUTED)
    return save(fig, output, 'cpp-quality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project.resolve()
    output = root / 'reports/ai/figures'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'svg.fonttype': 'path',
        'svg.hashsalt': 'questline-ai-results-v1', 'axes.unicode_minus': False,
    })
    training, framework, application, environment, quality = load_evidence(root)
    paths = training_figure(training, environment, output)
    paths += execution_figure(framework, application, environment, output)
    paths += quality_figure(quality, output)
    manifest = {
        'purpose': 'Static figures of existing recorded evidence; no training or tests rerun.',
        'renderer': 'scripts/plot_ai_results.py',
        'renderer_sha256': sha256(Path(__file__)),
        'python_version': platform.python_version(),
        'matplotlib_version': matplotlib.__version__,
        'source_sha256': {name: sha256(root / name) for name in SOURCES},
        'figure_sha256': {path.name: sha256(path) for path in paths},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'figures_written': len(paths), 'source_files_hashed': len(SOURCES),
                      'output_directory': str(output.relative_to(root))}, indent=2))


if __name__ == '__main__':
    main()
