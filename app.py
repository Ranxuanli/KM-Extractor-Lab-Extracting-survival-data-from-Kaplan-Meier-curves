import base64
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
STATIC_DIR = BASE_DIR / 'static'
CACHE_DIR = STATIC_DIR / 'cache'
PLOT_DIR = STATIC_DIR / 'plots'
DATA_DIR = BASE_DIR / 'data'
for p in [UPLOAD_DIR, STATIC_DIR, CACHE_DIR, PLOT_DIR, DATA_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')
LLM_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4.1-mini')

PROMPT = """
You are extracting data from a Kaplan–Meier survival curve image for a biomedical methods website.
Return ONLY valid JSON with this schema:
{
  "title": "...",
  "x_label": "...",
  "y_label": "...",
  "source_note": "...",
  "arms": [
    {
      "name": "Treatment A",
      "n_total": 100,
      "curve_points": [
        {"time": 0, "survival": 1.0},
        {"time": 3, "survival": 0.95}
      ]
    },
    {
      "name": "Treatment B",
      "n_total": 100,
      "curve_points": [
        {"time": 0, "survival": 1.0},
        {"time": 3, "survival": 0.92}
      ]
    }
  ]
}
Rules:
1. Include exactly two arms if visible.
2. Use monotone non-increasing survival values between 0 and 1.
3. Capture major step changes along each curve.
4. Infer n_total from the figure if shown; otherwise choose a plausible placeholder integer.
5. Do not include markdown fences.
""".strip()

DEMO_EXTRACT = {
    "title": "Demo extraction from cached Kaplan–Meier plot",
    "x_label": "Months",
    "y_label": "Survival probability",
    "source_note": "Illustrative cached demo for the course project website.",
    "arms": [
        {
            "name": "Treatment A",
            "n_total": 84,
            "curve_points": [
                {"time": 0, "survival": 1.00},
                {"time": 2, "survival": 0.98},
                {"time": 4, "survival": 0.95},
                {"time": 6, "survival": 0.92},
                {"time": 8, "survival": 0.89},
                {"time": 10, "survival": 0.84},
                {"time": 12, "survival": 0.79},
                {"time": 14, "survival": 0.73}
            ]
        },
        {
            "name": "Treatment B",
            "n_total": 82,
            "curve_points": [
                {"time": 0, "survival": 1.00},
                {"time": 2, "survival": 0.95},
                {"time": 4, "survival": 0.90},
                {"time": 6, "survival": 0.84},
                {"time": 8, "survival": 0.77},
                {"time": 10, "survival": 0.69},
                {"time": 12, "survival": 0.61},
                {"time": 14, "survival": 0.55}
            ]
        }
    ]
}

INDIRECT_DEMO_STUDIES = {
    "study_ab": {
        "article_title": "Demo study 1: Treatment A versus Treatment B",
        "population": "Illustrative oncology population",
        "comparison_label": "A vs B",
        "experimental_arm_name": "Treatment A",
        "control_arm_name": "Treatment B",
        "treatment_map": {"Treatment A": "A", "Treatment B": "B"},
        "extract": {
            "title": "Demo study 1: A vs B",
            "x_label": "Months",
            "y_label": "Survival probability",
            "source_note": "Cached run for indirect-comparison demo.",
            "arms": [
                {
                    "name": "Treatment A",
                    "n_total": 120,
                    "curve_points": [
                        {"time": 0, "survival": 1.00},
                        {"time": 3, "survival": 0.97},
                        {"time": 6, "survival": 0.93},
                        {"time": 9, "survival": 0.88},
                        {"time": 12, "survival": 0.82},
                        {"time": 15, "survival": 0.76},
                        {"time": 18, "survival": 0.70}
                    ]
                },
                {
                    "name": "Treatment B",
                    "n_total": 118,
                    "curve_points": [
                        {"time": 0, "survival": 1.00},
                        {"time": 3, "survival": 0.95},
                        {"time": 6, "survival": 0.89},
                        {"time": 9, "survival": 0.82},
                        {"time": 12, "survival": 0.74},
                        {"time": 15, "survival": 0.66},
                        {"time": 18, "survival": 0.59}
                    ]
                }
            ]
        }
    },
    "study_bc": {
        "article_title": "Demo study 2: Treatment B versus Treatment C",
        "population": "Illustrative oncology population",
        "comparison_label": "B vs C",
        "experimental_arm_name": "Treatment B",
        "control_arm_name": "Treatment C",
        "treatment_map": {"Treatment B": "B", "Treatment C": "C"},
        "extract": {
            "title": "Demo study 2: B vs C",
            "x_label": "Months",
            "y_label": "Survival probability",
            "source_note": "Cached run for indirect-comparison demo.",
            "arms": [
                {
                    "name": "Treatment B",
                    "n_total": 132,
                    "curve_points": [
                        {"time": 0, "survival": 1.00},
                        {"time": 3, "survival": 0.96},
                        {"time": 6, "survival": 0.91},
                        {"time": 9, "survival": 0.84},
                        {"time": 12, "survival": 0.77},
                        {"time": 15, "survival": 0.69},
                        {"time": 18, "survival": 0.61}
                    ]
                },
                {
                    "name": "Treatment C",
                    "n_total": 129,
                    "curve_points": [
                        {"time": 0, "survival": 1.00},
                        {"time": 3, "survival": 0.93},
                        {"time": 6, "survival": 0.86},
                        {"time": 9, "survival": 0.77},
                        {"time": 12, "survival": 0.68},
                        {"time": 15, "survival": 0.58},
                        {"time": 18, "survival": 0.49}
                    ]
                }
            ]
        }
    }
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def cache_path(key: str) -> Path:
    return CACHE_DIR / f'{key}.json'


def plot_path(key: str) -> Path:
    return PLOT_DIR / f'{key}.png'


def normalize_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    arms = data.get('arms', [])
    normalized_arms = []
    for arm in arms:
        pts = arm.get('curve_points', [])
        cleaned = []
        for p in pts:
            try:
                t = float(p['time'])
                s = float(p['survival'])
            except Exception:
                continue
            if math.isnan(t) or math.isnan(s):
                continue
            cleaned.append({'time': max(0.0, t), 'survival': min(1.0, max(0.0, s))})
        cleaned = sorted(cleaned, key=lambda x: x['time'])
        if not cleaned:
            continue
        if cleaned[0]['time'] != 0:
            cleaned.insert(0, {'time': 0.0, 'survival': 1.0})
        cleaned[0]['survival'] = 1.0

        mono = []
        running = 1.0
        seen_times = set()
        for p in cleaned:
            if p['time'] in seen_times:
                continue
            seen_times.add(p['time'])
            running = min(running, p['survival'])
            mono.append({'time': p['time'], 'survival': running})

        normalized_arms.append({
            'name': str(arm.get('name', 'Arm')).strip() or 'Arm',
            'n_total': max(2, int(arm.get('n_total', 100))),
            'curve_points': mono,
        })

    data['arms'] = normalized_arms
    data['title'] = data.get('title', 'Extracted Kaplan–Meier curves')
    data['x_label'] = data.get('x_label', 'Time')
    data['y_label'] = data.get('y_label', 'Survival probability')
    data['source_note'] = data.get('source_note', '')
    return data


def extract_with_openai(image_path: Path) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError('openai package is not installed.')
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY is missing. Use cached demo mode or set the environment variable.')

    client = OpenAI(api_key=api_key)
    mime = 'image/png' if image_path.suffix.lower() == '.png' else 'image/jpeg'
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the two Kaplan–Meier survival curves into JSON."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content
    return normalize_extraction(json.loads(content))


def km_points_to_pseudo_ipd(points: List[Dict[str, float]], n_total: int) -> pd.DataFrame:
    pts = sorted(points, key=lambda x: x['time'])
    if not pts:
        return pd.DataFrame(columns=['time', 'event'])

    current_risk = int(n_total)
    prev_s = 1.0
    last_time = float(pts[-1]['time'])
    rows = []

    for p in pts[1:]:
        t = float(p['time'])
        s = float(p['survival'])
        if current_risk <= 0:
            break
        if prev_s <= 0:
            events = 0
        else:
            drop_fraction = max(0.0, 1.0 - (s / prev_s))
            events = int(round(current_risk * drop_fraction))
        events = max(0, min(current_risk, events))
        rows.extend([{'time': t, 'event': 1}] * events)
        current_risk -= events
        prev_s = s

    rows.extend([{'time': last_time, 'event': 0}] * current_risk)
    return pd.DataFrame(rows)


def run_pairwise_analysis(extraction: Dict[str, Any]) -> Dict[str, Any]:
    arms = extraction.get('arms', [])
    if len(arms) != 2:
        raise ValueError('This website currently expects exactly two arms in each extraction result.')

    arm1, arm2 = arms
    df1 = km_points_to_pseudo_ipd(arm1['curve_points'], arm1['n_total'])
    df2 = km_points_to_pseudo_ipd(arm2['curve_points'], arm2['n_total'])

    lr = logrank_test(
        df1['time'], df2['time'],
        event_observed_A=df1['event'],
        event_observed_B=df2['event']
    )

    cox_df = pd.concat([
        df1.assign(group=1),
        df2.assign(group=0),
    ], ignore_index=True)
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time', event_col='event')
    summary = cph.summary.loc['group']

    return {
        'arm1_name': arm1['name'],
        'arm2_name': arm2['name'],
        'n1': int(arm1['n_total']),
        'n2': int(arm2['n_total']),
        'test_statistic': float(lr.test_statistic),
        'p_value': float(lr.p_value),
        'hr_group1_vs_group0': float(summary['exp(coef)']),
        'log_hr': float(summary['coef']),
        'se_log_hr': float(summary['se(coef)']),
        'ci_low': float(summary['exp(coef) lower 95%']),
        'ci_high': float(summary['exp(coef) upper 95%']),
        'events_group1': int(df1['event'].sum()),
        'events_group0': int(df2['event'].sum()),
        'note': (
            'Approximate analysis based on pseudo individual patient data reconstructed from extracted '
            'Kaplan–Meier step curves. Censoring and exact risk-table information are not fully recoverable '
            'from the figure alone.'
        ),
    }


def make_plot(extraction: Dict[str, Any], out_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for arm in extraction.get('arms', []):
        pts = arm.get('curve_points', [])
        if not pts:
            continue
        x = [p['time'] for p in pts]
        y = [p['survival'] for p in pts]
        plt.step(x, y, where='post', linewidth=2.2, label=arm.get('name', 'Arm'))
    plt.xlabel(extraction.get('x_label', 'Time'))
    plt.ylabel(extraction.get('y_label', 'Survival probability'))
    plt.ylim(0, 1.05)
    plt.title(extraction.get('title', 'Extracted Kaplan–Meier curves'))
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def render_extraction_result(extraction: Dict[str, Any], cache_key: str) -> Dict[str, Any]:
    analysis = run_pairwise_analysis(extraction)
    plot_file = plot_path(cache_key)
    make_plot(extraction, plot_file)
    record = {
        'cache_key': cache_key,
        'extraction': extraction,
        'analysis': analysis,
        'plot_relpath': f'plots/{plot_file.name}',
    }
    write_json(cache_path(cache_key), record)
    return {
        'cache_key': cache_key,
        'plot_url': url_for('static', filename=f'plots/{plot_file.name}'),
        'extraction_pretty': json.dumps(extraction, indent=2),
        'analysis': analysis,
        'curve_summary': summarize_curves(extraction),
    }


def summarize_curves(extraction: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for arm in extraction.get('arms', []):
        pts = arm.get('curve_points', [])
        final_survival = pts[-1]['survival'] if pts else None
        rows.append({
            'name': arm.get('name', 'Arm'),
            'n_total': arm.get('n_total', 0),
            'n_points': len(pts),
            'final_time': pts[-1]['time'] if pts else None,
            'final_survival': final_survival,
        })
    return rows


def list_cache_records() -> List[Dict[str, Any]]:
    records = []
    for p in sorted(CACHE_DIR.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            payload = load_json(p)
            extraction = payload.get('extraction', payload)
            arms = extraction.get('arms', [])
            records.append({
                'cache_key': p.stem,
                'title': extraction.get('title', p.stem),
                'source_note': extraction.get('source_note', ''),
                'arm_names': ' vs '.join(a.get('name', 'Arm') for a in arms[:2]),
                'plot_url': url_for('static', filename=payload.get('plot_relpath', f'plots/{p.stem}.png')) if payload.get('plot_relpath') or plot_path(p.stem).exists() else None,
            })
        except Exception:
            continue
    return records[:8]


def bucher_indirect(log_hr_ab: float, se_ab: float, log_hr_bc: float, se_bc: float) -> Dict[str, float]:
    log_hr_ac = log_hr_ab + log_hr_bc
    se_ac = math.sqrt(se_ab ** 2 + se_bc ** 2)
    z = log_hr_ac / se_ac if se_ac > 0 else float('nan')
    p = math.erfc(abs(z) / math.sqrt(2.0)) if se_ac > 0 else float('nan')
    ci_low = math.exp(log_hr_ac - 1.96 * se_ac)
    ci_high = math.exp(log_hr_ac + 1.96 * se_ac)
    return {
        'log_hr_ac': log_hr_ac,
        'se_log_hr_ac': se_ac,
        'hr_ac': math.exp(log_hr_ac),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'z_value': z,
        'p_value': p,
    }


def build_indirect_demo_context() -> Dict[str, Any]:
    studies = {}
    for key, value in INDIRECT_DEMO_STUDIES.items():
        extract = normalize_extraction(value['extract'])
        analysis = run_pairwise_analysis(extract)
        studies[key] = {
            **value,
            'extract': extract,
            'analysis': analysis,
        }
    indirect = bucher_indirect(
        studies['study_ab']['analysis']['log_hr'],
        studies['study_ab']['analysis']['se_log_hr'],
        studies['study_bc']['analysis']['log_hr'],
        studies['study_bc']['analysis']['se_log_hr'],
    )
    return {'studies': studies, 'indirect': indirect}


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        mode = request.form.get('mode', 'live')
        try:
            if mode == 'demo':
                result = render_extraction_result(normalize_extraction(DEMO_EXTRACT.copy()), 'demo_example')
                flash('Loaded cached demo run. This mode is ideal for screenshots because it requires no API key.')
            else:
                image = request.files.get('image')
                if not image or not image.filename:
                    flash('Please upload a PNG or JPEG Kaplan–Meier figure.')
                    return redirect(url_for('index'))
                suffix = Path(image.filename).suffix.lower()
                if suffix not in ['.png', '.jpg', '.jpeg']:
                    flash('Unsupported file type. Please upload .png, .jpg, or .jpeg.')
                    return redirect(url_for('index'))
                temp_path = UPLOAD_DIR / f'upload_{hashlib.md5(os.urandom(16)).hexdigest()}{suffix}'
                image.save(temp_path)
                image_hash = sha256_file(temp_path)
                existing = cache_path(image_hash)
                if existing.exists():
                    payload = load_json(existing)
                    extraction = payload.get('extraction', payload)
                    result = render_extraction_result(extraction, image_hash)
                    flash('Loaded existing cached result for this figure.')
                else:
                    extraction = extract_with_openai(temp_path)
                    result = render_extraction_result(extraction, image_hash)
                    flash('Live extraction completed and cached successfully.')
        except Exception as e:
            flash(f'Run failed: {e}')
            return redirect(url_for('index'))

    return render_template(
        'index.html',
        page='home',
        result=result,
        cache_records=list_cache_records(),
    )


@app.route('/indirect', methods=['GET', 'POST'])
def indirect():
    context = build_indirect_demo_context()

    if request.method == 'POST':
        try:
            log_hr_ab = float(request.form.get('log_hr_ab', '0'))
            se_ab = float(request.form.get('se_ab', '1'))
            log_hr_bc = float(request.form.get('log_hr_bc', '0'))
            se_bc = float(request.form.get('se_bc', '1'))
            custom = bucher_indirect(log_hr_ab, se_ab, log_hr_bc, se_bc)
            context['custom'] = {
                'log_hr_ab': log_hr_ab,
                'se_ab': se_ab,
                'log_hr_bc': log_hr_bc,
                'se_bc': se_bc,
                'result': custom,
            }
            flash('Custom indirect comparison computed successfully.')
        except Exception as e:
            flash(f'Could not compute custom indirect comparison: {e}')
            return redirect(url_for('indirect'))

    return render_template('indirect.html', page='indirect', **context)


if __name__ == '__main__':
    app.run(debug=True)
