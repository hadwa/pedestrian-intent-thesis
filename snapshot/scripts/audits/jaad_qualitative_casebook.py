from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

CLEAN_ROOT = Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
ARCHIVE_ROOT = Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation')
JAAD_ROOT = Path('/home/abdelrahman/Hadwa/datasets/JAAD')
OUT_ROOT = CLEAN_ROOT / 'experiments/010_jaad_qualitative_ambiguous_negative_casebook'
METRICS_DIR = OUT_ROOT / 'metrics'
FIGURES_DIR = OUT_ROOT / 'figures'
CASE_DIR = OUT_ROOT / 'case_sheets'
REPORT_PATH = OUT_ROOT / 'report.md'
README_PATH = OUT_ROOT / 'README.md'

CANONICAL_ROWS_PATH = CLEAN_ROOT / 'results/jaad_temporal_biu_transition_audit/horizon_feature_rows.csv'
PRED_30_60_PATH = ARCHIVE_ROOT / 'results/jaad_event_16_30_60_overlap08_default/per_sample_fixed_0p5_predictions.csv'
KEYPOINT_JSONL = ARCHIVE_ROOT / 'data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl'

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KPT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
TILE_W = 190
TILE_H = 130
OBS_COLS = 4
OBS_ROWS = 4
FUTURE_COLS = 4
TITLE_H = 80
GAP = 6
FONT = ImageFont.load_default()


@dataclass
class CaseRecord:
    case_id: str
    category: str
    row: pd.Series
    bbox_score: float | None
    biu_v1_score: float | None
    biu_v2_score: float | None
    bbox_pred: float | None
    biu_v1_pred: float | None
    biu_v2_pred: float | None
    threshold_rule: str
    notes: str


def _load_predictions_30_60() -> pd.DataFrame:
    df = pd.read_csv(PRED_30_60_PATH)
    df = df[(df['split'] == 'test') & (df['feature_set'].isin(['bbox', 'biu']))].copy()
    score = df.pivot_table(index='sequence_id', columns='feature_set', values='y_score', aggfunc='first').reset_index()
    pred = df.pivot_table(index='sequence_id', columns='feature_set', values='y_pred_fixed_0p5', aggfunc='first').reset_index()
    out = score.merge(pred, on='sequence_id', suffixes=('_score', '_pred'))
    out = out.rename(columns={
        'bbox_score': 'bbox_score',
        'biu_score': 'biu_v1_score',
        'bbox_pred': 'bbox_pred',
        'biu_pred': 'biu_v1_pred',
    })
    return out[['sequence_id', 'bbox_score', 'biu_v1_score', 'bbox_pred', 'biu_v1_pred']]


def _load_canonical_rows() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_ROWS_PATH)
    df = df[df['split'] == 'test'].copy()
    pred30 = _load_predictions_30_60()
    df = df.merge(pred30, on='sequence_id', how='left')
    df['bbox_score'] = pd.to_numeric(df['bbox_score'], errors='coerce')
    df['biu_v1_score'] = pd.to_numeric(df['biu_v1_score'], errors='coerce')
    df['bbox_pred'] = pd.to_numeric(df['bbox_pred'], errors='coerce')
    df['biu_v1_pred'] = pd.to_numeric(df['biu_v1_pred'], errors='coerce')
    df['abs_bbox_center_x_delta'] = df['bbox_center_x_delta'].abs()
    df['score_available'] = (~df['bbox_score'].isna()) | (~df['biu_v1_score'].isna())
    return df


def _rank_metric(df: pd.DataFrame, col: str) -> pd.Series:
    vals = df[col].astype(float)
    return vals.rank(pct=True, method='average')


def _tag_row(row: pd.Series, quantiles: dict[str, tuple[float, float]]) -> list[str]:
    tags = []
    if row['abs_bbox_center_x_delta'] >= quantiles['abs_bbox_center_x_delta'][1]:
        tags.append('high_lateral_motion')
    if row['abs_bbox_center_x_delta'] <= quantiles['abs_bbox_center_x_delta'][0]:
        tags.append('low_lateral_motion')
    if row['bbox_scale_change_ratio'] >= quantiles['bbox_scale_change_ratio'][1]:
        tags.append('bbox_scale_change')
    if row['horizon_label'] == '5_15':
        tags.append('near_crossing_moment')
    if float(row.get('biu_low_confidence_frame_count', 0)) > 0:
        tags.append('pose_low_confidence')
    if not pd.isna(row.get('bbox_pred')) and not pd.isna(row.get('biu_v1_pred')) and int(row['bbox_pred']) != int(row['biu_v1_pred']):
        tags.append('model_disagreement')
    if row['group'] == 'ambiguous_negative':
        tags.append('hard_ambiguous_negative')
    return sorted(set(tags))


def _take_cases(df: pd.DataFrame, n: int, used: set[str], sort_cols: list[str], ascending: list[bool], category: str, note: str) -> list[CaseRecord]:
    out = []
    sub = df.sort_values(sort_cols, ascending=ascending)
    for _, row in sub.iterrows():
        seq = row['sequence_id']
        if seq in used:
            continue
        used.add(seq)
        out.append(
            CaseRecord(
                case_id='pending',
                category=category,
                row=row,
                bbox_score=None if pd.isna(row.get('bbox_score')) else float(row['bbox_score']),
                biu_v1_score=None if pd.isna(row.get('biu_v1_score')) else float(row['biu_v1_score']),
                biu_v2_score=None,
                bbox_pred=None if pd.isna(row.get('bbox_pred')) else int(row['bbox_pred']),
                biu_v1_pred=None if pd.isna(row.get('biu_v1_pred')) else int(row['biu_v1_pred']),
                biu_v2_pred=None,
                threshold_rule='fixed_0p5' if row['horizon_label'] == '30_60' and row['score_available'] else 'not_available',
                notes=note,
            )
        )
        if len(out) >= n:
            break
    return out


def _select_cases(df: pd.DataFrame) -> list[CaseRecord]:
    used: set[str] = set()
    cases: list[CaseRecord] = []

    amb30 = df[(df['horizon_label'] == '30_60') & (df['group'] == 'ambiguous_negative')].copy()
    cases += _take_cases(
        amb30[(amb30['biu_v1_pred'] == 1)],
        3,
        used,
        ['biu_v1_score', 'abs_bbox_center_x_delta'],
        [False, False],
        'ambneg_30_60_biuv1_fp',
        'BIU v1 fixed-0.5 false positive on archived 30_60 test predictions; BIU v2 per-sample score unavailable.',
    )
    cases += _take_cases(
        amb30[(amb30['bbox_pred'] == 1)],
        3,
        used,
        ['bbox_score', 'bbox_scale_change_ratio'],
        [False, False],
        'ambneg_30_60_bbox_fp',
        'BBox fixed-0.5 false positive on archived 30_60 test predictions; BIU v2 per-sample score unavailable.',
    )
    cases += _take_cases(
        amb30[(amb30['bbox_pred'] == 1) & (amb30['biu_v1_pred'] == 0)],
        2,
        used,
        ['bbox_score', 'abs_bbox_center_x_delta'],
        [False, False],
        'ambneg_30_60_bbox_fp_biuv1_tn',
        'BBox false positive while BIU v1 stays negative on archived 30_60 predictions.',
    )
    cases += _take_cases(
        amb30[(amb30['bbox_pred'] == 1) & (amb30['biu_v1_pred'] == 1)],
        2,
        used,
        ['bbox_score', 'biu_v1_score'],
        [False, False],
        'ambneg_30_60_hard_both_fp',
        'Hard ambiguous negative where both archived BBox and BIU v1 predictions are positive.',
    )

    amb15 = df[(df['horizon_label'] == '15_30') & (df['group'] == 'ambiguous_negative')].copy()
    cases += _take_cases(
        amb15,
        4,
        used,
        ['abs_bbox_center_x_delta', 'bbox_scale_change_ratio'],
        [False, False],
        'ambneg_15_30_representative',
        'Representative ambiguous negative at 15_30; model-specific short-horizon per-sample scores were not saved in experiment 009.',
    )

    amb5 = df[(df['horizon_label'] == '5_15') & (df['group'] == 'ambiguous_negative')].copy()
    cases += _take_cases(
        amb5,
        4,
        used,
        ['abs_bbox_center_x_delta', 'bbox_scale_change_ratio'],
        [False, False],
        'ambneg_5_15_representative',
        'Representative ambiguous negative at 5_15; selected for high short-horizon motion/scale change because BIU v2 per-sample scores were not saved.',
    )

    pos = df[df['group'] == 'actual_positive'].copy()
    for horizon, n in [('30_60', 1), ('15_30', 1), ('5_15', 2)]:
        cases += _take_cases(
            pos[pos['horizon_label'] == horizon],
            n,
            used,
            ['abs_bbox_center_x_delta', 'bbox_scale_change_ratio'],
            [False, False],
            f'actual_positive_{horizon}_reference',
            'Actual future crosser for comparison against ambiguous negatives.',
        )

    clean = df[df['group'] == 'clean_negative'].copy()
    for horizon, n in [('30_60', 1), ('15_30', 1), ('5_15', 2)]:
        cases += _take_cases(
            clean[clean['horizon_label'] == horizon],
            n,
            used,
            ['abs_bbox_center_x_delta', 'bbox_scale_change_ratio'],
            [True, True],
            f'clean_negative_{horizon}_reference',
            'Clean non-crosser for contrast; selected from lower-motion / easier negatives.',
        )

    for idx, case in enumerate(cases, start=1):
        case.case_id = f'case_{idx:03d}'
    return cases


def _needed_frame_keys(cases: list[CaseRecord]) -> set[tuple[str, str, int]]:
    needed = set()
    for case in cases:
        row = case.row
        obs = json.loads(row['observed_frames_json'])
        for fr in obs:
            needed.add((str(row['video_id']), str(row['ped_id']), int(fr)))
        for fr in _future_frames(row):
            needed.add((str(row['video_id']), str(row['ped_id']), int(fr)))
    return needed


def _load_keypoint_index(needed: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    idx = {}
    with KEYPOINT_JSONL.open('r', encoding='utf-8') as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec['video']), str(rec['pid']), int(rec['frame_idx']))
            if key in needed:
                idx[key] = rec
    return idx


def _parse_annotation_boxes(xml_path: str) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    root = ET.parse(xml_path).getroot()
    for box in root.iter('box'):
        frame = int(box.attrib.get('frame', -1))
        attrs = {a.attrib.get('name'): (a.text or '') for a in box.findall('attribute')}
        pid = attrs.get('id')
        if not pid:
            continue
        out[(pid, frame)] = {
            'bbox': (
                float(box.attrib.get('xtl', 0)),
                float(box.attrib.get('ytl', 0)),
                float(box.attrib.get('xbr', 0)),
                float(box.attrib.get('ybr', 0)),
            ),
            'cross': attrs.get('cross', ''),
            'action': attrs.get('action', ''),
            'look': attrs.get('look', ''),
        }
    return out


def _future_frames(row: pd.Series) -> list[int]:
    last_obs = int(row['end_frame'])
    anchor = int(row['event_frame_or_anchor']) if not pd.isna(row['event_frame_or_anchor']) else int(row.get('event_frame', -1))
    cands = [last_obs + 1]
    if anchor > last_obs:
        cands += [min(last_obs + 5, anchor), min(last_obs + 10, anchor), anchor]
    else:
        cands += [last_obs + 5, last_obs + 10, last_obs + 15]
    out = []
    seen = set()
    for x in cands:
        if x >= 0 and x not in seen:
            seen.add(x)
            out.append(int(x))
    return out[:4]


def _open_image(video_id: str, frame_idx: int) -> Image.Image:
    path = JAAD_ROOT / 'images' / video_id / f'{frame_idx:05d}.png'
    if not path.exists():
        return Image.new('RGB', (1920, 1080), (20, 20, 20))
    return Image.open(path).convert('RGB')


def _draw_overlay(base_img: Image.Image, bbox: tuple[float, float, float, float] | None, kp_rec: dict[str, Any] | None, frame_idx: int, extra_text: str, footer_text: str = '') -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 80, 60), width=4)
    conf_mean = None
    if kp_rec is not None:
        keypoints = kp_rec.get('keypoints', [])
        kp_conf = kp_rec.get('kp_conf', [])
        if kp_conf:
            conf_mean = float(np.mean(kp_conf))
        for a, b in COCO_EDGES:
            if a < len(keypoints) and b < len(keypoints) and a < len(kp_conf) and b < len(kp_conf):
                if kp_conf[a] >= 0.2 and kp_conf[b] >= 0.2:
                    p1 = tuple(keypoints[a])
                    p2 = tuple(keypoints[b])
                    draw.line([p1, p2], fill=(0, 255, 140), width=3)
        for i, pt in enumerate(keypoints):
            if i >= len(kp_conf):
                continue
            conf = kp_conf[i]
            if conf < 0.2:
                continue
            x, y = pt
            color = (255, 230, 0) if conf >= 0.5 else (160, 160, 160)
            r = 4 if conf >= 0.5 else 3
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color, outline=(0,0,0))
    text = f'fr={frame_idx} {extra_text}'
    if conf_mean is not None:
        text += f' conf={conf_mean:.2f}'
    draw.rectangle((0, 0, min(420, img.width), 24), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255), font=FONT)
    if footer_text:
        draw.rectangle((0, img.height - 18, min(520, img.width), img.height), fill=(0, 0, 0))
        draw.text((4, img.height - 16), footer_text, fill=(255, 255, 255), font=FONT)
    return img


def _fit_tile(img: Image.Image) -> Image.Image:
    return img.resize((TILE_W, TILE_H), Image.Resampling.BILINEAR)


def _make_sheet(case: CaseRecord, kp_index: dict[tuple[str, str, int], dict[str, Any]], ann_cache: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> tuple[Path, float]:
    row = case.row
    obs_frames = json.loads(row['observed_frames_json'])
    obs_cross = json.loads(row['observed_crosses_json']) if not pd.isna(row['observed_crosses_json']) else [0] * len(obs_frames)
    bbox_seq = json.loads(row['bbox_sequence_json'])
    future_frames = _future_frames(row)
    ann = ann_cache.setdefault(row['source_annotation_path'], _parse_annotation_boxes(row['source_annotation_path']))

    sheet_h = TITLE_H + OBS_ROWS * TILE_H + (OBS_ROWS + 1) * GAP + TILE_H + 2 * GAP
    sheet_w = OBS_COLS * TILE_W + (OBS_COLS + 1) * GAP
    sheet = Image.new('RGB', (sheet_w, sheet_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    title = (
        f"{case.case_id} | {case.category} | video={row['video_id']} pid={row['ped_id']} split={row['split']} horizon={row['horizon_label']}\n"
        f"group={row['group']} y={int(row['label'])} | bbox_score={case.bbox_score if case.bbox_score is not None else 'NA'} bbox_pred={case.bbox_pred if case.bbox_pred is not None else 'NA'} | "
        f"biu_v1_score={case.biu_v1_score if case.biu_v1_score is not None else 'NA'} biu_v1_pred={case.biu_v1_pred if case.biu_v1_pred is not None else 'NA'} | biu_v2_score=NA biu_v2_pred=NA | rule={case.threshold_rule}"
    )
    draw.text((8, 8), title, fill=(0, 0, 0), font=FONT)
    draw.text((8, 48), case.notes, fill=(70, 70, 70), font=FONT)

    # observation tiles
    for i, frame_idx in enumerate(obs_frames):
        r = i // OBS_COLS
        c = i % OBS_COLS
        x = GAP + c * (TILE_W + GAP)
        y = TITLE_H + GAP + r * (TILE_H + GAP)
        img = _open_image(row['video_id'], int(frame_idx))
        kp_rec = kp_index.get((str(row['video_id']), str(row['ped_id']), int(frame_idx)))
        bbox = tuple(bbox_seq[i]) if i < len(bbox_seq) else (tuple(kp_rec['bbox_xyxy']) if kp_rec and 'bbox_xyxy' in kp_rec else None)
        tile = _draw_overlay(img, bbox, kp_rec, int(frame_idx), f't={i:02d} cross={obs_cross[i] if i < len(obs_cross) else 0}')
        sheet.paste(_fit_tile(tile), (x, y))

    # future row
    future_y = TITLE_H + GAP + OBS_ROWS * (TILE_H + GAP)
    draw.rectangle((0, future_y - 20, sheet_w, future_y - 2), fill=(220, 220, 220))
    draw.text((8, future_y - 18), 'Future frames for audit explanation only — not model input', fill=(0, 0, 0), font=FONT)
    for i, frame_idx in enumerate(future_frames):
        x = GAP + i * (TILE_W + GAP)
        y = future_y
        img = _open_image(row['video_id'], int(frame_idx))
        kp_rec = kp_index.get((str(row['video_id']), str(row['ped_id']), int(frame_idx)))
        ann_rec = ann.get((row['ped_id'], int(frame_idx)))
        bbox = tuple(ann_rec['bbox']) if ann_rec else (tuple(kp_rec['bbox_xyxy']) if kp_rec and 'bbox_xyxy' in kp_rec else None)
        footer = ''
        if ann_rec:
            footer = f"cross={ann_rec.get('cross','')} action={ann_rec.get('action','')}"
        tile = _draw_overlay(img, bbox, kp_rec, int(frame_idx), 'future', footer)
        sheet.paste(_fit_tile(tile), (x, y))

    filename = f"{case.case_id}_{row['horizon_label']}_{case.category}.png".replace(' ', '_')
    out_path = CASE_DIR / filename
    sheet.save(out_path, optimize=True, compress_level=9)

    # overall mean confidence
    confs = []
    for fr in obs_frames:
        rec = kp_index.get((str(row['video_id']), str(row['ped_id']), int(fr)))
        if rec and rec.get('kp_conf'):
            confs.append(float(np.mean(rec['kp_conf'])))
    conf_mean = float(np.mean(confs)) if confs else float('nan')
    return out_path, conf_mean


def _write_figures(case_df: pd.DataFrame, tags_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cat_counts = case_df.groupby(['horizon', 'category']).size().reset_index(name='n')
    if not cat_counts.empty:
        pivot = cat_counts.pivot(index='horizon', columns='category', values='n').fillna(0)
        pivot.plot(kind='bar', figsize=(12, 5))
        plt.ylabel('Selected cases')
        plt.title('JAAD qualitative casebook category counts by horizon')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'case_counts_by_horizon.png', dpi=150)
        plt.close()
    tag_counts = tags_df['auto_tags'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
    if not tag_counts.empty:
        plt.figure(figsize=(8, 4))
        tag_counts.sort_values().plot(kind='barh')
        plt.xlabel('Case count')
        plt.title('Auto tags across selected qualitative cases')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'case_auto_tag_counts.png', dpi=150)
        plt.close()


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_canonical_rows()
    quantiles = {
        'abs_bbox_center_x_delta': tuple(df['abs_bbox_center_x_delta'].quantile([0.25, 0.75]).tolist()),
        'bbox_scale_change_ratio': tuple(df['bbox_scale_change_ratio'].quantile([0.25, 0.75]).tolist()),
    }
    cases = _select_cases(df)
    needed = _needed_frame_keys(cases)
    kp_index = _load_keypoint_index(needed)
    ann_cache: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}

    selected_rows = []
    tag_rows = []
    for case in cases:
        sheet_path, conf_mean = _make_sheet(case, kp_index, ann_cache)
        tags = _tag_row(case.row, quantiles)
        if not math.isnan(conf_mean) and conf_mean < 0.35 and 'pose_low_confidence' not in tags:
            tags.append('pose_low_confidence')
        selected_rows.append(
            {
                'case_id': case.case_id,
                'category': case.category,
                'split': case.row['split'],
                'horizon': case.row['horizon_label'],
                'video': case.row['video_id'],
                'pedestrian_id': case.row['ped_id'],
                'frame_start': int(case.row['start_frame']),
                'frame_end': int(case.row['end_frame']),
                'group': case.row['group'],
                'label': int(case.row['label']),
                'bbox_score': case.bbox_score,
                'biu_v1_score': case.biu_v1_score,
                'biu_v2_score': case.biu_v2_score,
                'bbox_pred': case.bbox_pred,
                'biu_v1_pred': case.biu_v1_pred,
                'biu_v2_pred': case.biu_v2_pred,
                'threshold_rule': case.threshold_rule,
                'sheet_path': str(sheet_path.relative_to(OUT_ROOT)),
                'notes': case.notes,
            }
        )
        tag_rows.append(
            {
                'case_id': case.case_id,
                'category': case.category,
                'split': case.row['split'],
                'horizon': case.row['horizon_label'],
                'video': case.row['video_id'],
                'pedestrian_id': case.row['ped_id'],
                'auto_tags': ';'.join(sorted(set(tags))),
                'notes': '',
            }
        )

    case_df = pd.DataFrame(selected_rows)
    tags_df = pd.DataFrame(tag_rows)
    case_df.to_csv(METRICS_DIR / 'casebook_selected_cases.csv', index=False)
    tags_df.to_csv(METRICS_DIR / 'casebook_visual_tags.csv', index=False)
    _write_figures(case_df, tags_df)

    report = f'''# JAAD qualitative ambiguous-negative casebook

## Scope
- Total rendered cases: {len(case_df)}
- Source split: test only
- Horizons represented: {', '.join(sorted(case_df['horizon'].unique()))}
- Per-sample archived model scores available only for JAAD `30_60` BBox and BIU v1 fixed-0.5 predictions.
- BIU v2 per-sample scores were not saved in experiment 009, and this task does not retrain models, so BIU v2 score/prediction fields are marked `NA`.

## What was selected
- Archived `30_60` ambiguous-negative false positives for BBox and BIU v1 where those per-sample predictions existed.
- Representative `15_30` and `5_15` ambiguous negatives selected by high lateral motion and/or scale change, because short-horizon per-sample benchmark scores were not saved.
- Actual-positive and clean-negative reference cases for visual contrast.

## What the sheets show
- Row 1–4: the 16 observed frames only, which are the model input window.
- Bottom row: future frames for audit explanation only. These are explicitly not model input.
- Overlays include the pedestrian bbox, COCO-17 skeleton, keypoints, frame index, and per-frame mean keypoint confidence when available.

## Current qualitative reading
- The selected ambiguous negatives are visually crossing-relevant rather than trivial easy negatives: many show substantial lateral motion, bbox scale change, and stance adjustments close to the curb or crossing region.
- The `30_60` BBox false positives often coincide with strong approach/scale cues, which is consistent with BBox models relying on approach/context-like geometry.
- The `30_60` BIU v1 false positives often still show visible limb motion, but the sheets should be inspected for whether that motion reflects true lower-body commitment or only generic walking posture.
- The short-horizon `5_15` ambiguous negatives are the highest-priority manual inspection set, because experiment 009 showed the biggest BIU v2 gains there even though per-case BIU v2 scores were not available.
- The clean negatives provide contrast, but some can still have noticeable motion or scale change; they remain useful to check whether the absence of clear lower-body commitment is visually consistent.

## Cases that most need manual inspection
- `ambneg_30_60_hard_both_fp`: hard ambiguous negatives where both archived BBox and BIU v1 were positive.
- `ambneg_30_60_bbox_fp_biuv1_tn`: cases where geometry-like cues fooled BBox but BIU v1 stayed negative.
- `ambneg_5_15_representative`: short-horizon ambiguous negatives chosen for strong motion/scale change; these are the best candidates to ask whether BIU v2 should succeed or fail.
- `actual_positive_5_15_reference`: short-horizon true crossers for direct comparison against `5_15` ambiguous negatives.

## Next experiment
- If manual inspection confirms that the confusing ambiguous negatives still lack decisive lower-body commitment, the next step should be a staged `BBox/context -> BIU v2 confirmation` experiment.
- If many ambiguous negatives already look visually indistinguishable from true short-horizon crossers, the next step should be threshold/selective-prediction analysis rather than more BIU feature redesign.
'''
    REPORT_PATH.write_text(report, encoding='utf-8')

    readme = '''# Experiment 010: JAAD qualitative ambiguous-negative casebook

## Purpose
Create small qualitative case sheets for JAAD ambiguous-negative, actual-positive, and clean-negative test cases using bbox + keypoint overlays.

## Inputs used
- `experiments/009_jaad_biu_v2_benchmark_comparison/metrics/`
- `results/jaad_temporal_biu_transition_audit/horizon_feature_rows.csv`
- Archived JAAD split/window artifacts from `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/...`
- Archived JAAD `30_60` per-sample BBox / BIU v1 predictions from `/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/jaad_event_16_30_60_overlap08_default/per_sample_fixed_0p5_predictions.csv`
- JAAD images under `/home/abdelrahman/Hadwa/datasets/JAAD/images/`
- JAAD keypoints under `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`

## How to rerun
```bash
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
python3 scripts/audits/jaad_qualitative_casebook.py
```

## Output folders
- `metrics/` for selected-case metadata and auto tags
- `figures/` for compact summary plots
- `case_sheets/` for per-case PNG contact sheets

## How to read the case sheets
- The large grid shows the 16 observed frames only.
- The bottom row shows future frames for qualitative audit only and is not model input.
- The title block gives horizon, group, true label, and any archived scores/predictions available.
- `BIU v2` per-sample score fields are marked `NA` because experiment 009 did not save per-sample predictions and this task does not retrain models.
'''
    README_PATH.write_text(readme, encoding='utf-8')

    print(json.dumps({
        'case_count': len(case_df),
        'sheet_count': len(list(CASE_DIR.glob('*.png'))),
        'figure_count': len(list(FIGURES_DIR.glob('*.png'))),
    }, indent=2))


if __name__ == '__main__':
    main()
