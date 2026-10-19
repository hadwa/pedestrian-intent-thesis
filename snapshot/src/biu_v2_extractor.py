"""Direction-aware BIU v2 feature specification and extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from .biu_extractor import extract_biu_features
    from .pose_utils import midpoint
except ImportError:  # Supports direct execution from the repository's src directory.
    from biu_extractor import extract_biu_features
    from pose_utils import midpoint


COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KPT_INDEX = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}
LEFT_RIGHT_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]
SUMMARY_POLICY = {
    "original_anatomical_biu": "mean;std;last;delta",
    "lower_body_commitment_biu": "last;delta;velocity_mean;velocity_std",
    "torso_center_commitment_biu": "last;delta;velocity_mean;velocity_std",
    "head_attention_proxy_biu": "last;delta;velocity_mean",
    "scale_viewpoint_metadata": "mean;delta",
    "confidence_metadata": "mean;min",
}


FEATURE_DICTIONARY: list[dict[str, str]] = [
    {
        "feature_name": "torso_lean_angle",
        "group": "original_anatomical_biu",
        "body_region": "torso",
        "definition": "Signed torso lean angle from shoulder midpoint to hip midpoint.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "required",
        "rationale": "Retained for continuity with BIU v1.",
        "evidence_source": "v1_continuity;experiment_007_reproduction",
        "caveats": "Weaker than lower-body motion cues on JAAD.",
    },
    {
        "feature_name": "body_yaw_angle",
        "group": "original_anatomical_biu",
        "body_region": "torso",
        "definition": "Signed shoulder-line yaw angle in canonical coordinates.",
        "required_keypoints": "left_shoulder;right_shoulder",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "required",
        "rationale": "Retained posture cue with directional consistency.",
        "evidence_source": "v1_continuity;experiment_007",
        "caveats": "Secondary to lower-body commitment features.",
    },
    {
        "feature_name": "head_yaw_angle",
        "group": "original_anatomical_biu",
        "body_region": "head",
        "definition": "Head yaw angle from neck to nose vector.",
        "required_keypoints": "nose;left_shoulder;right_shoulder",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "required",
        "rationale": "Head orientation proxy retained for continuity.",
        "evidence_source": "v1_continuity",
        "caveats": "Less robust than lower-body motion on JAAD.",
    },
    {
        "feature_name": "face_yaw_angle",
        "group": "original_anatomical_biu",
        "body_region": "head",
        "definition": "Face yaw from nose to hip-mid vector.",
        "required_keypoints": "nose;left_hip;right_hip",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "optional",
        "rationale": "Retained head-attention proxy.",
        "evidence_source": "v1_continuity",
        "caveats": "Can be noisy under occlusion or view changes.",
    },
    {
        "feature_name": "gaze_to_car_angle",
        "group": "original_anatomical_biu",
        "body_region": "head",
        "definition": "Ear-midpoint to nose orientation proxy relative to image-up reference.",
        "required_keypoints": "nose;left_ear;right_ear",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "optional",
        "rationale": "Keep only as attention proxy, not primary commitment cue.",
        "evidence_source": "v1_continuity;experiment_006",
        "caveats": "Ear visibility can fail; not primary signal.",
    },
    {
        "feature_name": "stride_length",
        "group": "original_anatomical_biu",
        "body_region": "lower_body",
        "definition": "Distance between left and right ankles normalized by shoulder width.",
        "required_keypoints": "left_ankle;right_ankle;left_shoulder;right_shoulder",
        "normalization": "shoulder_width_normalized",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "required",
        "rationale": "Compact legacy gait cue.",
        "evidence_source": "v1_continuity",
        "caveats": "Magnitude-only; weaker than directional deltas.",
    },
    {
        "feature_name": "lead_foot_elevation",
        "group": "original_anatomical_biu",
        "body_region": "lower_body",
        "definition": "Vertical ankle separation normalized by shoulder width.",
        "required_keypoints": "left_ankle;right_ankle;left_hip;right_hip;left_shoulder;right_shoulder",
        "normalization": "shoulder_width_normalized",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "required",
        "rationale": "Retained v1 gait cue.",
        "evidence_source": "v1_continuity",
        "caveats": "Does not encode forward commitment direction.",
    },
    {
        "feature_name": "lead_foot_lifted",
        "group": "original_anatomical_biu",
        "body_region": "lower_body",
        "definition": "Binary indicator derived from lead_foot_elevation threshold.",
        "required_keypoints": "left_ankle;right_ankle;left_hip;right_hip;left_shoulder;right_shoulder",
        "normalization": "shoulder_width_normalized",
        "direction_aware": "no",
        "summary_stats": "mean;last;delta",
        "required_or_optional": "optional",
        "rationale": "Legacy binary cue for continuity only.",
        "evidence_source": "v1_continuity",
        "caveats": "Thresholded summary can be brittle.",
    },
    {
        "feature_name": "arm_swing_asymmetry",
        "group": "original_anatomical_biu",
        "body_region": "arms",
        "definition": "Vertical wrist asymmetry normalized by shoulder width.",
        "required_keypoints": "left_wrist;right_wrist;left_shoulder;right_shoulder",
        "normalization": "shoulder_width_normalized",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["original_anatomical_biu"],
        "required_or_optional": "optional",
        "rationale": "Retained for continuity but not a top JAAD signal.",
        "evidence_source": "v1_continuity",
        "caveats": "Sensitive to occlusion and arm pose noise.",
    },
    {
        "feature_name": "norm_shoulder_width",
        "group": "original_anatomical_biu",
        "body_region": "scale",
        "definition": "Shoulder width normalized by nose-to-ankle height proxy.",
        "required_keypoints": "left_shoulder;right_shoulder;nose;left_ankle",
        "normalization": "height_proxy_normalized",
        "direction_aware": "no",
        "summary_stats": "mean;delta",
        "required_or_optional": "optional",
        "rationale": "Keep as legacy scale/viewpoint cue for continuity.",
        "evidence_source": "v1_continuity",
        "caveats": "Not a pure intent cue.",
    },
    {
        "feature_name": "hip_center_x",
        "group": "lower_body_commitment_biu",
        "body_region": "hips",
        "definition": "Canonical hip-center x trajectory over the observation window.",
        "required_keypoints": "left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Primary lower-body commitment cue supported by raw KP and BIU variant audits.",
        "evidence_source": "experiment_006_top_features;experiment_007_family_summary",
        "caveats": "Control bbox fallback only for direction estimation, not feature value.",
    },
    {
        "feature_name": "knee_center_x",
        "group": "lower_body_commitment_biu",
        "body_region": "knees",
        "definition": "Canonical knee-center x trajectory.",
        "required_keypoints": "left_knee;right_knee",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Strongest single-feature family in audits.",
        "evidence_source": "experiment_006_top_features;experiment_007_top_features",
        "caveats": "Requires stable knee detections.",
    },
    {
        "feature_name": "ankle_center_x",
        "group": "lower_body_commitment_biu",
        "body_region": "ankles",
        "definition": "Canonical ankle-center x trajectory.",
        "required_keypoints": "left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Directly captures forward foot commitment.",
        "evidence_source": "experiment_006_top_features;experiment_007_top_features",
        "caveats": "Can be noisy when ankles are occluded.",
    },
    {
        "feature_name": "leg_center_x",
        "group": "lower_body_commitment_biu",
        "body_region": "legs",
        "definition": "Average of hip, knee, and ankle center x positions.",
        "required_keypoints": "left_hip;right_hip;left_knee;right_knee;left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Compact aggregate of lower-body forward commitment.",
        "evidence_source": "experiment_006_top_features;experiment_007_top_features",
        "caveats": "Blends multiple joints; inspect per-joint failures separately.",
    },
    {
        "feature_name": "lead_ankle_forward",
        "group": "lower_body_commitment_biu",
        "body_region": "ankles",
        "definition": "Max canonical x of left/right ankle, interpreted as forward-most ankle after canonicalization.",
        "required_keypoints": "left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Explicit interpretable forward foot commitment signal.",
        "evidence_source": "experiment_007_top_features",
        "caveats": "Depends on correct direction estimate and left/right swapping.",
    },
    {
        "feature_name": "stride_opening",
        "group": "lower_body_commitment_biu",
        "body_region": "ankles",
        "definition": "Distance between ankles over time.",
        "required_keypoints": "left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "indirect",
        "summary_stats": SUMMARY_POLICY["lower_body_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Captures widening step and leg opening.",
        "evidence_source": "experiment_007_report",
        "caveats": "Magnitude-only; pair with directional x features.",
    },
    {
        "feature_name": "lower_body_motion_magnitude",
        "group": "lower_body_commitment_biu",
        "body_region": "legs",
        "definition": "Mean magnitude of frame-to-frame lower-body center motion.",
        "required_keypoints": "left_hip;right_hip;left_knee;right_knee;left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "mean;std",
        "required_or_optional": "required",
        "rationale": "Compact motion energy cue.",
        "evidence_source": "experiment_007_report",
        "caveats": "Can overlap with generic speed effects.",
    },
    {
        "feature_name": "lower_body_motion_consistency",
        "group": "lower_body_commitment_biu",
        "body_region": "legs",
        "definition": "Consistency of lower-body x-motion, computed from stepwise x displacement variability.",
        "required_keypoints": "left_hip;right_hip;left_knee;right_knee;left_ankle;right_ankle",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "mean;std",
        "required_or_optional": "optional",
        "rationale": "Distinguishes steady commitment from jitter.",
        "evidence_source": "experiment_007_design",
        "caveats": "More exploratory than center-x deltas.",
    },
    {
        "feature_name": "torso_center_x",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Canonical torso-center x trajectory from shoulder/hip centers.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["torso_center_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Strong torso commitment cue in revised BIU audit.",
        "evidence_source": "experiment_007_family_summary",
        "caveats": "Close to global center-motion; keep semantically separated from bbox features.",
    },
    {
        "feature_name": "torso_center_y",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Torso-center y trajectory.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["torso_center_commitment_biu"],
        "required_or_optional": "optional",
        "rationale": "Adds vertical body commitment or braking cues.",
        "evidence_source": "experiment_007_design",
        "caveats": "Secondary to x-direction commitment in JAAD.",
    },
    {
        "feature_name": "shoulder_center_x",
        "group": "torso_center_commitment_biu",
        "body_region": "shoulders",
        "definition": "Canonical shoulder-center x trajectory.",
        "required_keypoints": "left_shoulder;right_shoulder",
        "normalization": "canon_first_frame_relative_primary;canon_image_normalized_control",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["torso_center_commitment_biu"],
        "required_or_optional": "required",
        "rationale": "Upper-torso forward commitment companion to hip/torso center.",
        "evidence_source": "experiment_006_body_region_summary;experiment_007",
        "caveats": "Can partly encode viewpoint and arm motion.",
    },
    {
        "feature_name": "shoulder_hip_x_delta_difference",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Difference between shoulder-center x delta and hip-center x delta.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "last;delta",
        "required_or_optional": "optional",
        "rationale": "Measures torso advance relative to pelvis.",
        "evidence_source": "experiment_007_design",
        "caveats": "Interpretability depends on view angle.",
    },
    {
        "feature_name": "shoulder_to_hip_vector",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Vector from hip center to shoulder center, summarized by x and y components.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "torso_centered_optional_control",
        "direction_aware": "yes_canonical_x",
        "summary_stats": "last;delta",
        "required_or_optional": "optional",
        "rationale": "Captures upper-body shift relative to pelvis.",
        "evidence_source": "experiment_007_design",
        "caveats": "More exploratory than center-x features.",
    },
    {
        "feature_name": "torso_lean_vector_x",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "X component of shoulder-center minus hip-center vector.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "torso_centered_optional_control",
        "direction_aware": "yes",
        "summary_stats": "last;delta",
        "required_or_optional": "optional",
        "rationale": "Directional lean proxy beyond raw angle.",
        "evidence_source": "experiment_007_design",
        "caveats": "Can overlap with torso lean angle.",
    },
    {
        "feature_name": "torso_lean_vector_y",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Y component of shoulder-center minus hip-center vector.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "torso_centered_optional_control",
        "direction_aware": "no",
        "summary_stats": "last;delta",
        "required_or_optional": "optional",
        "rationale": "Complements torso lean proxy.",
        "evidence_source": "experiment_007_design",
        "caveats": "Less directly supported than x motion.",
    },
    {
        "feature_name": "torso_motion_magnitude",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Magnitude of frame-to-frame torso-center motion.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "mean;std",
        "required_or_optional": "required",
        "rationale": "Compact torso motion energy cue.",
        "evidence_source": "experiment_007_family_summary",
        "caveats": "May overlap with track motion if used alone.",
    },
    {
        "feature_name": "torso_motion_consistency",
        "group": "torso_center_commitment_biu",
        "body_region": "torso",
        "definition": "Consistency of torso-center stepwise motion.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "mean;std",
        "required_or_optional": "optional",
        "rationale": "Separates smooth commitment from jitter.",
        "evidence_source": "experiment_007_design",
        "caveats": "Exploratory.",
    },
    {
        "feature_name": "head_center_x",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Average x of nose, eyes, and ears.",
        "required_keypoints": "nose;left_eye;right_eye;left_ear;right_ear",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": SUMMARY_POLICY["head_attention_proxy_biu"],
        "required_or_optional": "optional",
        "rationale": "Compact head motion companion, separated from lower-body cues.",
        "evidence_source": "experiment_006_head_summary",
        "caveats": "Less reliable under occlusion.",
    },
    {
        "feature_name": "nose_relative_to_ear_midpoint_x",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Nose x relative to ear midpoint.",
        "required_keypoints": "nose;left_ear;right_ear",
        "normalization": "image_normalized_control",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["head_attention_proxy_biu"],
        "required_or_optional": "optional",
        "rationale": "Interpretable head orientation proxy without invoking car context.",
        "evidence_source": "experiment_007_design",
        "caveats": "Ear detections can fail.",
    },
    {
        "feature_name": "nose_relative_to_ear_midpoint_y",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Nose y relative to ear midpoint.",
        "required_keypoints": "nose;left_ear;right_ear",
        "normalization": "image_normalized_control",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["head_attention_proxy_biu"],
        "required_or_optional": "optional",
        "rationale": "Complements head orientation proxy.",
        "evidence_source": "experiment_007_design",
        "caveats": "Secondary attention cue only.",
    },
    {
        "feature_name": "head_relative_to_torso_x",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Head-center x relative to torso center.",
        "required_keypoints": "nose;left_eye;right_eye;left_ear;right_ear;left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "torso_centered_optional_control",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": SUMMARY_POLICY["head_attention_proxy_biu"],
        "required_or_optional": "optional",
        "rationale": "Separates local head pose from global body movement.",
        "evidence_source": "experiment_007_design",
        "caveats": "Lower priority than lower-body commitment.",
    },
    {
        "feature_name": "head_relative_to_torso_y",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Head-center y relative to torso center.",
        "required_keypoints": "nose;left_eye;right_eye;left_ear;right_ear;left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "torso_centered_optional_control",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["head_attention_proxy_biu"],
        "required_or_optional": "optional",
        "rationale": "Supports head-to-body posture separation.",
        "evidence_source": "experiment_007_design",
        "caveats": "Secondary attention cue.",
    },
    {
        "feature_name": "head_motion_magnitude",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Magnitude of frame-to-frame head-center motion.",
        "required_keypoints": "nose;left_eye;right_eye;left_ear;right_ear",
        "normalization": "canon_first_frame_relative_primary",
        "direction_aware": "yes",
        "summary_stats": "mean;std",
        "required_or_optional": "optional",
        "rationale": "Compact head movement cue, but not primary.",
        "evidence_source": "experiment_006_head_summary",
        "caveats": "Weaker than lower-body commitment features.",
    },
    {
        "feature_name": "head_orientation_delta",
        "group": "head_attention_proxy_biu",
        "body_region": "head",
        "definition": "Delta of head_yaw_angle or face-oriented proxy over the observation window.",
        "required_keypoints": "nose;left_shoulder;right_shoulder",
        "normalization": "intrinsic_angle",
        "direction_aware": "yes_canonical_frame",
        "summary_stats": "delta",
        "required_or_optional": "optional",
        "rationale": "Compact directional attention change cue.",
        "evidence_source": "experiment_007_design",
        "caveats": "Exploratory and view-sensitive.",
    },
    {
        "feature_name": "shoulder_width",
        "group": "scale_viewpoint_metadata",
        "body_region": "scale",
        "definition": "Distance between shoulders.",
        "required_keypoints": "left_shoulder;right_shoulder",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["scale_viewpoint_metadata"],
        "required_or_optional": "required",
        "rationale": "Viewpoint/scale metadata for auditability.",
        "evidence_source": "experiment_007_report",
        "caveats": "Not a primary intent cue.",
    },
    {
        "feature_name": "hip_width",
        "group": "scale_viewpoint_metadata",
        "body_region": "scale",
        "definition": "Distance between hips.",
        "required_keypoints": "left_hip;right_hip",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["scale_viewpoint_metadata"],
        "required_or_optional": "optional",
        "rationale": "Body scale/control metadata.",
        "evidence_source": "experiment_007_design",
        "caveats": "Not a direct intent feature.",
    },
    {
        "feature_name": "torso_length",
        "group": "scale_viewpoint_metadata",
        "body_region": "scale",
        "definition": "Distance from shoulder center to hip center.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["scale_viewpoint_metadata"],
        "required_or_optional": "required",
        "rationale": "Scale/control metadata for robustness.",
        "evidence_source": "experiment_007_design",
        "caveats": "Not a direct intent cue.",
    },
    {
        "feature_name": "skeleton_conf",
        "group": "confidence_metadata",
        "body_region": "all",
        "definition": "Frame-level skeleton confidence aggregated across the observation window.",
        "required_keypoints": "all_detected",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["confidence_metadata"],
        "required_or_optional": "optional",
        "rationale": "Track extraction quality metadata only.",
        "evidence_source": "experiment_006_confidence_columns",
        "caveats": "Not the main next experiment in this spec.",
    },
    {
        "feature_name": "lower_body_conf",
        "group": "confidence_metadata",
        "body_region": "lower_body",
        "definition": "Mean confidence of hips, knees, and ankles across frames.",
        "required_keypoints": "left_hip;right_hip;left_knee;right_knee;left_ankle;right_ankle",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["confidence_metadata"],
        "required_or_optional": "optional",
        "rationale": "Quality metadata for lower-body commitment features.",
        "evidence_source": "experiment_006_confidence_columns",
        "caveats": "Metadata only in BIU v2 baseline.",
    },
    {
        "feature_name": "torso_conf",
        "group": "confidence_metadata",
        "body_region": "torso",
        "definition": "Mean confidence of shoulders and hips.",
        "required_keypoints": "left_shoulder;right_shoulder;left_hip;right_hip",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["confidence_metadata"],
        "required_or_optional": "optional",
        "rationale": "Quality metadata for torso commitment features.",
        "evidence_source": "experiment_006_confidence_columns",
        "caveats": "Metadata only in BIU v2 baseline.",
    },
    {
        "feature_name": "head_conf",
        "group": "confidence_metadata",
        "body_region": "head",
        "definition": "Mean confidence of nose, eyes, and ears.",
        "required_keypoints": "nose;left_eye;right_eye;left_ear;right_ear",
        "normalization": "none_metadata",
        "direction_aware": "no",
        "summary_stats": SUMMARY_POLICY["confidence_metadata"],
        "required_or_optional": "optional",
        "rationale": "Quality metadata for head/attention proxies.",
        "evidence_source": "experiment_006_confidence_columns",
        "caveats": "Metadata only in BIU v2 baseline.",
    },
]


V1_V2_MAPPING: list[dict[str, str]] = [
    {"v1_feature": "torso_lean_angle", "keep_drop_modify": "keep", "v2_replacement_or_extension": "torso_lean_angle", "reason": "Retain for continuity; still interpretable posture cue."},
    {"v1_feature": "stride_length", "keep_drop_modify": "keep", "v2_replacement_or_extension": "stride_length + stride_opening", "reason": "Keep legacy cue and extend with directional opening dynamics."},
    {"v1_feature": "head_yaw_angle", "keep_drop_modify": "keep", "v2_replacement_or_extension": "head_yaw_angle + head_orientation_delta", "reason": "Preserve continuity and add compact temporal change."},
    {"v1_feature": "lead_foot_elevation", "keep_drop_modify": "keep", "v2_replacement_or_extension": "lead_foot_elevation + lead_ankle_forward", "reason": "Legacy vertical cue is retained but forward ankle motion is more predictive."},
    {"v1_feature": "lead_foot_lifted", "keep_drop_modify": "modify", "v2_replacement_or_extension": "lead_foot_lifted (optional) + stride_opening", "reason": "Binary lift alone is weak; pair with motion magnitude or treat as optional."},
    {"v1_feature": "arm_swing_asymmetry", "keep_drop_modify": "keep", "v2_replacement_or_extension": "arm_swing_asymmetry", "reason": "Retain for continuity, but not a primary JAAD signal."},
    {"v1_feature": "gaze_to_car_angle", "keep_drop_modify": "modify", "v2_replacement_or_extension": "gaze_to_car_angle (optional metadata/head proxy)", "reason": "Keep only as separated attention proxy, not core commitment feature."},
    {"v1_feature": "body_yaw_angle", "keep_drop_modify": "keep", "v2_replacement_or_extension": "body_yaw_angle", "reason": "Retain as original anatomical BIU."},
    {"v1_feature": "face_yaw_angle", "keep_drop_modify": "keep", "v2_replacement_or_extension": "face_yaw_angle", "reason": "Retain but mark optional due to head-view fragility."},
    {"v1_feature": "norm_shoulder_width", "keep_drop_modify": "modify", "v2_replacement_or_extension": "norm_shoulder_width + shoulder_width metadata", "reason": "Demote from intent cue to scale/viewpoint metadata."},
    {"v1_feature": "none", "keep_drop_modify": "add", "v2_replacement_or_extension": "hip_center_x / knee_center_x / ankle_center_x / leg_center_x", "reason": "Direction-aware lower-body commitment features are the main new BIU v2 block."},
    {"v1_feature": "none", "keep_drop_modify": "add", "v2_replacement_or_extension": "torso_center_x / shoulder_center_x / torso_motion_magnitude", "reason": "Compact torso/body-center commitment features are strongly supported by the audits."},
    {"v1_feature": "none", "keep_drop_modify": "add", "v2_replacement_or_extension": "direction_sign / direction_source / was_flipped / direction_confidence", "reason": "Direction handling is required to prevent x-motion cancellation."},
    {"v1_feature": "none", "keep_drop_modify": "add", "v2_replacement_or_extension": "confidence metadata fields", "reason": "Keep extraction-quality metadata separate from the main BIU signal."},
]


@dataclass
class DirectionMetadata:
    direction_sign: int
    direction_source: str
    was_flipped: bool
    direction_confidence: float


def _to_xy(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=float)[:2]


def _pair_center(coords: np.ndarray, left: str, right: str) -> np.ndarray:
    li, ri = KPT_INDEX[left], KPT_INDEX[right]
    return np.nanmean(coords[:, [li, ri], :], axis=1)


def _safe_norm(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.linalg.norm(arr, axis=axis)


def _frame_joints(frame_coords: np.ndarray) -> dict[str, np.ndarray]:
    return {name: frame_coords[idx, :].astype(float) for name, idx in KPT_INDEX.items()}


def estimate_direction(
    keypoints_xy: np.ndarray,
    bbox_sequence: np.ndarray | None = None,
) -> DirectionMetadata:
    hip_center = _pair_center(keypoints_xy, "left_hip", "right_hip")
    ankle_center = _pair_center(keypoints_xy, "left_ankle", "right_ankle")
    hip_dx = float(hip_center[-1, 0] - hip_center[0, 0])
    ankle_dx = float(ankle_center[-1, 0] - ankle_center[0, 0])
    shoulder_width = _safe_norm(
        keypoints_xy[:, KPT_INDEX["left_shoulder"], :] - keypoints_xy[:, KPT_INDEX["right_shoulder"], :]
    )
    norm_scale = float(np.nanmean(shoulder_width)) if np.isfinite(shoulder_width).any() else 1.0

    if abs(hip_dx) >= 1.0:
        chosen_dx = hip_dx
        source = "hip_center_x"
    elif abs(ankle_dx) >= 1.0:
        chosen_dx = ankle_dx
        source = "ankle_center_x"
    elif bbox_sequence is not None and len(bbox_sequence):
        bbox_centers = (bbox_sequence[:, 0] + bbox_sequence[:, 2]) / 2.0
        chosen_dx = float(bbox_centers[-1] - bbox_centers[0])
        source = "bbox_center_x"
    else:
        chosen_dx = hip_dx
        source = "low_motion_hip_center_x"

    sign = 1 if chosen_dx >= 0 else -1
    confidence = min(1.0, abs(chosen_dx) / max(norm_scale, 1e-6))
    return DirectionMetadata(
        direction_sign=sign,
        direction_source=source,
        was_flipped=bool(sign < 0),
        direction_confidence=float(confidence),
    )


def canonicalize_keypoints(
    keypoints_xy: np.ndarray,
    direction: DirectionMetadata,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    coords = np.asarray(keypoints_xy, dtype=float).copy()
    if not direction.was_flipped:
        return coords

    if image_size is not None:
        width = float(image_size[0])
        coords[:, :, 0] = width - coords[:, :, 0]
    else:
        coords[:, :, 0] = -coords[:, :, 0]

    for left, right in LEFT_RIGHT_PAIRS:
        li, ri = KPT_INDEX[left], KPT_INDEX[right]
        coords[:, [li, ri], :] = coords[:, [ri, li], :]
    return coords


def _summary(values: np.ndarray, stats: list[str]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    out: dict[str, float] = {}
    if "first" in stats:
        out["first"] = float(arr[0])
    if "last" in stats:
        out["last"] = float(arr[-1])
    if "mean" in stats:
        out["mean"] = float(np.nanmean(arr))
    if "std" in stats:
        out["std"] = float(np.nanstd(arr))
    if "min" in stats:
        out["min"] = float(np.nanmin(arr))
    if "max" in stats:
        out["max"] = float(np.nanmax(arr))
    if "delta" in stats:
        out["delta"] = float(arr[-1] - arr[0])
    if "velocity_mean" in stats or "velocity_std" in stats:
        vel = np.diff(arr, prepend=arr[0])
        if "velocity_mean" in stats:
            out["velocity_mean"] = float(np.nanmean(vel))
        if "velocity_std" in stats:
            out["velocity_std"] = float(np.nanstd(vel))
    return out


def _apply_summary(feature_name: str, values: np.ndarray, stats_spec: str) -> dict[str, float]:
    stats = [s for s in stats_spec.split(";") if s]
    return {f"{feature_name}_{k}": v for k, v in _summary(values, stats).items()}


def feature_dictionary_df() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_DICTIONARY)


def v1_v2_mapping_df() -> pd.DataFrame:
    return pd.DataFrame(V1_V2_MAPPING)


def extract_biu_v2(
    keypoints_xy: np.ndarray,
    kp_conf: np.ndarray | None = None,
    bbox_sequence: np.ndarray | None = None,
    image_size: tuple[int, int] = (1920, 1080),
    include_optional: bool = False,
) -> dict[str, float | int | str]:
    coords = np.asarray(keypoints_xy, dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2):
        raise ValueError(f"Expected keypoints shape [T,17,2], got {coords.shape}")
    conf = np.asarray(kp_conf, dtype=float) if kp_conf is not None else None
    bbox_arr = np.asarray(bbox_sequence, dtype=float) if bbox_sequence is not None else None

    direction = estimate_direction(coords, bbox_arr)
    canon = canonicalize_keypoints(coords, direction, image_size=image_size)
    canon_img = canon.copy()
    canon_img[:, :, 0] /= float(image_size[0])
    canon_img[:, :, 1] /= float(image_size[1])
    canon_firstrel = canon_img.copy()
    canon_firstrel[:, :, 0] -= canon_firstrel[0:1, :, 0]
    canon_firstrel[:, :, 1] -= canon_firstrel[0:1, :, 1]

    out: dict[str, float | int | str] = {
        "direction_sign": direction.direction_sign,
        "direction_source": direction.direction_source,
        "was_flipped": int(direction.was_flipped),
        "direction_confidence": direction.direction_confidence,
    }

    # Original anatomical BIUs from canonicalized absolute coordinates.
    per_frame_biu = {name: [] for name in [
        "torso_lean_angle",
        "body_yaw_angle",
        "head_yaw_angle",
        "face_yaw_angle",
        "gaze_to_car_angle",
        "stride_length",
        "lead_foot_elevation",
        "lead_foot_lifted",
        "arm_swing_asymmetry",
        "norm_shoulder_width",
    ]}
    for t in range(canon.shape[0]):
        feats = extract_biu_features(_frame_joints(canon[t]))
        for key in per_frame_biu:
            per_frame_biu[key].append(float(feats[key]))

    include_names = {row["feature_name"] for row in FEATURE_DICTIONARY if include_optional or row["required_or_optional"] == "required"}
    spec_map = {row["feature_name"]: row for row in FEATURE_DICTIONARY}

    for name, vals in per_frame_biu.items():
        if name in include_names:
            out.update(_apply_summary(name, np.asarray(vals, dtype=float), spec_map[name]["summary_stats"]))

    hip_center = np.nanmean(canon_firstrel[:, [KPT_INDEX["left_hip"], KPT_INDEX["right_hip"]], :], axis=1)
    knee_center = np.nanmean(canon_firstrel[:, [KPT_INDEX["left_knee"], KPT_INDEX["right_knee"]], :], axis=1)
    ankle_center = np.nanmean(canon_firstrel[:, [KPT_INDEX["left_ankle"], KPT_INDEX["right_ankle"]], :], axis=1)
    shoulder_center = np.nanmean(canon_firstrel[:, [KPT_INDEX["left_shoulder"], KPT_INDEX["right_shoulder"]], :], axis=1)
    torso_center = np.nanmean(np.stack([shoulder_center, hip_center], axis=1), axis=1)
    leg_center = np.nanmean(np.stack([hip_center, knee_center, ankle_center], axis=1), axis=1)
    head_center = np.nanmean(canon_firstrel[:, [KPT_INDEX["nose"], KPT_INDEX["left_eye"], KPT_INDEX["right_eye"], KPT_INDEX["left_ear"], KPT_INDEX["right_ear"]], :], axis=1)

    lower_body_series = {
        "hip_center_x": hip_center[:, 0],
        "knee_center_x": knee_center[:, 0],
        "ankle_center_x": ankle_center[:, 0],
        "leg_center_x": leg_center[:, 0],
        "lead_ankle_forward": np.maximum(canon_firstrel[:, KPT_INDEX["left_ankle"], 0], canon_firstrel[:, KPT_INDEX["right_ankle"], 0]),
        "stride_opening": _safe_norm(canon_firstrel[:, KPT_INDEX["left_ankle"], :] - canon_firstrel[:, KPT_INDEX["right_ankle"], :]),
        "lower_body_motion_magnitude": np.sqrt(np.square(np.diff(leg_center[:, 0], prepend=leg_center[0, 0])) + np.square(np.diff(leg_center[:, 1], prepend=leg_center[0, 1]))),
        "lower_body_motion_consistency": np.abs(np.diff(ankle_center[:, 0], prepend=ankle_center[0, 0])),
    }
    for name, vals in lower_body_series.items():
        if name in include_names:
            out.update(_apply_summary(name, vals, spec_map[name]["summary_stats"]))

    torso_vector = shoulder_center - hip_center
    torso_series = {
        "torso_center_x": torso_center[:, 0],
        "torso_center_y": torso_center[:, 1],
        "shoulder_center_x": shoulder_center[:, 0],
        "shoulder_hip_x_delta_difference": shoulder_center[:, 0] - hip_center[:, 0],
        "shoulder_to_hip_vector": np.sqrt(np.square(torso_vector[:, 0]) + np.square(torso_vector[:, 1])),
        "torso_lean_vector_x": torso_vector[:, 0],
        "torso_lean_vector_y": torso_vector[:, 1],
        "torso_motion_magnitude": np.sqrt(np.square(np.diff(torso_center[:, 0], prepend=torso_center[0, 0])) + np.square(np.diff(torso_center[:, 1], prepend=torso_center[0, 1]))),
        "torso_motion_consistency": np.abs(np.diff(torso_center[:, 0], prepend=torso_center[0, 0])),
    }
    for name, vals in torso_series.items():
        if name in include_names:
            out.update(_apply_summary(name, vals, spec_map[name]["summary_stats"]))

    ear_mid = np.nanmean(canon_firstrel[:, [KPT_INDEX["left_ear"], KPT_INDEX["right_ear"]], :], axis=1)
    nose = canon_firstrel[:, KPT_INDEX["nose"], :]
    head_series = {
        "head_center_x": head_center[:, 0],
        "nose_relative_to_ear_midpoint_x": nose[:, 0] - ear_mid[:, 0],
        "nose_relative_to_ear_midpoint_y": nose[:, 1] - ear_mid[:, 1],
        "head_relative_to_torso_x": head_center[:, 0] - torso_center[:, 0],
        "head_relative_to_torso_y": head_center[:, 1] - torso_center[:, 1],
        "head_motion_magnitude": np.sqrt(np.square(np.diff(head_center[:, 0], prepend=head_center[0, 0])) + np.square(np.diff(head_center[:, 1], prepend=head_center[0, 1]))),
        "head_orientation_delta": np.asarray(per_frame_biu["head_yaw_angle"], dtype=float),
    }
    for name, vals in head_series.items():
        if name in include_names:
            out.update(_apply_summary(name, vals, spec_map[name]["summary_stats"]))

    raw_shoulders = np.nanmean(canon[:, [KPT_INDEX["left_shoulder"], KPT_INDEX["right_shoulder"]], :], axis=1)
    raw_hips = np.nanmean(canon[:, [KPT_INDEX["left_hip"], KPT_INDEX["right_hip"]], :], axis=1)
    scale_series = {
        "shoulder_width": _safe_norm(canon[:, KPT_INDEX["left_shoulder"], :] - canon[:, KPT_INDEX["right_shoulder"], :]),
        "hip_width": _safe_norm(canon[:, KPT_INDEX["left_hip"], :] - canon[:, KPT_INDEX["right_hip"], :]),
        "torso_length": _safe_norm(raw_shoulders - raw_hips),
    }
    for name, vals in scale_series.items():
        if name in include_names:
            out.update(_apply_summary(name, vals, spec_map[name]["summary_stats"]))

    if conf is not None:
        lower_idx = [KPT_INDEX[k] for k in ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]]
        torso_idx = [KPT_INDEX[k] for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]]
        head_idx = [KPT_INDEX[k] for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]]
        conf_series = {
            "skeleton_conf": np.nanmean(conf, axis=1),
            "lower_body_conf": np.nanmean(conf[:, lower_idx], axis=1),
            "torso_conf": np.nanmean(conf[:, torso_idx], axis=1),
            "head_conf": np.nanmean(conf[:, head_idx], axis=1),
        }
        for name, vals in conf_series.items():
            if name in include_names:
                out.update(_apply_summary(name, vals, spec_map[name]["summary_stats"]))

    return out
