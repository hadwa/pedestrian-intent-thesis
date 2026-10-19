# import numpy as np
# from pose_utils import torso_lean_angle, stride_length, head_yaw_angle, lead_foot_elevation, arm_swing_asymmetry, gaze_to_car, angle_between, midpoint, dist

# def extract_biu_features(joints):
    

#     """
#     Given a dict of named keypoints (from ViTPose), extract interpretable BIU features.
#     Returns a dict of boolean or numeric cues.
#     """
       
#     features = {}

#     # 1. Torso lean > 10°
#     angle_torso = torso_lean_angle(
#         joints["left_shoulder"], joints["right_shoulder"],
#         joints["left_hip"], joints["right_hip"]
#     )
#     features["torso_lean_gt10"] = angle_torso > 10
#     features["torso_lean_angle"] = angle_torso

#     # 2. Stride length > 40 px
#     stride = stride_length(joints["left_ankle"], joints["right_ankle"])
#     features["stride_length_gt40"] = stride > 40
#     features["stride_length"] = stride

#     # 3. Head yaw > 15°
#     angle_yaw = head_yaw_angle(
#         joints["nose"], joints["left_shoulder"], joints["right_shoulder"]
#     )
#     features["head_yaw_gt15"] = angle_yaw > 15
#     features["head_yaw_angle"] = angle_yaw

#     # --- BIU 4: Lead Foot Elevation ---
#     foot_elev, foot_lifted = lead_foot_elevation(
#         joints["left_ankle"],
#         joints["right_ankle"],
#         joints["left_hip"],
#         joints["right_hip"]
#     )
#     features["lead_foot_lifted"] = foot_lifted
#     features["lead_foot_elevation"] = foot_elev

#         # --- BIU 5: Arm Swing Asymmetry ---
#     swing_val, swing_asym = arm_swing_asymmetry(
#         joints["left_wrist"],
#         joints["right_wrist"],
#         joints["left_shoulder"],
#         joints["right_shoulder"]
#     )
#     features["arm_swing_asymmetry"] = swing_val
#     features["arm_swing_asym"] = swing_asym

#     # --- BIU 6: Gaze Toward Car ---
#     yaw_angle, gaze_toward_car = gaze_to_car(
#         joints["nose"],
#         joints["left_ear"],
#         joints["right_ear"]
#     )

    
#     features["gaze_to_car_angle"] = yaw_angle
#     features["gaze_toward_car"] = gaze_toward_car

#     #  3 interpretable BIU Scalars isnpired by the paper:
#     # "VRU-CIPI: Crossing Intention Prediction at Intersections for Improving Vulnerable Road Users Safety"

#     # ---- BIU 7: Body yaw (shoulder line) ----
#     body_yaw = angle_between(joints["left_shoulder"], joints["right_shoulder"])
#     features["body_yaw_angle"] = body_yaw

#     # ---- BIU 8: Face yaw (nose→hip-mid) ----
#     hip_mid = midpoint(joints["left_hip"], joints["right_hip"])
#     face_yaw = angle_between(joints["nose"], hip_mid)
#     features["face_yaw_angle"] = face_yaw

#     # ---- BIU 9: Normalised shoulder width ----
#     shoulder_px = dist(joints["left_shoulder"], joints["right_shoulder"])
#     height_px   = dist(joints["nose"], joints["left_ankle"])          # scale
#     features["norm_shoulder_width"] = shoulder_px / (height_px + 1e-6)
    
#     return features
# # Order of features in the output vector (for consistency)  
    
# ORDER = [
#         "torso_lean_angle",
#         "stride_length",
#         "head_yaw_angle",
#         "lead_foot_elevation","lead_foot_lifted",
#         "arm_swing_asymmetry","arm_swing_asym","gaze_to_car_angle",
#         "body_yaw_angle","face_yaw_angle","norm_shoulder_width"
#     ]    

# dt = 5/30.0  # stride/fps
# def deriv(x):
#     v = np.empty_like(x)
#     v[1:-1] = (x[2:] - x[:-2])/(2*dt)
#     v[0]  = (x[1]-x[0])/dt
#     v[-1] = (x[-1]-x[-2])/dt
#     return v

# body_yaw_rate   = deriv(body_yaw_angle)
# face_yaw_rate   = deriv(face_yaw_angle)
# lead_foot_lift_speed = deriv(lead_foot_elevation)
# ankle_L_vy, ankle_R_vy = deriv(ankle_L_y), deriv(ankle_R_y)
# knee_L_vy,  knee_R_vy  = deriv(knee_L_y),  deriv(knee_R_y)
# gait_step_freq = bandpower(signal=pelvis_y_or_bbox_h, fs=30/5, fmin=1.0, fmax=3.0)
# scan_bouts     = count_zero_crossings(face_yaw_angle, min_sep=3)  # avoid jitter
# # optionally: multiply the velocity/rate features by w_pose (from kp visibility)


# # def extract_biu_feat_in_order(joints):
# #     feats = extract_biu_features(joints)
# #     return {k: feats[k] for k in ORDER}

# def extract_biu_vector(joints):
#     feats = extract_biu_features(joints)
#     return {k: feats[k] for k in ORDER}
#     # return np.array([float(feats[k]) for k in ORDER], dtype=np.float32)



# import numpy as np
# from pose_utils import torso_lean_angle, stride_length, head_yaw_angle, lead_foot_elevation, arm_swing_asymmetry, gaze_to_car, angle_between, midpoint, dist

# def extract_biu_features(joints):
    

#     """
#     Given a dict of named keypoints (from ViTPose), extract interpretable BIU features.
#     Returns a dict of boolean or numeric cues.
#     """
       
#     features = {}

#     # 1. Torso lean > 10°
#     angle_torso = torso_lean_angle(
#         joints["left_shoulder"], joints["right_shoulder"],
#         joints["left_hip"], joints["right_hip"]
#     )
#     ## booleanfeatures were redundant
#     # features["torso_lean_gt10"] = angle_torso > 10
#     features["torso_lean_angle"] = angle_torso

#     # 2. Stride length > 40 px
#     stride = stride_length(joints["left_ankle"], joints["right_ankle"])
#     ## booleanfeatures were redundant
#     # features["stride_length_gt40"] = stride > 40
#     features["stride_length"] = stride

#     # 3. Head yaw > 15°
#     angle_yaw = head_yaw_angle(
#         joints["nose"], joints["left_shoulder"], joints["right_shoulder"]
#     )
#     ## booleanfeatures were redundant
#     # features["head_yaw_gt15"] = angle_yaw > 15
#     features["head_yaw_angle"] = angle_yaw

#     # --- BIU 4: Lead Foot Elevation ---
#     foot_elev, foot_lifted = lead_foot_elevation(
#         joints["left_ankle"],
#         joints["right_ankle"],
#         joints["left_hip"],
#         joints["right_hip"]
#     )
#     features["lead_foot_lifted"] = foot_lifted
#     features["lead_foot_elevation"] = foot_elev

#         # --- BIU 5: Arm Swing Asymmetry ---
#     swing_val, swing_asym = arm_swing_asymmetry(
#         joints["left_wrist"],
#         joints["right_wrist"],
#         joints["left_shoulder"],
#         joints["right_shoulder"]
#     )
#     features["arm_swing_asymmetry"] = swing_val
#     ## booleanfeatures were redundant
#     # features["arm_swing_asym"] = swing_asym

#     # --- BIU 6: Gaze Toward Car ---
#     yaw_angle, gaze_toward_car = gaze_to_car(
#         joints["nose"],
#         joints["left_ear"],
#         joints["right_ear"]
#     )

    
#     features["gaze_to_car_angle"] = yaw_angle
#     ## booleanfeatures were redundant
#     # features["gaze_toward_car"] = gaze_toward_car

#     #  3 interpretable BIU Scalars isnpired by the paper:
#     # "VRU-CIPI: Crossing Intention Prediction at Intersections for Improving Vulnerable Road Users Safety"

#     # ---- BIU 7: Body yaw (shoulder line) ----
#     body_yaw = angle_between(joints["left_shoulder"], joints["right_shoulder"])
#     features["body_yaw_angle"] = body_yaw

#     # ---- BIU 8: Face yaw (nose→hip-mid) ----
#     hip_mid = midpoint(joints["left_hip"], joints["right_hip"])
#     face_yaw = angle_between(joints["nose"], hip_mid)
#     features["face_yaw_angle"] = face_yaw

#     # ---- BIU 9: Normalised shoulder width ----
#     shoulder_px = dist(joints["left_shoulder"], joints["right_shoulder"])
#     height_px   = dist(joints["nose"], joints["left_ankle"])          # scale
#     features["norm_shoulder_width"] = shoulder_px / (height_px + 1e-6)
    
#     return features

# ORDER = [
#     "torso_lean_angle",
#     "stride_length",
#     "head_yaw_angle",
#     "lead_foot_elevation","lead_foot_lifted",
#     "arm_swing_asymmetry",
#     "gaze_to_car_angle",
#     "body_yaw_angle","face_yaw_angle","norm_shoulder_width"
# ]                               # 10-dim

# def extract_biu_vector(joints):
#     feats = extract_biu_features(joints)
#     return np.array([float(feats[k]) for k in ORDER], dtype=np.float32)



    


import numpy as np
try:
    from .pose_utils import (
        torso_lean_angle, stride_length, head_yaw_angle, lead_foot_elevation,
        arm_swing_asymmetry, gaze_to_car, angle_between, midpoint, dist
    )
    from .motion_utils import add_temporal_features, verify_temporal_hygiene
except ImportError:  # Supports direct execution from the repository's src directory.
    from pose_utils import (
        torso_lean_angle, stride_length, head_yaw_angle, lead_foot_elevation,
        arm_swing_asymmetry, gaze_to_car, angle_between, midpoint, dist
    )
    from motion_utils import add_temporal_features, verify_temporal_hygiene

def extract_biu_features(joints):
    """
    Extract interpretable BIU features.
    NOTE:
      - All angles are now **radians** (not degrees).
      - lead_foot_elevation is **positive-up**.
      - stride_length and lead_foot_elevation are normalized by shoulder width.
    """
    features = {}

    # --- scale proxies ---
    left_shoulder, right_shoulder = joints["left_shoulder"], joints["right_shoulder"]
    shoulder_px = dist(left_shoulder, right_shoulder)  # normalization scale
    height_px   = dist(joints["nose"], joints["left_ankle"])

    # 1) Torso lean (radians, signed, |θ| ≤ π/2)
    angle_torso = torso_lean_angle(
        left_shoulder, right_shoulder,
        joints["left_hip"], joints["right_hip"]
    )
    features["torso_lean_angle"] = angle_torso

    # 2) Stride length (normalized by shoulder width)
    stride = stride_length(joints["left_ankle"], joints["right_ankle"], norm_scale=shoulder_px)
    features["stride_length"] = stride

    # 3) Head yaw (radians, clockwise positive)
    angle_yaw = head_yaw_angle(
        joints["nose"], left_shoulder, right_shoulder
    )
    features["head_yaw_angle"] = angle_yaw

    # 4) Lead foot elevation (normalized, positive-up) + boolean
    # 4) Lead foot elevation (normalized, non-negative) + boolean as 0/1
    foot_elev, foot_lifted = lead_foot_elevation(
        joints["left_ankle"], joints["right_ankle"],
        joints["left_hip"], joints["right_hip"],
        norm_scale=shoulder_px
    )
    features["lead_foot_elevation"] = foot_elev
    features["lead_foot_lifted"]    = int(foot_lifted)  # <- cast to 0/1

    # 5) Arm swing asymmetry (ratio by shoulder width)
    swing_val, _ = arm_swing_asymmetry(
        joints["left_wrist"], joints["right_wrist"],
        left_shoulder, right_shoulder
    )
    features["arm_swing_asymmetry"] = swing_val

    # 6) Gaze angle (radians) vs up (or toward car if car_vec provided)
    yaw_angle, _ = gaze_to_car(
        joints["nose"], joints["left_ear"], joints["right_ear"]
    )
    features["gaze_to_car_angle"] = yaw_angle

    # 7) Body yaw (shoulder line vs x-axis, radians, signed)
    shoulder_vec = np.asarray(right_shoulder[:2], float) - np.asarray(left_shoulder[:2], float)
    body_yaw = angle_between(shoulder_vec, np.array([1.0, 0.0]), signed=True)  # vs horizontal
    features["body_yaw_angle"] = body_yaw

    # 8) Face yaw (nose vector vs vertical up, radians, signed)
    hip_mid = midpoint(joints["left_hip"], joints["right_hip"])
    face_vec = np.asarray(joints["nose"][:2], float) - hip_mid
    face_yaw = angle_between(face_vec, np.array([0.0, -1.0]), signed=True)  # vs up
    features["face_yaw_angle"] = face_yaw

    # 9) Normalized shoulder width (relative to height proxy)
    features["norm_shoulder_width"] = shoulder_px / (height_px + 1e-6)

    return features

ORDER = [
    "torso_lean_angle",
    "stride_length",
    "head_yaw_angle",
    "lead_foot_elevation", "lead_foot_lifted",
    "arm_swing_asymmetry",
    "gaze_to_car_angle",
    "body_yaw_angle", "face_yaw_angle", "norm_shoulder_width"
]

def extract_biu_vector(joints):
    feats = extract_biu_features(joints)
    # bool -> float converts to 0.0/1.0 automatically
    return np.array([float(feats[k]) for k in ORDER], dtype=np.float32)
