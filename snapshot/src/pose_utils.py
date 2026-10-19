# import numpy as np


# #Calculates the angle (in degrees) between two vectors
# def angle_between(v1, v2):
#     """
#     Compute angle in degrees between two vectors v1 and v2.
#     """
#     v1_u = v1 / np.linalg.norm(v1)
#     v2_u = v2 / np.linalg.norm(v2)
#     dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
#     return np.degrees(np.arccos(dot_product))

# def dist(p1, p2):
#     """Euclidean distance between two key-points [x,y] or [x,y,c]."""
#     return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

# def midpoint(p1, p2):
#     """Mid-point coordinates between two key-points."""
#     return (np.array(p1[:2]) + np.array(p2[:2])) / 2.0


# # Measures the angle between the torso vector and vertical
# def torso_lean_angle(left_shoulder, right_shoulder, left_hip, right_hip):
#     """
#     Compute the torso lean angle (in degrees) relative to vertical, always ≤ 90°.
#     """
#     shoulder_mid = (left_shoulder + right_shoulder) / 2
#     hip_mid = (left_hip + right_hip) / 2
#     torso_vec = hip_mid - shoulder_mid
#     vertical_vec = np.array([0, 1])  # vertical down (image space)
#     angle = angle_between(torso_vec, vertical_vec)
#     return min(angle, 180 - angle)  # always acute


# #	Measures distance between left and right ankles
# def stride_length(left_ankle, right_ankle):
#     """
#     Compute the stride length as the distance between both ankles.
#     """
#     return np.linalg.norm(left_ankle - right_ankle)

# # Measures head orientation vs. forward direction
# def head_yaw_angle(nose, left_shoulder, right_shoulder):
#     """
#     Estimate the yaw of the head based on the vector from the neck (avg. shoulders) to the nose.
#     """
#     neck = (left_shoulder + right_shoulder) / 2
#     head_vec = nose - neck
#     forward_vec = np.array([0, -1])  # vertical axis (up)
#     return angle_between(head_vec, forward_vec)

# # Measures how far the lead ankle is raised relative to the ground (Y-axis).
# def lead_foot_elevation(left_ankle, right_ankle, left_hip, right_hip, threshold=5.0):
#     """
#     Determines if either foot is lifted above a reference baseline (hip level - heuristic).
#     Returns: elevation value (max of the two feet) and a boolean if it exceeds threshold (in pixels).
#     """
#     # Compute average hip y-position as ground reference
#     hip_y_baseline = (left_hip[1] + right_hip[1]) / 2

#     # Compute elevation of each foot (how far above baseline)
#     left_elev = hip_y_baseline - left_ankle[1]
#     right_elev = hip_y_baseline - right_ankle[1]

#     # Take the greater elevation (lead foot)
#     max_elevation = max(left_elev, right_elev)

#     # Return elevation and binary indicator
#     return max_elevation, max_elevation > threshold

# # Is pedestrian walking vs. standing/texting
# def arm_swing_asymmetry(left_wrist, right_wrist, left_shoulder, right_shoulder, threshold=0.1):
#     """
#     Measure vertical asymmetry in wrist positions.
#     Returns: difference in y-coordinates and a boolean if exceeds threshold.
#     """
#     height_baseline = (left_shoulder[1] + right_shoulder[1]) / 2
#     norm_factor = height_baseline * threshold  # normalize for body height

#     delta_y = abs(left_wrist[1] - right_wrist[1])
#     return delta_y, delta_y > norm_factor

# #Is head turned toward vehicle

# def gaze_to_car(nose, left_ear, right_ear, threshold_deg=15):
#     """
#     Estimate head yaw based on ear-to-nose vector.
#     Returns yaw angle (in degrees) and boolean if head is facing left/right past threshold.
#     """
#     head_vec = right_ear - left_ear
#     nose_vec = nose - (left_ear + right_ear) / 2  # from head center to nose

#     angle = angle_between(nose_vec, head_vec)
#     return angle, angle > threshold_deg


import numpy as np

_EPS = 1e-9

# ----------------- helpers -----------------

def _to_xy(p):
    a = np.asarray(p, dtype=float)
    return a[:2]

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > _EPS else np.zeros_like(v)

def dist(p1, p2):
    """Euclidean distance between two key-points [x,y] or [x,y,c]."""
    return np.linalg.norm(_to_xy(p1) - _to_xy(p2))

def midpoint(p1, p2):
    """Mid-point coordinates between two key-points."""
    return (_to_xy(p1) + _to_xy(p2)) / 2.0

# ----------------- angles -----------------

def angle_between(v1, v2, *, signed=False, return_degrees=False):
    """
    Angle between two vectors (default: radians).
    - signed=False => [0, pi]
    - signed=True  => [-pi, pi] (positive = clockwise if image y grows downward)
    """
    v1u, v2u = _unit(v1), _unit(v2)
    if signed:
        # z-component of 2D cross (v2 -> v1)
        cross_z = v2u[0] * v1u[1] - v2u[1] * v1u[0]
        dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
        ang = np.arctan2(cross_z, dot)
    else:
        dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
        ang = np.arccos(dot)
    return np.degrees(ang) if return_degrees else ang

# Measures the angle between the torso vector and vertical
def torso_lean_angle(
    left_shoulder, right_shoulder, left_hip, right_hip, *,
    return_degrees=False, signed=True
):
    """
    Torso lean relative to vertical (image-space y+ downward).
    Returns radians by default; magnitude is folded to <= 90°.
    If signed=True, sign indicates left/right lean (clockwise positive).
    """
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    torso_vec = hip_mid - shoulder_mid
    vertical_vec = np.array([0.0, 1.0])  # vertical down in image space

    theta = angle_between(torso_vec, vertical_vec, signed=True, return_degrees=False)  # [-pi, pi]
    # fold magnitude into [0, pi/2], preserve sign if requested
    a = abs(theta)
    if a > np.pi / 2:
        a = np.pi - a
        theta = np.sign(theta) * a if signed else a
    else:
        theta = theta if signed else a

    return np.degrees(theta) if return_degrees else theta

#	Measures distance between left and right ankles
def stride_length(left_ankle, right_ankle, *, norm_scale=None):
    """
    Distance between ankles.
    If norm_scale is provided (e.g., shoulder width or bbox height), returns normalized value.
    """
    d = dist(left_ankle, right_ankle)
    return d / norm_scale if (norm_scale is not None and norm_scale > 0) else d

# Measures head orientation vs. forward direction
def head_yaw_angle(nose, left_shoulder, right_shoulder, *, return_degrees=False, signed=True):
    """
    Head yaw using vector from neck (mid-shoulders) to nose vs vertical 'up' (0,-1).
    Returns radians by default; signed yaw (clockwise positive).
    """
    neck = midpoint(left_shoulder, right_shoulder)
    head_vec = _to_xy(nose) - neck
    forward_vec = np.array([0.0, -1.0])  # vertical up
    ang = angle_between(head_vec, forward_vec, signed=True, return_degrees=False)
    return np.degrees(ang) if return_degrees else ang

# Measures how far the lead ankle is raised relative to baseline (positive-up).
def lead_foot_elevation(
    left_ankle, right_ankle, left_hip, right_hip, *,
    norm_scale=None, threshold=None
):
    """
    Elevation proxy for the 'lead' foot using vertical ankle separation.
    - Positive by construction (>= 0).
    - Units: pixels, or normalized by 'norm_scale' (e.g., shoulder width) if provided.
    - is_lifted = elevation > threshold (default: 5 px or 0.05*norm_scale).
    """
    ya_l = float(left_ankle[1])
    ya_r = float(right_ankle[1])

    # image y increases downward → "higher" means smaller y
    elev_px = abs(ya_l - ya_r)  # vertical separation between ankles

    if norm_scale is not None and norm_scale > 0:
        elev = elev_px / norm_scale
        default_thr = 0.05
    else:
        elev = elev_px
        default_thr = 5.0

    thr = default_thr if threshold is None else float(threshold)
    return elev, (elev > thr)


# Is pedestrian walking vs. standing/texting
def arm_swing_asymmetry(
    left_wrist, right_wrist, left_shoulder, right_shoulder, *, threshold=0.15, return_ratio=True
):
    """
    Vertical asymmetry |y_L - y_R|, normalized by shoulder width (default).
    Returns (value, exceeds_threshold). If return_ratio=False, value is pixels and threshold is pixels.
    """
    delta_y = abs(_to_xy(left_wrist)[1] - _to_xy(right_wrist)[1])
    if return_ratio:
        sw = dist(left_shoulder, right_shoulder)
        val = delta_y / sw if sw > _EPS else np.nan
        exceeds = (val > threshold) if not np.isnan(val) else False
        return val, exceeds
    else:
        return delta_y, (delta_y > threshold)

# Is head turned toward vehicle
def gaze_to_car(
    nose, left_ear, right_ear, *, car_vec=None, threshold=np.deg2rad(15), return_degrees=False, signed=True
):
    """
    Angle between head-forward (mid-ears -> nose) and car_vec (if provided) or vertical up (0,-1).
    Returns (angle, flag). Angle is radians by default.
    - If car_vec provided: flag = looking toward car (abs(angle) <= threshold).
    - Else: flag = turned beyond threshold (abs(angle) > threshold).
    """
    head_center = midpoint(left_ear, right_ear)
    head_fwd = _to_xy(nose) - head_center
    if np.linalg.norm(head_fwd) <= _EPS:
        ang = np.nan
        return (np.nan, False)

    if car_vec is not None:
        ang = angle_between(head_fwd, car_vec, signed=signed, return_degrees=False)
        flag = (abs(ang) <= threshold)
    else:
        ref_up = np.array([0.0, -1.0])
        ang = angle_between(head_fwd, ref_up, signed=signed, return_degrees=False)
        flag = (abs(ang) > threshold)

    return (np.degrees(ang), flag) if return_degrees else (ang, flag)

