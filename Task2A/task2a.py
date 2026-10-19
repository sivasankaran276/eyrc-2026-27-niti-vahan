'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2A of Niti Vahan (NV) Theme.
*
*  Filename:         task2a.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         task2a.py
# Functions:        detect_lane, compute_control
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy, read off Task_2A.ttt.
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# The integer signal the evaluation raises when the vehicle must stop.
STOP_SIGNAL = 'NV_stop'
##############################################################


##############################################################
##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

import math
import atexit

LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"

# ----------------------------------------------------------------------------
# TUNING KNOBS  (the only numbers you should need to touch)
# ----------------------------------------------------------------------------
LOOKAHEAD_M   = 0.18    # pure-pursuit look-ahead distance of the far row (m).
                        #   weaves / oscillates  -> increase (0.22, 0.26 ...)
                        #   cuts / runs wide in bends -> decrease (0.15, 0.13 ...)
DEBUG_SAVE    = False   # True for ONE dev run: saves annotated frames to ./debug_frames/
CAM_BIAS_PX   = 0.0     # camera axis vs vehicle axis (px). Measure with calib_task2a.py
CRUISE_SPEED  = 1.2     # rad/s on straights (was 0.7). Weaving on straights -> 1.0; still slow -> 1.4
BEND_SLOW     = 2.6     # speed = CRUISE * max(BEND_MIN, 1 - BEND_SLOW*|steer|)
BEND_MIN      = 0.38

# Image geometry (fractions of frame height, so any resolution works)
Y_FAR_FRAC    = 0.66    # look-ahead row (317 px on a 480 px frame)
Y_NEAR_FRAC   = 0.85    # near row (408 px): where the car actually is
ROI_TOP_FRAC  = 0.56
ROI_BOT_FRAC  = 0.885
DARK_V        = 70      # asphalt brightness ceiling (HSV V)
FLANK_PX      = 14      # how far either side of a segment we look for asphalt
DEFAULT_PAIR_PX = 400.0 # yellow<->white gap at the far row; re-learned online

# Vehicle / controller constants
WHEELBASE     = 0.120
TRACK_WIDTH   = 0.110
WHEEL_OFFSET  = 0.0275
LANE_WIDTH_M  = 0.20
MAX_STEERING  = math.radians(45.0)
K_PP          = 2.0 * WHEELBASE * LANE_WIDTH_M / (LOOKAHEAD_M ** 2)
STEER_ALPHA   = 0.6     # low-pass on the steering command
MAX_STEP      = 0.20    # rad per iteration - stops one bad frame from yanking the wheel
K_I           = 0.05    # slow near-row trim: removes the steady offset a wrong LOOKAHEAD_M leaves
I_LIMIT       = 0.35    # max trim, in lane widths (small: a big trim wound up and pushed the car off-centre)
TRIM_LEAK     = 0.97    # trim forgets old offsets every frame
W_NEAR        = 0.25    # weight of the near-row error in the steering error (rest is far row)
YELLOW_AVOID  = 0.0     # constant push away from yellow. Left at 0: it fixed the inner lane but made
                        #   the outer lane touch the white divider.
CURVE_CAP     = 0.10    # the outward shift never exceeds this many lane widths. Too much of it
                        #   pushed the inner-lane car over the white divider in the tight bottom-left bend.
K_CURVE       = 0.20    # in a bend the car cuts toward the INSIDE of the curve (yellow edge in the inner
                        #   lane, white divider in the outer lane). This aims the car outward by
                        #   K_CURVE lane widths per rad of steering. Inner lane still touches yellow
                        #   -> 0.5; outer lane touches white -> 0.3. Keep K_CURVE * K_PP below 1.


class _State(object):
    def __init__(self):
        self.lane = None            # remembered lane ('left' / 'right')
        self.width_far = DEFAULT_PAIR_PX
        self.have_width = False
        self.prev_pair = None       # (yellow_x, white_x) of previous pair
        self.last_xc = None         # last lane-centre x at the far row


_S = _State()
_STEER = 0.0
_TRIM = 0.0
_STEER_HIST = []
_STATS = {"n": 0, "sum_cm": 0.0, "within3": 0, "single": 0, "lost": 0, "frames": 0}


def _print_stats():
    if _STATS["frames"] == 0:
        return
    n = max(1, _STATS["n"])
    print("[self-check] frames=%d  pair-seen=%d  single-line=%d  lost=%d" % (
        _STATS["frames"], _STATS["n"], _STATS["single"], _STATS["lost"]))
    print("[self-check] near-row offset: mean %.1f cm, within 3 cm: %.0f%%" % (
        _STATS["sum_cm"] / n, 100.0 * _STATS["within3"] / n))


atexit.register(_print_stats)


# ----------------------------------------------------------------------------
# VISION
# ----------------------------------------------------------------------------
def _make_masks(frame):
    h, w = frame.shape[:2]
    y0 = int(h * ROI_TOP_FRAC)
    y1 = int(h * ROI_BOT_FRAC)
    strip = frame[y0:y1, :]

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(strip, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(hsv, np.array([12, 60, 60], dtype=np.uint8),
                         np.array([42, 255, 255], dtype=np.uint8))
    white = cv2.inRange(hls, np.array([0, 150, 0], dtype=np.uint8),
                        np.array([180, 255, 150], dtype=np.uint8))

    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    dark = (hsv[:, :, 2] < DARK_V).astype(np.uint8)
    return yellow, white, dark


def _flanked_by_asphalt(dark, mx, my):
    """True if there is dark road surface just left or right of (mx, my).
    Painted markings always have it; building walls / roof borders do not."""
    hh, ww = dark.shape
    my = int(max(2, min(hh - 3, my)))
    ok = 0.0
    for sx in (-FLANK_PX, FLANK_PX):
        x = int(mx + sx)
        if 0 <= x < ww:
            ok = max(ok, float(dark[my - 2:my + 3, x].mean()))
    return ok > 0.6


def _detect_lines(mask, dark, y_off, h):
    """Hough segments -> clusters (one per painted marking).
    `mask` is the ROI strip; y_off converts its rows back to frame rows and
    h is the full frame height.  Every cluster carries its x at the far row
    ('x'), at the near row ('xn') and its image slope."""
    w = mask.shape[1]
    y_far = Y_FAR_FRAC * h
    y_near = Y_NEAR_FRAC * h

    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180.0, threshold=8,
                            minLineLength=12, maxLineGap=25)
    if lines is None:
        return []

    segs = np.asarray(lines).reshape(-1, 4).astype(float)
    keep = [i for i, (a, b, c, d) in enumerate(segs)
            if _flanked_by_asphalt(dark, 0.5 * (a + c), 0.5 * (b + d))]
    if keep:                      # never let the filter empty the list
        segs = segs[keep]

    cands = []
    for x1, y1, x2, y2 in segs:
        y1 += y_off
        y2 += y_off
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 12.0 or abs(dy) < 5.0:
            continue
        slope = dx / dy
        if abs(slope) > 3.0:
            continue
        xf = x1 + slope * (y_far - y1)
        xn = x1 + slope * (y_near - y1)
        if not (math.isfinite(xf) and math.isfinite(xn)):
            continue
        # a boundary may legitimately sit just outside the image in a bend
        if xf < -250.0 or xf > w + 250.0:
            continue
        cands.append({"x": xf, "xn": xn, "length": length, "slope": slope})

    if not cands:
        return []

    cands.sort(key=lambda c: c["x"])
    clusters = [[cands[0]]]
    for c in cands[1:]:
        mean_x = sum(i["x"] for i in clusters[-1]) / len(clusters[-1])
        if abs(c["x"] - mean_x) <= 30.0:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    out = []
    for cl in clusters:
        tot = sum(i["length"] for i in cl)
        out.append({
            "x": sum(i["x"] * i["length"] for i in cl) / tot,
            "xn": sum(i["xn"] * i["length"] for i in cl) / tot,
            "slope": sum(i["slope"] * i["length"] for i in cl) / tot,
            "length": tot,
            "support": len(cl),
        })
    return out


def _orient(slope, expected_sign, pts):
    v = slope * expected_sign
    if v > 0.05:
        return pts
    if v < -0.05:
        return -pts
    return -5.0


def _choose_pair(yellow_lines, white_lines, width, hint):
    """Pick the yellow(road edge) + white(divider) pair.
    `hint` is the remembered lane; with no hint the lane comes from which side
    of the image centre the white divider is on."""
    if not yellow_lines or not white_lines:
        return None

    fc = width * 0.5
    best = None
    best_score = -1e18

    for y in yellow_lines:
        for wt in white_lines:
            sep = abs(wt["x"] - y["x"])
            if sep < 150.0 or sep > 650.0:
                continue

            if hint is not None:
                lane = hint
            elif wt["x"] > fc:
                lane = LANE_LEFT
            elif wt["x"] < fc:
                lane = LANE_RIGHT
            else:
                continue

            # yellow must lie OUTSIDE the divider
            if lane == LANE_LEFT and y["x"] >= wt["x"]:
                continue
            if lane == LANE_RIGHT and y["x"] <= wt["x"]:
                continue

            mid = 0.5 * (y["x"] + wt["x"])

            if lane == LANE_LEFT:
                orient = _orient(y["slope"], -1.0, 35.0) + _orient(wt["slope"], 1.0, 45.0)
            else:
                orient = _orient(wt["slope"], -1.0, 45.0) + _orient(y["slope"], 1.0, 35.0)

            width_score = max(0.0, 130.0 - 0.30 * abs(sep - _S.width_far))
            support = 2.0 * y["support"] + 2.0 * wt["support"]
            length = y["length"] / 10.0 + wt["length"] / 10.0
            centre = max(0.0, 40.0 - 0.08 * abs(mid - fc))
            divider = 6.0 * wt["support"]

            bonus = 0.0
            if _S.prev_pair is not None:
                d = abs(y["x"] - _S.prev_pair[0]) + abs(wt["x"] - _S.prev_pair[1])
                bonus = max(0.0, 300.0 - 0.75 * d)

            score = width_score + support + length + centre + divider + orient + bonus
            if score > best_score:
                best_score = score
                best = {"lane": lane, "yellow": y, "white": wt, "sep": sep}

    if best is not None:
        _S.prev_pair = (best["yellow"]["x"], best["white"]["x"])
    return best


def _best_single(lines, centre_for, last_xc):
    """Most trustworthy single line; centre_for(line_x) -> implied lane centre."""
    def key(c):
        k = 2.0 * c["support"] + c["length"] / 15.0
        if last_xc is not None:
            k -= 0.1 * abs(centre_for(c["x"]) - last_xc)
        return k
    return max(lines, key=key)


_DBG = {"n": 0, "k": 0}


def _debug_dump(frame, out, yl, wl):
    if not DEBUG_SAVE or _DBG["n"] >= 40:
        return
    _DBG["k"] += 1
    interesting = (not out["valid"]) or (not out["pair"]) or abs(out["r_far"]) > 0.55
    if not (interesting and _DBG["k"] % 4 == 0) and _DBG["k"] % 60 != 0:
        return
    import os
    os.makedirs("debug_frames", exist_ok=True)
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.line(img, (0, int(h * ROI_TOP_FRAC)), (w, int(h * ROI_TOP_FRAC)), (255, 0, 0), 1)
    cv2.line(img, (0, int(h * ROI_BOT_FRAC)), (w, int(h * ROI_BOT_FRAC)), (255, 0, 0), 1)
    yf = int(h * Y_FAR_FRAC)
    for c in yl:
        cv2.circle(img, (int(c["x"]), yf), 6, (0, 255, 255), -1)
    for c in wl:
        cv2.circle(img, (int(c["x"]), yf), 6, (255, 255, 255), -1)
    if out["xc_far"] is not None:
        cv2.circle(img, (int(out["xc_far"]), yf), 9, (0, 0, 255), 2)
    cv2.line(img, (w // 2, yf - 20), (w // 2, yf + 20), (0, 255, 0), 2)
    cv2.putText(img, "r=%.2f pair=%s lane=%s valid=%s steer=%.2f" % (
        out["r_far"], out["pair"], out["lane"], out["valid"], _STEER),
        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("debug_frames/f%04d_raw.png" % _DBG["n"], frame)
    cv2.imwrite("debug_frames/f%04d_ann.png" % _DBG["n"], img)
    _DBG["n"] += 1


def detect_lane(frame):
    out, yl, wl = _detect_lane_impl(frame)
    _debug_dump(frame, out, yl, wl)
    return out


def _detect_lane_impl(frame):
    """Returns a dict.  r_far = lateral position of the lane centre at the
    look-ahead row, in LANE WIDTHS, +ve = lane centre is to the RIGHT of the
    vehicle axis.  Dividing by the measured lane width removes the perspective
    scale, so no bird's-eye warp or pixel->metre constant is needed."""
    h, w = frame.shape[:2]
    cx = w / 2.0 + CAM_BIAS_PX

    yellow_mask, white_mask, dark = _make_masks(frame)
    y_off = int(h * ROI_TOP_FRAC)
    yl = _detect_lines(yellow_mask, dark, y_off, h)
    wl = _detect_lines(white_mask, dark, y_off, h)

    out = {"center_x": -1, "lane": LANE_UNKNOWN, "valid": False, "pair": False,
           "r_far": 0.0, "r_near": None, "xc_far": None, "sep_far": None}

    pair = _choose_pair(yl, wl, w, _S.lane)
    if pair is None and _S.lane is not None:
        pair = _choose_pair(yl, wl, w, None)

    if pair is not None:
        y, wt, lane, sep = pair["yellow"], pair["white"], pair["lane"], pair["sep"]

        if abs(wt["x"] - w / 2.0) > 60.0:
            _S.lane = lane

        if not _S.have_width:
            _S.width_far = sep
            _S.have_width = True
        elif abs(sep - _S.width_far) < 0.25 * _S.width_far:
            _S.width_far += 0.05 * (sep - _S.width_far)

        width_used = sep if abs(sep - _S.width_far) < 0.25 * _S.width_far else _S.width_far
        xc = 0.5 * (y["x"] + wt["x"])
        _S.last_xc = xc

        # +1 if the white divider is to the right of the yellow edge, else -1.
        # Moving the target toward the white keeps the wheels off the yellow.
        away = 1.0 if wt["x"] > y["x"] else -1.0
        out.update(valid=True, pair=True, lane=lane, center_x=int(round(xc)),
                   r_far=(xc - cx) / width_used + away * YELLOW_AVOID,
                   xc_far=xc, sep_far=sep)

        sep_n = abs(wt["xn"] - y["xn"])
        if sep_n > 60.0:
            out["r_near"] = (0.5 * (y["xn"] + wt["xn"]) - cx) / sep_n + away * YELLOW_AVOID
        return out, yl, wl

    wd = _S.width_far
    lane = _S.lane

    if wl:
        def centre_w(x, lane=lane):
            ln = lane if lane is not None else (LANE_LEFT if x > cx else LANE_RIGHT)
            return x - wd / 2.0 if ln == LANE_LEFT else x + wd / 2.0
        b = _best_single(wl, centre_w, _S.last_xc)
        xc = centre_w(b["x"])
    elif yl:
        def centre_y(x, lane=lane):
            ln = lane if lane is not None else (LANE_LEFT if x < cx else LANE_RIGHT)
            return x + wd / 2.0 if ln == LANE_LEFT else x - wd / 2.0
        b = _best_single(yl, centre_y, _S.last_xc)
        xc = centre_y(b["x"])
    else:
        return out, yl, wl

    _S.last_xc = xc
    out.update(valid=True, pair=False, lane=lane if lane else LANE_UNKNOWN,
               center_x=int(round(max(0.0, min(w - 1.0, xc)))),
               r_far=(xc - cx) / wd, xc_far=xc, sep_far=None)
    return out, yl, wl


# ----------------------------------------------------------------------------
# CONTROL
# ----------------------------------------------------------------------------
def ackermann_wheel_angles(delta):
    delta = float(delta)
    if abs(delta) < 1e-9:
        return 0.0, 0.0

    half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET
    turn_radius = WHEELBASE / math.tan(abs(delta))
    inner_radius = max(turn_radius - half_track, 1e-6)
    outer_radius = turn_radius + half_track

    inner_angle = math.atan(WHEELBASE / inner_radius)
    outer_angle = math.atan(WHEELBASE / outer_radius)

    if delta > 0.0:                       # steering joints are +ve to the LEFT
        return float(inner_angle), float(outer_angle)
    return float(-outer_angle), float(-inner_angle)


def compute_control(detection, stop_requested):
    """Pure pursuit on the look-ahead lane-centre offset.

    Sign convention (this was the bug in the previous version):
        r_far > 0  -> lane centre is to the RIGHT of the vehicle
        steering joints are positive to the LEFT
        so the correction must be NEGATIVE:  delta = -atan(K_PP * r_far)
    """
    global _STEER, _TRIM

    _STATS["frames"] += 1
    valid = bool(detection) and detection.get("valid", False)

    if valid:
        rn = detection.get("r_near")
        _TRIM *= TRIM_LEAK
        if detection.get("pair", False) and rn is not None and not stop_requested:
            _TRIM = max(-I_LIMIT, min(I_LIMIT, _TRIM + K_I * float(rn)))

        r_far = float(detection["r_far"])
        if detection.get("pair", False) and rn is not None:
            r_far = (1.0 - W_NEAR) * r_far + W_NEAR * float(rn)
        # steering > 0 is a left turn, so the inside is on the left and the target moves right (r > 0)
        curve_shift = max(-CURVE_CAP, min(CURVE_CAP, K_CURVE * _STEER))
        r = max(-1.5, min(1.5, r_far + _TRIM + curve_shift))
        raw = -math.atan(K_PP * r)
        raw = max(-MAX_STEERING, min(MAX_STEERING, raw))

        step = STEER_ALPHA * (raw - _STEER)
        step = max(-MAX_STEP, min(MAX_STEP, step))
        _STEER = _STEER + step

        speed = CRUISE_SPEED * max(BEND_MIN, 1.0 - BEND_SLOW * abs(_STEER))
        # already far from the centre (e.g. drifting toward the divider): slow down and recover
        speed *= max(0.5, 1.0 - 0.8 * max(0.0, abs(r) - 0.25))
        _STEER_HIST.append(_STEER)
        if len(_STEER_HIST) > 20:
            _STEER_HIST.pop(0)

        if detection.get("pair", False):
            _STATS["n"] += 1
            if rn is not None:
                off_cm = abs(rn - 0.0) * LANE_WIDTH_M * 100.0
                _STATS["sum_cm"] += off_cm
                if off_cm <= 3.0:
                    _STATS["within3"] += 1
        else:
            _STATS["single"] += 1
    else:
        _STATS["lost"] += 1
        # No lane seen: keep turning the way the road has been turning
        # (average of recent steering), never snap back to straight, and creep.
        if _STEER_HIST:
            _STEER = 0.7 * _STEER + 0.3 * (sum(_STEER_HIST) / len(_STEER_HIST))
        speed = 0.35 * CRUISE_SPEED

    if stop_requested:
        speed = 0.0

    left, right = ackermann_wheel_angles(_STEER)
    return float(left), float(right), float(speed)

##############################################################

#################### DO NOT EDIT BELOW THIS LINE ####################
#
# Connecting to the simulator, reading the camera and writing to the joints.
# None of it involves a decision you are being marked on - it is here so that
# every team reads the same picture and drives the same vehicle.
#
#####################################################################

def connect_and_locate():
    '''Open the remote API connection and look up every scene handle once.

    Each getObject() is a network round trip, which is why they all happen
    here rather than inside the loop.
    '''
    try:
        sim = RemoteAPIClient().getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
            f'  Is CoppeliaSim running with Task_2A.ttt open?\n'
            f'  Simulator said: {err}')

    handles = {}
    for name, path in (('vision_sensor', VISION_SENSOR),
                       ('steer_left', STEER_LEFT),
                       ('steer_right', STEER_RIGHT),
                       ('motor_left', MOTOR_LEFT),
                       ('motor_right', MOTOR_RIGHT)):
        try:
            handles[name] = sim.getObject(path)
        except Exception as err:
            raise RuntimeError(
                f'No object at "{path}" in the open scene.\n'
                f'  Open Task_2A.ttt from the repository.\n'
                f'  Simulator said: {err}')
    return sim, handles


def read_frame(sim, handles):
    '''Grab one frame from the vehicle's camera as a BGR image.

    The vision sensor hands back a raw byte buffer and a resolution, not an
    image, and three corrections are needed to turn one into the other. None
    of these problems announces itself - each produces a detector that looks
    like it is working on a frame that is wrong:

      * handleVisionSensor() - this scene's sensor renders only when asked.
        Without it every read returns the SAME frame for the whole lap.
      * cv2.flip(frame, 0) - the buffer's first row is the BOTTOM of the
        image, so the road comes out vertically mirrored.
      * RGB2BGR - the buffer is RGB and OpenCV assumes BGR. Without the swap
        every HSV threshold you tuned in Task 1C selects the wrong colour.

    Note `res[1], res[0]` in the reshape too: the resolution comes back as
    (width, height) while NumPy wants rows first.

    Returns None if no frame is ready yet, which happens on the first few
    iterations after the simulation starts.
    '''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Write the two steering angles and the drive speed to the joints.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


def main():
    sim, handles = connect_and_locate()

    # While you are developing you press Play yourself. Under evaluation the
    # harness resets the scene, places the vehicle in the lane being tested
    # and starts the simulation - so wait for that rather than starting your
    # own, which would race the harness into position.
    #
    # Your script must never call startSimulation(), stopSimulation() or move
    # the vehicle. Reading the camera and driving the joints is all it does.
    deadline = time.monotonic() + 120.0
    while sim.getSimulationState() != sim.simulation_advancing_running:
        if time.monotonic() > deadline:
            print('The simulation never started. Press Play in CoppeliaSim, '
                  'then run this script again.')
            return 1
        time.sleep(0.05)
    print('Simulation running - driving.')

    frames = 0
    try:
        # The harness stops the simulation when the lap is over, which is the
        # cue to exit - hence testing the state rather than looping forever.
        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                continue
            frames += 1

            # getInt32Signal() returns None until the signal is first set, so
            # "not set" reads as "not stopping" and the first frames behave.
            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))

            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection,
                                                             stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except Exception as err:
        # A call raising because the simulation stopped mid-frame is normal at
        # the end of a run; anything else is a bug worth seeing.
        print(f'Loop ended: {type(err).__name__}: {err}')
    finally:
        # Without this, whatever speed was last commanded is still being
        # applied and the vehicle drives on unsteered.
        try:
            sim.setJointTargetVelocity(handles['motor_left'], 0.0)
            sim.setJointTargetVelocity(handles['motor_right'], 0.0)
        except Exception:
            pass

    print(f'{frames} frames processed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
