'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is the standalone implementation of Task 3A of Niti Vahan (NV) Theme.
*  Phase 1: Autonomous Lane Following & Evaluator Stop Handling Baseline.
*
*  Filename:         task3.py
*  Created:          2026
*  Last Modified:    2026
*  Author:           Team 4027
*
*****************************************************************************************
'''

####################### IMPORT MODULES #######################
import atexit
import math
import sys
import time

import cv2
import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
##############################################################


##################### SCENE CONSTANTS ########################
# Paths into the CoppeliaSim scene hierarchy
VISION_SENSOR = '/Niti_Vahan/Camera_joint/Cuboid/visionSensor'
STEER_LEFT    = '/steeringLeft'
STEER_RIGHT   = '/steeringRight'
MOTOR_LEFT    = '/motorLeft'
MOTOR_RIGHT   = '/motorRight'

# Evaluator stop signal
STOP_SIGNAL = 'NV_stop'
##############################################################


################### CONTROLLER TUNING CONSTANTS ##############
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"

# Pure pursuit & speed knobs
LOOKAHEAD_M   = 0.18        # Look-ahead distance of the far row (m)
CAM_BIAS_PX   = 0.0         # Camera axis vs vehicle axis bias (px)
CRUISE_SPEED  = 1.2         # Base speed on straights (rad/s)
BEND_SLOW     = 2.6         # Deceleration scaling on curves
BEND_MIN      = 0.38        # Minimum speed fraction in bends

# Image geometry (fraction of frame height)
Y_FAR_FRAC    = 0.66        # Look-ahead row
Y_NEAR_FRAC   = 0.85        # Near row (vehicle position)
ROI_TOP_FRAC  = 0.56
ROI_BOT_FRAC  = 0.885
DARK_V        = 70          # Asphalt brightness ceiling (HSV V)
FLANK_PX      = 14          # Flank sampling distance for asphalt validation
DEFAULT_PAIR_PX = 400.0     # Default divider-to-boundary pixel separation

# Vehicle dimensions & kinematics
WHEELBASE     = 0.120       # L: distance between front and rear axles (m)
TRACK_WIDTH   = 0.110       # W: distance between left and right wheel centres (m)
WHEEL_OFFSET  = 0.0275      # O: distance between kingpin axis and wheel centre (m)
LANE_WIDTH_M  = 0.20        # Physical lane width (m)
MAX_STEERING  = math.radians(45.0)

# Pure Pursuit gain: K_PP = 2 * L * Lane_Width / (Lookahead^2)
K_PP          = 2.0 * WHEELBASE * LANE_WIDTH_M / (LOOKAHEAD_M ** 2)
STEER_ALPHA   = 0.6         # Low-pass filter coefficient on steering
MAX_STEP      = 0.20        # Maximum steering angle change per frame (rad)
K_I           = 0.05        # Near-row integral trim gain
I_LIMIT       = 0.35        # Maximum integral trim authority
TRIM_LEAK     = 0.97        # Integral trim decay rate per frame
W_NEAR        = 0.25        # Weight of near-row error in total lateral error
YELLOW_AVOID  = 0.0         # Offset pushing away from yellow boundary
CURVE_CAP     = 0.10        # Maximum outward shift cap in curves
K_CURVE       = 0.20        # Curve feedforward outward compensation gain
##############################################################


class _State(object):
    def __init__(self):
        self.lane = None            # Latched lane ('left' / 'right')
        self.width_far = DEFAULT_PAIR_PX
        self.have_width = False
        self.prev_pair = None       # (yellow_x, white_x) of previous frame
        self.last_xc = None         # Last lane center x at far row


# Module-level state tracking
_S = _State()
_STEER = 0.0
_TRIM = 0.0
_STEER_HIST = []
_STATS = {"n": 0, "sum_cm": 0.0, "within3": 0, "single": 0, "lost": 0, "frames": 0}


def _print_stats():
    if _STATS["frames"] == 0:
        return
    n = max(1, _STATS["n"])
    print("[Task 3A Baseline Stats] frames=%d  pair-seen=%d  single-line=%d  lost=%d" % (
        _STATS["frames"], _STATS["n"], _STATS["single"], _STATS["lost"]))
    print("[Task 3A Baseline Stats] near-row offset: mean %.1f cm, within 3 cm: %.0f%%" % (
        _STATS["sum_cm"] / n, 100.0 * _STATS["within3"] / n))


atexit.register(_print_stats)


# ----------------------------------------------------------------------------
# VISION PERCEPTION PIPELINE
# ----------------------------------------------------------------------------
def _make_masks(frame):
    '''Extract yellow boundary and white divider masks in the road ROI.'''
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
    '''Verify whether a line segment is flanked by dark road surface.'''
    hh, ww = dark.shape
    my = int(max(2, min(hh - 3, my)))
    ok = 0.0
    for sx in (-FLANK_PX, FLANK_PX):
        x = int(mx + sx)
        if 0 <= x < ww:
            ok = max(ok, float(dark[my - 2:my + 3, x].mean()))
    return ok > 0.6


def _detect_lines(mask, dark, y_off, h):
    '''Detect Hough segments, filter by asphalt context, extrapolate and cluster.'''
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
    if keep:
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
    '''Score orientation consistency of boundary lines.'''
    v = slope * expected_sign
    if v > 0.05:
        return pts
    if v < -0.05:
        return -pts
    return -5.0


def _choose_pair(yellow_lines, white_lines, width, hint):
    '''Pair yellow road edge and white dashed divider.'''
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

            # Yellow boundary must lie outside the white divider
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
    '''Select the most confident single marking when pairing is partial.'''
    def key(c):
        k = 2.0 * c["support"] + c["length"] / 15.0
        if last_xc is not None:
            k -= 0.1 * abs(centre_for(c["x"]) - last_xc)
        return k
    return max(lines, key=key)


def _detect_lane_impl(frame):
    '''Estimate normalized lateral offsets and lane position.'''
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

        away = 1.0 if wt["x"] > y["x"] else -1.0
        out.update(valid=True, pair=True, lane=lane, center_x=int(round(xc)),
                   r_far=(xc - cx) / width_used + away * YELLOW_AVOID,
                   xc_far=xc, sep_far=sep)

        sep_n = abs(wt["xn"] - y["xn"])
        if sep_n > 60.0:
            out["r_near"] = (0.5 * (y["xn"] + wt["xn"]) - cx) / sep_n + away * YELLOW_AVOID
        return out

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
        return out

    _S.last_xc = xc
    out.update(valid=True, pair=False, lane=lane if lane else LANE_UNKNOWN,
               center_x=int(round(max(0.0, min(w - 1.0, xc)))),
               r_far=(xc - cx) / wd, xc_far=xc, sep_far=None)
    return out


def detect_lane(frame):
    '''Public lane detection entry point.'''
    return _detect_lane_impl(frame)


# ----------------------------------------------------------------------------
# CONTROL & ACTUATION
# ----------------------------------------------------------------------------
def ackermann_wheel_angles(delta):
    '''Convert single steering angle into dual front-wheel steering angles.'''
    delta = float(delta)
    if abs(delta) < 1e-9:
        return 0.0, 0.0

    half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET
    turn_radius = WHEELBASE / math.tan(abs(delta))
    inner_radius = max(turn_radius - half_track, 1e-6)
    outer_radius = turn_radius + half_track

    inner_angle = math.atan(WHEELBASE / inner_radius)
    outer_angle = math.atan(WHEELBASE / outer_radius)

    if delta > 0.0:
        return float(inner_angle), float(outer_angle)
    return float(-outer_angle), float(-inner_angle)


def compute_control(detection, stop_requested):
    '''Pure Pursuit lookahead steering controller with cornering deceleration.'''
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

        curve_shift = max(-CURVE_CAP, min(CURVE_CAP, K_CURVE * _STEER))
        r = max(-1.5, min(1.5, r_far + _TRIM + curve_shift))
        raw = -math.atan(K_PP * r)
        raw = max(-MAX_STEERING, min(MAX_STEERING, raw))

        step = STEER_ALPHA * (raw - _STEER)
        step = max(-MAX_STEP, min(MAX_STEP, step))
        _STEER = _STEER + step

        speed = CRUISE_SPEED * max(BEND_MIN, 1.0 - BEND_SLOW * abs(_STEER))
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
        if _STEER_HIST:
            _STEER = 0.7 * _STEER + 0.3 * (sum(_STEER_HIST) / len(_STEER_HIST))
        speed = 0.35 * CRUISE_SPEED

    if stop_requested:
        speed = 0.0

    left, right = ackermann_wheel_angles(_STEER)
    return float(left), float(right), float(speed)


def drive(sim, handles, left_angle, right_angle, speed):
    '''Single point of joint and motor actuation.'''
    sim.setJointTargetPosition(handles['steer_left'], float(left_angle))
    sim.setJointTargetPosition(handles['steer_right'], float(right_angle))
    sim.setJointTargetVelocity(handles['motor_left'], float(speed))
    sim.setJointTargetVelocity(handles['motor_right'], float(speed))


# ----------------------------------------------------------------------------
# SIMULATION INTERFACE & MAIN LOOP
# ----------------------------------------------------------------------------
MAX_CONSECUTIVE_DROPPED_FRAMES = 60   # Safety threshold for consecutive None frames


def connect_and_locate():
    '''Initialize remote API connection and retrieve scene handles once.'''
    try:
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        sim.getSimulationTime()
    except Exception as err:
        raise ConnectionError(
            f'Could not reach CoppeliaSim on the ZMQ Remote API port.\n'
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
                f'  Simulator said: {err}')
    return client, sim, handles


def read_frame(sim, handles):
    '''Capture and format one frame from the vision sensor.'''
    sim.handleVisionSensor(handles['vision_sensor'])
    img, res = sim.getVisionSensorImg(handles['vision_sensor'], 0)
    if len(img) == 0:
        return None
    frame = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    frame = cv2.flip(frame, 0)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    '''Phase 1 standalone execution entry point.'''
    client, sim, handles = connect_and_locate()

    exit_code = 0
    frames = 0
    consecutive_dropped = 0

    try:
        # Wait for the evaluator harness to reset and start the simulation
        deadline = time.monotonic() + 120.0
        while sim.getSimulationState() != sim.simulation_advancing_running:
            if time.monotonic() > deadline:
                print('Simulation start timeout. Start simulation in CoppeliaSim.')
                return 1
            time.sleep(0.05)
        print('Simulation advancing - Task 3A Phase 1 driving active.')

        while sim.getSimulationState() == sim.simulation_advancing_running:
            frame = read_frame(sim, handles)
            if frame is None:
                drive(sim, handles, 0.0, 0.0, 0.0)
                consecutive_dropped += 1
                if consecutive_dropped > MAX_CONSECUTIVE_DROPPED_FRAMES:
                    raise RuntimeError(
                        f'Vision sensor returned None for {consecutive_dropped} consecutive frames.'
                    )
                continue

            consecutive_dropped = 0
            frames += 1

            stop_requested = bool(sim.getInt32Signal(STOP_SIGNAL))
            detection = detect_lane(frame)
            left_angle, right_angle, speed = compute_control(detection, stop_requested)
            drive(sim, handles, left_angle, right_angle, speed)

    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        exit_code = 0
    except Exception as err:
        print(f'Execution halted with error: {type(err).__name__}: {err}')
        exit_code = 1
    finally:
        # Mandatory fail-safe: Zero all drive motors
        try:
            drive(sim, handles, 0.0, 0.0, 0.0)
        except Exception:
            pass

    if exit_code == 0:
        print(f'Run concluded cleanly: {frames} frames processed.')
    else:
        print(f'Run terminated abnormally after {frames} frames.')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
