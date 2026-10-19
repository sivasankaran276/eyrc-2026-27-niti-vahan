'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
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
# Filename:         lane_detection.py
# Functions:        detect_lane, _detect_lane_raw, _make_masks, _detect_lines,
#                   _choose_pair, _single_line_lane, _reset_state, _clamp_x
# Global variables: LANE_WIDTH_PX, DEFAULT_PAIR_W, Y_REF, BETWEEN_MARGIN, AMBIGUOUS_BAND,
#                   SCENE_CUT_DIFF, _state


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

LANE_WIDTH_PX = 315.0
DEFAULT_PAIR_W = 400.0      # yellow<->white separation at Y_REF (learned online)
Y_REF = 300.0               # common reference row for line extrapolation
BETWEEN_MARGIN = 90.0       # vehicle (frame centre) must lie between yellow and white
AMBIGUOUS_BAND = 50.0       # single line this close to the centre -> keep previous lane
SCENE_CUT_DIFF = 15.0       # mean thumbnail change that signals a new clip

# State kept across frames (smoothing + lane hysteresis)
_state = {
    "lane": None,
    "cx": None,
    "pending": None,
    "pending_n": 0,
    "miss": 0,
    "width": DEFAULT_PAIR_W,
    "thumb": None,
}


def _reset_state():
    """Forget the tracking state (keeps the learned lane width)."""
    _state.update(lane=None, cx=None, pending=None, pending_n=0, miss=0)


def _clamp_x(x, width):
    return max(0.0, min(float(width - 1), float(x)))


def _make_masks(frame):
    """Build yellow and white masks restricted to the useful road region."""
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(
        hsv,
        np.array([12, 60, 60], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8)
    )
    white = cv2.inRange(
        hls,
        np.array([0, 150, 0], dtype=np.uint8),
        np.array([180, 255, 150], dtype=np.uint8)
    )

    # Ignore the bottom of the frame (white clutter) and the sky/horizon.
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.56):int(h * 0.885), :] = 255

    yellow = cv2.bitwise_and(yellow, roi)
    white = cv2.bitwise_and(white, roi)

    kernel = np.ones((3, 3), dtype=np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)

    return yellow, white


def _detect_lines(mask):
    """Hough segments -> extrapolated to row Y_REF -> clustered per marking."""
    h, w = mask.shape

    lines = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 180.0,
        threshold=8, minLineLength=12, maxLineGap=25
    )
    if lines is None:
        return []

    candidates = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length < 12.0 or abs(dy) < 5.0:
            continue

        slope = dx / dy
        if abs(slope) > 3.0:
            continue

        x_ref = x1 + slope * (Y_REF - y1)
        if not np.isfinite(x_ref):
            continue
        if x_ref < -250.0 or x_ref > w + 250.0:
            continue

        candidates.append({"x": float(x_ref), "length": length, "slope": slope})

    if not candidates:
        return []

    candidates.sort(key=lambda c: c["x"])

    clusters = []
    for candidate in candidates:
        placed = False
        for cluster in clusters:
            mean_x = np.mean([item["x"] for item in cluster])
            if abs(candidate["x"] - mean_x) <= 30.0:
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])

    output = []
    for cluster in clusters:
        total_length = sum(item["length"] for item in cluster)
        weighted_x = sum(item["x"] * item["length"] for item in cluster) / total_length
        weighted_slope = sum(item["slope"] * item["length"] for item in cluster) / total_length
        output.append({
            "x": float(weighted_x),
            "length": float(total_length),
            "support": len(cluster),
            "slope": float(weighted_slope),
        })

    return output


def _choose_pair(yellow_lines, white_lines, width):
    """
    Pick the best yellow/white pair.

    The lane follows from the ORDER of the two markings (independent of where
    the camera sits on the vehicle):
        yellow left of white  -> LEFT lane  (yellow slope < 0, white slope > 0)
        yellow right of white -> RIGHT lane (white slope < 0, yellow slope > 0)
    and the vehicle (frame centre) must lie between the two markings.
    """
    if not yellow_lines or not white_lines:
        return None

    frame_center = width * 0.5
    pair_width_px = _state["width"]

    best = None
    best_score = -1e9

    for yellow in yellow_lines:
        yellow_x = float(yellow["x"])
        yellow_slope = float(yellow.get("slope", 0.0))

        for white in white_lines:
            white_x = float(white["x"])
            white_slope = float(white.get("slope", 0.0))

            separation = abs(white_x - yellow_x)
            if separation < 150.0 or separation > 650.0:
                continue

            lo = min(yellow_x, white_x)
            hi = max(yellow_x, white_x)

            # The vehicle must be inside the lane bounded by the two markings.
            if frame_center < lo - BETWEEN_MARGIN or frame_center > hi + BETWEEN_MARGIN:
                continue

            midpoint = (yellow_x + white_x) * 0.5
            expected_lane = LANE_LEFT if yellow_x < white_x else LANE_RIGHT

            # ---- orientation ----
            if expected_lane == LANE_LEFT:
                y_good, w_good = yellow_slope < -0.05, white_slope > 0.05
                y_bad, w_bad = yellow_slope > 0.05, white_slope < -0.05
            else:
                y_good, w_good = yellow_slope > 0.05, white_slope < -0.05
                y_bad, w_bad = yellow_slope < -0.05, white_slope > 0.05

            orientation_score = 0.0
            orientation_score += 35.0 if y_good else (-35.0 if y_bad else -5.0)
            orientation_score += 45.0 if w_good else (-45.0 if w_bad else -5.0)

            # ---- width ----
            width_score = max(0.0, 130.0 - 0.30 * abs(separation - pair_width_px))

            # ---- visual support ----
            support_score = 2.0 * float(yellow["support"]) + 2.0 * float(white["support"])
            length_score = float(yellow["length"]) / 10.0 + float(white["length"]) / 10.0

            # ---- vehicle inside the pair ----
            inside_score = 70.0 if lo <= frame_center <= hi else 20.0

            # ---- centre proximity ----
            centre_score = max(0.0, 40.0 - 0.08 * abs(midpoint - frame_center))

            # ---- divider support ----
            divider_score = 6.0 * float(white["support"])

            score = (width_score + support_score + length_score + inside_score
                     + centre_score + divider_score + orientation_score)

            if score > best_score:
                best_score = score
                best = {
                    "yellow_x": yellow_x,
                    "white_x": white_x,
                    "width": separation,
                    "lane": expected_lane,
                }

    return best


def _single_line_lane(x, is_white, frame_center):
    """
    Lane from ONE visible marking. White divider right of vehicle -> left lane;
    yellow boundary left of vehicle -> left lane. Near the centre the answer is
    ambiguous, so the previous lane is kept when there is one.
    """
    if abs(x - frame_center) < AMBIGUOUS_BAND and _state["lane"] is not None:
        return _state["lane"]
    if is_white:
        return LANE_LEFT if x > frame_center else LANE_RIGHT
    return LANE_LEFT if x < frame_center else LANE_RIGHT


def _detect_lane_raw(frame):
    """Single-frame detector. 'pair_w' is returned when a yellow+white pair was used."""
    height, width = frame.shape[:2]
    frame_center = width / 2.0

    yellow_mask, white_mask = _make_masks(frame)
    yellow_lines = _detect_lines(yellow_mask)
    white_lines = _detect_lines(white_mask)

    pair = _choose_pair(yellow_lines, white_lines, width)

    if pair is not None:
        return {
            "center_x": int(round(_clamp_x(
                (pair["white_x"] + pair["yellow_x"]) / 2.0,
                width
            ))),
            "lane": pair["lane"],
            "pair_w": pair["width"],
        }

    half = _state["width"] / 2.0

    # ---- white-only fallback ----
    if white_lines:
        best = max(
            white_lines,
            key=lambda c: 2.0 * c["support"] + c["length"] / 15.0
        )
        wx = best["x"]
        lane = _single_line_lane(wx, True, frame_center)
        cx = wx - half if lane == LANE_LEFT else wx + half
        return {
            "center_x": int(round(_clamp_x(cx, width))),
            "lane": lane
        }

    # ---- yellow-only fallback ----
    if yellow_lines:
        best = max(
            yellow_lines,
            key=lambda c: 2.0 * c["support"] + c["length"] / 15.0
        )
        yx = best["x"]
        lane = _single_line_lane(yx, False, frame_center)
        cx = yx + half if lane == LANE_LEFT else yx - half
        return {
            "center_x": int(round(_clamp_x(cx, width))),
            "lane": lane
        }

    return {
        "center_x": -1,
        "lane": LANE_UNKNOWN
    }


def detect_lane(frame):
    '''
    Detect the lane centre and vehicle lane, with temporal smoothing and
    lane-label hysteresis.
    '''
    s = _state

    # New clip? (big change of the whole picture) -> drop the tracking state
    thumb = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        (32, 24),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    if s["thumb"] is not None and float(
        np.mean(np.abs(thumb - s["thumb"]))
    ) > SCENE_CUT_DIFF:
        _reset_state()

    s["thumb"] = thumb

    raw = _detect_lane_raw(frame)

    # Learn the real yellow<->white separation from good pairs
    if "pair_w" in raw and 300.0 <= raw["pair_w"] <= 500.0:
        s["width"] = 0.9 * s["width"] + 0.1 * raw["pair_w"]

    # Nothing found: hold the previous answer briefly
    if raw["lane"] == LANE_UNKNOWN or raw["center_x"] < 0:
        s["miss"] += 1
        if s["lane"] is not None and s["miss"] <= 5:
            return {
                "center_x": int(round(s["cx"])),
                "lane": s["lane"]
            }
        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN
        }

    s["miss"] = 0

    cx = float(raw["center_x"])
    lane = raw["lane"]

    # First frame, or a big jump: accept directly
    if s["lane"] is None or abs(cx - s["cx"]) > 150.0:
        s.update(
            lane=lane,
            cx=cx,
            pending=None,
            pending_n=0
        )
        return {
            "center_x": int(round(cx)),
            "lane": lane
        }

    # Lane hysteresis: switch only after 3 consecutive disagreeing frames
    if lane != s["lane"]:
        if s["pending"] == lane:
            s["pending_n"] += 1
        else:
            s["pending"], s["pending_n"] = lane, 1

        if s["pending_n"] >= 3:
            s["lane"], s["pending"], s["pending_n"] = lane, None, 0
            s["cx"] = cx
        else:
            cx = s["cx"]

    else:
        s["pending"], s["pending_n"] = None, 0

    s["cx"] = 0.6 * cx + 0.4 * s["cx"]

    return {
        "center_x": int(round(s["cx"])),
        "lane": s["lane"]
    }


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]   whatever detect_lane() returned
    `frame_index` :     [ int ]      index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]      {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError(
            "{} must return a dict, got {}".format(
                where,
                type(result).__name__
            )
        )

    missing = {"center_x", "lane"} - set(result.keys())

    if missing:
        raise ValueError(
            "{} is missing the key(s): {}".format(
                where,
                ", ".join(sorted(missing))
            )
        )

    center_x = result["center_x"]

    if isinstance(center_x, bool) or not isinstance(
        center_x,
        (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "{} returned center_x of type {}, expected a number".format(
                where,
                type(center_x).__name__
            )
        )

    center_x = int(round(float(center_x)))

    lane = result["lane"]

    if not isinstance(lane, str):
        raise TypeError(
            "{} returned lane of type {}, expected a string".format(
                where,
                type(lane).__name__
            )
        )

    lane = lane.strip().lower()

    if lane not in VALID_LANES:
        raise ValueError(
            "{} returned lane = '{}', expected one of {}".format(
                where,
                result["lane"],
                ", ".join(VALID_LANES)
            )
        )

    return {
        "center_x": center_x,
        "lane": lane
    }


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :          [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :         [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :         [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(
        canvas,
        (width // 2, height),
        (width // 2, height - 40),
        (128, 128, 128),
        1
    )

    center_x = result["center_x"]

    if 0 <= center_x < width:
        cv2.line(
            canvas,
            (center_x, height),
            (center_x, height // 2),
            (0, 0, 255),
            2
        )
        cv2.circle(
            canvas,
            (center_x, height - 10),
            5,
            (0, 0, 255),
            -1
        )

    label = "lane: {}   center_x: {}".format(
        result["lane"],
        center_x
    )

    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]       path to the video file
    `show` :        [ bool ]      if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]      one dict per frame:
                                  {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            "no such video file: {}".format(video_path)
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise IOError(
            "OpenCV could not open the video: {}".format(video_path)
        )

    window = "Task 1C - {}".format(
        os.path.basename(video_path)
    )

    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(
                detect_lane(frame.copy()),
                frame_index
            )

            results.append({
                "frame": frame_index,
                **result
            })

            if show:
                cv2.imshow(
                    window,
                    draw_overlay(frame, result)
                )

                if cv2.waitKey(1) & 0xFF in (
                    ord('q'),
                    27
                ):
                    break

            frame_index += 1

    finally:
        capture.release()

        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError(
            "no frames could be read from: {}".format(video_path)
        )

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :      [ str ]      path to the video that was processed
    `results` :         [ list ]      output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)

    counts = {
        lane: 0 for lane in VALID_LANES
    }

    for entry in results:
        counts[entry["lane"]] += 1

    not_found = sum(
        1 for entry in results
        if entry["center_x"] < 0
    )

    print(
        "{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
            os.path.basename(video_path),
            total,
            counts[LANE_LEFT],
            counts[LANE_RIGHT],
            counts[LANE_UNKNOWN],
            not_found
        )
    )


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :       [ list ]       the raw command-line arguments

    Returns:
    ---
    `videos` :      [ list ]       paths to individual video files, folders expanded
    '''
    videos = []

    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(
                f for f in os.listdir(raw)
                if f.lower().endswith(".mp4")
            )

            if not found:
                raise FileNotFoundError(
                    "no .mp4 files in the folder: {}".format(raw)
                )

            videos.extend(
                os.path.join(raw, f)
                for f in found
            )

        else:
            videos.append(raw)

    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos."
    )

    parser.add_argument(
        "videos",
        nargs="+",
        help="video file(s), or a folder holding them "
             "(e.g. 'public')"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="display the detection overlay while processing (press q to stop)"
    )

    parser.add_argument(
        "--out",
        metavar="FILE",
        help="write the per-frame results to this JSON file"
    )

    args = parser.parse_args()

    all_results = {}

    for video_path in expand_videos(args.videos):
        results = process_video(
            video_path,
            show=args.show
        )

        summarise(
            video_path,
            results
        )

        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(
            args.out,
            "w",
            encoding="utf-8"
        ) as handle:
            json.dump(
                all_results,
                handle,
                indent=2
            )

        print(
            "\nresults written to {}".format(args.out)
        )


if __name__ == "__main__":
    main()