'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 2B of Niti Vahan (NV) Theme.
*
*  Filename:         sign_obstacle_detection.py
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
# Filename:         sign_obstacle_detection.py
# Functions:        detect_signs, detect_obstacle
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import cv2
import numpy as np
##############################################################

# The only sign labels detect_signs() is allowed to return.
SIGN_LABELS = ("School", "Speed40", "Parking", "Hospital")


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_signs(image):
    '''
    Purpose:
    ---
    Detect every road sign visible in a single image, and report what each
    one is and how large it appears. An image can hold zero, one, or more
    than one sign at once.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `detections` :  [ list of dict ]
        Zero, one, or many entries - one per sign found in the image:
        [
            {
                "label" : str,   one of SIGN_LABELS
                "area"  : int,   that sign's area in pixels
            },
            ...
        ]

    Example call:
    ---
    detections = detect_signs(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    detections = []

    #################### ADD YOUR CODE HERE ####################
    if image is None or image.ndim != 3:
        return detections
    closed, blobs = _find_obstacle_blobs(image)
    detections.extend(_detect_red_signs(image, closed, blobs))   # School, Speed40
    detections.extend(_detect_blue_signs(image))                 # Parking, Hospital
    ############################################################

    return detections


def detect_obstacle(image):
    '''
    Purpose:
    ---
    Detect whether the obstacle is visible in a single image, and how large
    it appears.

    Input Arguments:
    ---
    `image` :   [ numpy.ndarray ]
        A single BGR image, of shape (height, width, 3).

    Returns:
    ---
    `result` :  [ dict ]
        {
            "present" : bool,   True if the obstacle is visible
            "area"    : int,    its area in pixels, 0 if not present
        }

    Example call:
    ---
    result = detect_obstacle(image)

    NOTE:
    ---
    This function must ONLY compute and return the result. Do not call
    cv2.imshow(), cv2.waitKey(), cv2.imwrite() or print() from inside it -
    reading images and reporting results is the evaluator's job, not this
    function's.
    '''

    present = False
    area = 0

    #################### ADD YOUR CODE HERE ####################
    if image is not None and image.ndim == 3:
        _, blobs = _find_obstacle_blobs(image)
        if blobs:
            area = max(a for _, a in blobs)      # the biggest solid red blob
            present = area > 0
    ############################################################

    return {"present": present, "area": int(area) if present else 0}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_signs() or detect_obstacle() - the
# evaluation script only ever calls those two functions. List them in
# the file header too.
# ------------------------------------------------------------------

# ---- tunable constants (globals) ----
MIN_OBS_AREA = 500       # smallest obstacle blob (px); red debug dots are far smaller
MIN_RING_AREA = 450      # smallest red-ring sign (px)
MIN_BLUE_AREA = 120      # smallest blue sign (px)


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _red_mask(hsv):
    lo = cv2.inRange(hsv, (0, 100, 80), (9, 255, 255))
    hi = cv2.inRange(hsv, (171, 100, 80), (180, 255, 255))
    return cv2.bitwise_or(lo, hi)


def _find_obstacle_blobs(image):
    """Returns (closed_red_mask, list of (opened_blob_mask, area_px)) of solid red blobs."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = _red_mask(hsv)
    closed = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    k = _odd(max(5, round(0.01 * min(image.shape[:2]))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ker)

    blobs = []
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_OBS_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        extent = a / float(w * h)
        hull_a = cv2.contourArea(cv2.convexHull(c))
        solidity = a / hull_a if hull_a > 0 else 0
        aspect = w / float(h)
        # solid box-like blob: not a ring (sign rim) and not a plus (red cross)
        if extent < 0.58 or solidity < 0.78 or not (0.30 < aspect < 3.0):
            continue
        blob = np.zeros(opened.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        # solid check: red pixels must fill the blob
        if cv2.countNonZero(cv2.bitwise_and(closed, blob)) < 0.8 * a:
            continue
        # grow back by the opening size so thin edges are measured too
        grown = cv2.dilate(blob, ker)
        area = cv2.countNonZero(cv2.bitwise_and(closed, grown))
        blobs.append((grown, int(area)))
    return closed, blobs


def _detect_red_signs(image, closed, blobs):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = closed.copy()
    for grown, _ in blobs:                       # remove obstacle so it cannot merge with a sign
        red[cv2.dilate(grown, np.ones((3, 3), np.uint8)) > 0] = 0
    white = cv2.bitwise_and(cv2.inRange(hsv[:, :, 1], 0, 70), cv2.inRange(hsv[:, :, 2], 150, 255))

    found = []
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a_f = cv2.contourArea(c)
        if a_f < MIN_RING_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.6 < w / float(h) < 1.7):
            continue
        filled = np.zeros(red.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        # a sign rim is hollow: few red pixels compared with the filled outline
        if cv2.countNonZero(cv2.bitwise_and(red, filled)) / a_f > 0.7:
            continue
        # inside of the rim must be mostly white
        if cv2.countNonZero(cv2.bitwise_and(white, filled)) / a_f < 0.3:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        circ = a_f / (np.pi * r * r) if r > 0 else 0
        try:
            tri_a, _ = cv2.minEnclosingTriangle(c.astype(np.float32))
            tri = a_f / tri_a if tri_a > 0 else 0
        except cv2.error:
            tri = 0
        if circ >= 0.65 and circ > tri:
            found.append({"label": "Speed40", "area": int(round(a_f))})
        elif tri >= 0.70:
            found.append({"label": "School", "area": int(round(a_f))})
    return found


def _detect_blue_signs(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (98, 150, 120), (130, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    letter_px = cv2.bitwise_and(cv2.inRange(hsv[:, :, 1], 0, 110), cv2.inRange(hsv[:, :, 2], 110, 255))

    found = []
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a_f = cv2.contourArea(c)
        if a_f < MIN_BLUE_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.5 < w / float(h) < 2.0) or a_f / float(w * h) < 0.55:
            continue
        filled = np.zeros(blue.shape, np.uint8)
        cv2.drawContours(filled, [c], -1, 255, -1)
        e = max(1, int(0.1 * min(w, h)))
        inner = cv2.erode(filled, np.ones((2 * e + 1, 2 * e + 1), np.uint8))
        letter = cv2.bitwise_and(letter_px, inner)
        frac = cv2.countNonZero(letter) / a_f
        if not (0.08 < frac < 0.65):
            continue
        ys, xs = np.nonzero(letter)
        sub = letter[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        hh, ww = sub.shape
        if hh < 4 or ww < 4:
            continue
        # H has a bar at the bottom-right; P is empty there
        reg = sub[int(0.7 * hh):, int(0.55 * ww):]
        lr = float(reg.mean()) / 255.0
        if lr > 0.20:
            label = "Hospital"
        elif lr < 0.08:
            label = "Parking"
        else:                                    # unsure: P has a hole (its bowl), H has none
            pad = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            cs, hier = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            holes = 0
            if hier is not None:
                for i, ct in enumerate(cs):
                    if hier[0][i][3] != -1 and cv2.contourArea(ct) >= max(2, 0.01 * hh * ww):
                        holes += 1
            label = "Parking" if holes > 0 else "Hospital"
        found.append({"label": label, "area": int(round(a_f))})
    return found


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_signs(detections, image_name):
    '''
    Purpose:
    ---
    Check that detect_signs() returned the expected structure and normalise
    it, so a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `detections` :  [ object ]     whatever detect_signs() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ list of dict ]   [{"label": str, "area": int}, ...]
    '''
    where = "detect_signs() on {}".format(image_name)

    if not isinstance(detections, list):
        raise TypeError("{} must return a list, got {}".format(where, type(detections).__name__))

    clean = []
    for index, entry in enumerate(detections):
        entry_where = "{}, entry {}".format(where, index)

        if not isinstance(entry, dict):
            raise TypeError("{} must be a dict, got {}".format(entry_where, type(entry).__name__))

        missing = {"label", "area"} - set(entry.keys())
        if missing:
            raise ValueError("{} is missing the key(s): {}".format(entry_where, ", ".join(sorted(missing))))

        label = entry["label"]
        if label not in SIGN_LABELS:
            raise ValueError("{} returned label = '{}', expected one of {}".format(
                entry_where, label, ", ".join(SIGN_LABELS)))

        area = _validate_area(entry["area"], entry_where)
        clean.append({"label": label, "area": area})

    return clean


def validate_obstacle(result, image_name):
    '''
    Purpose:
    ---
    Check that detect_obstacle() returned the expected structure and
    normalise it.

    Input Arguments:
    ---
    `result` :      [ object ]     whatever detect_obstacle() returned
    `image_name` :  [ str ]        name of the image, used in error messages

    Returns:
    ---
    `clean` :       [ dict ]       {"present": bool, "area": int}
    '''
    where = "detect_obstacle() on {}".format(image_name)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"present", "area"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    present = result["present"]
    if not isinstance(present, (bool, np.bool_)):
        raise TypeError("{} returned present of type {}, expected a bool".format(
            where, type(present).__name__))
    present = bool(present)

    area = _validate_area(result["area"], where)
    if not present and area != 0:
        raise ValueError("{} returned present=False but area={}, expected 0".format(where, area))

    return {"present": present, "area": area}


def _validate_area(area, where):
    if isinstance(area, bool) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned area of type {}, expected a number".format(
            where, type(area).__name__))
    area = int(round(float(area)))
    if area < 0:
        raise ValueError("{} returned a negative area: {}".format(where, area))
    return area
