'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1B of Niti Vahan (NV) Theme.
*
*  Filename:         path_tracking.py
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
# Filename:         path_tracking.py
# Functions:        ackermann_wheel_angles, compute_steering


####################### IMPORT MODULES #######################
import argparse
import csv
import math
import queue
import sys
import threading
import time
##############################################################


#################### VEHICLE CONSTANTS #######################
WHEELBASE = 0.120           # L: distance between front and rear axle centrelines
TRACK_WIDTH = 0.110         # W: distance between left and right wheel centre
WHEEL_OFFSET = 0.0275       # O: distance between kingpin axis and wheel centre.
##############################################################


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

# Stanley controller parameters
STANLEY_K = 0.40
STANLEY_SOFTENING = 0.08
HEADING_GAIN = 1.0


def ackermann_wheel_angles(delta):
    '''
    Convert one virtual steering angle into the two front-wheel
    steering angles using Ackermann geometry.

    Positive delta = left turn.
    '''

    delta = float(delta)

    # Straight-line case
    if abs(delta) < 1e-9:
        return 0.0, 0.0

    # Turning radius corresponding to the virtual steering angle
    turn_radius = WHEELBASE / math.tan(abs(delta))

    # Effective half-track between the steering axes
    effective_half_track = (TRACK_WIDTH / 2.0) - WHEEL_OFFSET

    # Inner and outer radii
    inner_radius = turn_radius - effective_half_track
    outer_radius = turn_radius + effective_half_track

    # Prevent invalid geometry
    inner_radius = max(inner_radius, 1e-6)

    # Ackermann wheel angles
    inner_angle = math.atan(WHEELBASE / inner_radius)
    outer_angle = math.atan(WHEELBASE / outer_radius)

    if delta > 0.0:
        # Positive = left turn
        left_wheel_angle = inner_angle
        right_wheel_angle = outer_angle
    else:
        # Negative = right turn
        left_wheel_angle = -outer_angle
        right_wheel_angle = -inner_angle

    return float(left_wheel_angle), float(right_wheel_angle)


def compute_steering(target_y, current_values):
    '''
    Stanley lane tracking controller.
    '''

    y = float(current_values["y"])
    yaw = float(current_values["yaw"])
    speed = float(current_values["speed"])

    # Task-defined cross-track error
    cross_track_error = y - float(target_y)

    # Task-defined heading error
    heading_error = -yaw

    # Prevent division by an extremely small speed
    speed = max(speed, 0.0)

    # Stanley lateral correction
    lateral_correction = math.atan2(
        STANLEY_K * cross_track_error,
        speed + STANLEY_SOFTENING
    )

    # Final steering command
    steering = (
        HEADING_GAIN * heading_error
        + lateral_correction
    )

    return float(steering)


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

VEHICLE_PATH = "/Niti_Vahan"
STEER_LEFT_JOINT = "/Niti_Vahan/steeringLeft"
STEER_RIGHT_JOINT = "/Niti_Vahan/steeringRight"
DRIVE_LEFT_JOINT = "/Niti_Vahan/motorLeft"
DRIVE_RIGHT_JOINT = "/Niti_Vahan/motorRight"

WHEEL_RATE = 1.3                        # rad/s, constant drive speed
MAX_STEER = math.radians(30.0)          # steering limit, radians
RUN_TIME = 120.0                        # s of simulated time, every run

# The vehicle drives along world -x, so its left-hand side faces -y.
LANE_Y = {"L": 0.0, "R": 0.2}
DEFAULT_LANE = "L"
DEFAULT_CSV = "trajectory.csv"
DEFAULT_LOG_RATE = 10.0                 # Hz, rows written to the CSV

CSV_COLUMNS = ("t", "x", "y", "yaw", "speed", "target_y",
               "steering", "left_wheel", "right_wheel")


def _stdin_reader(commands):
    '''Read the terminal in a background thread so the control loop never blocks.'''
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put("q")           # stdin closed - treat it as "stop"


def read_target(commands, lane):
    '''Apply every lane command typed since the last step. Returns (lane, keep_running).'''
    while True:
        try:
            command = commands.get_nowait().strip().upper()
        except queue.Empty:
            return lane, True

        if not command:
            continue
        if command in ("Q", "QUIT", "EXIT"):
            return lane, False
        if command not in LANE_Y:
            print("  ignored %r - type L, R or q" % command)
            continue
        if command != lane:
            lane = command
            print("  lane -> %s  (y = %.2f m)" % (lane, LANE_Y[lane]))


def parse_schedule(text):
    '''Turn "12:R,45:L" into [(12.0, "R"), (45.0, "L")], sorted by simulated time.'''
    schedule = []
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        when, _, side = entry.partition(":")
        side = side.strip().upper()
        if side not in LANE_Y:
            raise ValueError("%r: lane must be L or R" % entry)
        schedule.append((float(when), side))
    return sorted(schedule)


def heading(matrix):
    '''Heading relative to the road, in radians, from the vehicle's pose matrix.

    The body's Euler angles sit at beta = -90 degrees, where the decomposition
    is degenerate and the reported angles jump around, so the heading comes
    from the rotation matrix instead. Column 3 is the local +z axis - the
    direction the vehicle faces - and the road runs along world -x.
    '''
    yaw = math.atan2(matrix[6], matrix[2]) - math.pi
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def run_session(sim, commands, csv_path, log_rate, schedule=None):
    '''Drive one RUN_TIME-second run, logging the trajectory.

    With a `schedule` the lane changes at the simulated times it lists;
    without one it comes from whatever is typed at the terminal.
    '''
    vehicle = sim.getObject(VEHICLE_PATH)
    steer_left = sim.getObject(STEER_LEFT_JOINT)
    steer_right = sim.getObject(STEER_RIGHT_JOINT)
    drive_left = sim.getObject(DRIVE_LEFT_JOINT)
    drive_right = sim.getObject(DRIVE_RIGHT_JOINT)

    log_interval = 1.0 / log_rate
    next_log = 0.0
    lane = DEFAULT_LANE
    pending = list(schedule or ())
    rows = 0

    handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(CSV_COLUMNS)

    sim.startSimulation()
    try:
        t0 = sim.getSimulationTime()
        prev_t = t0

        while True:
            now = sim.getSimulationTime()
            elapsed = now - t0
            if elapsed >= RUN_TIME:
                print("run complete at %.1f s" % elapsed)
                break

            while pending and elapsed >= pending[0][0]:
                _, lane = pending.pop(0)
                print("  t=%6.2f s  lane -> %s  (y = %.2f m)"
                      % (elapsed, lane, LANE_Y[lane]))

            lane, keep_running = read_target(commands, lane)
            if not keep_running:
                print("stopping")
                break

            dt = now - prev_t
            if dt <= 0.0:
                sim.step()
                continue
            prev_t = now

            pos = sim.getObjectPosition(vehicle, -1)
            yaw = heading(sim.getObjectMatrix(vehicle, -1))
            linear, _ = sim.getObjectVelocity(vehicle)
            target_y = LANE_Y[lane]

            steering = compute_steering(target_y, {
                "y": pos[1],
                "yaw": yaw,
                "speed": math.hypot(linear[0], linear[1]),
                "dt": dt,
                "t": elapsed,
            })
            steering = max(-MAX_STEER, min(MAX_STEER, steering))
            left_angle, right_angle = ackermann_wheel_angles(steering)

            sim.setJointTargetPosition(steer_left, left_angle)
            sim.setJointTargetPosition(steer_right, right_angle)
            sim.setJointTargetVelocity(drive_left, WHEEL_RATE)
            sim.setJointTargetVelocity(drive_right, WHEEL_RATE)

            if elapsed >= next_log:
                writer.writerow([
                    "%.3f" % elapsed, "%.4f" % pos[0], "%.4f" % pos[1],
                    "%.4f" % yaw, "%.4f" % math.hypot(linear[0], linear[1]),
                    "%.2f" % target_y, "%.4f" % steering,
                    "%.4f" % left_angle, "%.4f" % right_angle,
                ])
                rows += 1
                next_log += log_interval

            sim.step()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        handle.close()
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.05)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Task 1B - drive the Ackermann vehicle to the lane you ask for."
    )
    parser.add_argument(
        "--out", default=DEFAULT_CSV,
        help="Where to write the trajectory CSV (default: %s)." % DEFAULT_CSV,
    )
    parser.add_argument(
        "--log-rate", type=float, default=DEFAULT_LOG_RATE,
        help="Rows per second written to the CSV (default: %g Hz)." % DEFAULT_LOG_RATE,
    )
    parser.add_argument(
        "--schedule",
        help="Lane changes at fixed SIMULATED times, e.g. '15:R,60:L,95:R'. "
             "The evaluation supplies this so every team drives the same run; "
             "without it, the lane is whatever you type.",
    )
    args = parser.parse_args()
    schedule = parse_schedule(args.schedule) if args.schedule else None

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sim = RemoteAPIClient().require("sim")
    sim.setStepping(True)

    commands = queue.Queue()
    if schedule:
        # No terminal reader: the schedule is the only thing that may change lane.
        print("lane %s (y = %.2f m) - fixed schedule: %s"
              % (DEFAULT_LANE, LANE_Y[DEFAULT_LANE],
                 ", ".join("%gs->%s" % entry for entry in schedule)))
    else:
        threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()
        print("lane %s (y = %.2f m) - type L or R and press Enter to change lane, q to stop"
              % (DEFAULT_LANE, LANE_Y[DEFAULT_LANE]))

    rows = run_session(sim, commands, args.out, args.log_rate, schedule)
    print("wrote %d rows to %s" % (rows, args.out))


if __name__ == "__main__":
    main()
