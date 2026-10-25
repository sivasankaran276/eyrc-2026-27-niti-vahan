# e-Yantra Robotics Competition 2026–27 — Niti Vahan

**Team ID:** 4027  
**Repository:** `eyrc-2026-27-niti-vahan`  
**Focus Areas:** Autonomous Driving · Computer Vision · Control Systems · Robotics Software

## About the Project

This repository contains my implementations, evaluation artifacts, and selected candidate versions developed for the Niti Vahan theme of the e-Yantra Robotics Competition 2026–27.

The work progresses from steering geometry and vehicle control to camera-based lane detection, autonomous lane following, road-sign classification, obstacle detection, and autonomous parking.

The repository preserves Python source code, submission ZIP files, evaluation JSON files, trajectory data, and selected candidate implementations to support traceability and reproducibility.

## Task Overview

| Task | Description | Recorded Result |
|---|---|---:|
| Task 1A | Ackermann Steering Geometry | 20/20 |
| Task 1B | Path Tracking and Lane-Change Control | 30/30 |
| Task 1C | Vision-Based Lane Detection | 49.35/50 — earlier reported private result |
| Task 2A | Lane Following and Stop Handling | 39.74/40 |
| Task 2B | Road-Sign and Obstacle Detection | 40/40 |
| Task 2C | ArUco-Based Autonomous Parking | 20/20 |

**Task 1C result note:** An earlier private evaluation was reported as 49.35/50, with a public-set score of 45.38/50. A later result displayed 49.40/50 but was marked “Late, not eligible.” The eligibility and relationship between these attempts require confirmation from the competition portal.

Scores above are based on the reported evaluation results and are not independently certified by this repository. Task 2C's manual simulation results are not official portal scores.

---

## Task 1A — Ackermann Steering Geometry

**Primary source:** `Task1A/NV_Task1A.py`

### Objective

Convert a virtual steering angle into the corresponding left and right front-wheel angles using Ackermann steering geometry.

### Key Concepts

- Ackermann steering geometry
- Wheelbase and track-width relationships
- Steering-angle conversion
- Straight-line steering behaviour
- Numerical correctness and edge-case handling

### Files

- `NV_Task1A.py` — task implementation
- `ackermann_steering.py` — steering implementation and local test file

### Evaluation

The implementation is intended to handle the zero-steering case and return geometrically correct wheel angles for positive and negative steering inputs.

**Recorded score:** Not included in this README.

---

## Task 1B — Path Tracking and Lane-Change Control

**Primary source:** `Task1B/path_tracking.py`

### Objective

Control the vehicle's steering in response to a target lane position and vehicle-state measurements, while following a prescribed lane-change schedule.

### Key Concepts

- Vehicle path tracking
- Lateral and heading errors
- Steering control
- Ackermann steering integration
- Lane-change convergence
- Settling time and overshoot
- Simulation trajectory evaluation

### Files

- `path_tracking.py` — controller implementation
- `NV_Task1B.zip` — packaged submission
- `trajectory.csv` — recorded vehicle trajectory
- `result-2026-NV-1b-20260928.json` — evaluation metadata

### Recorded Run Metadata

The included result JSON records a 120-second simulated run, 2,400 trajectory rows, and two lane changes.

The source Python file and trajectory CSV were checked against their counterparts inside the submission ZIP using SHA-256 hashes; both pairs matched exactly.

**Recorded score:** Not included in this README because the available result JSON does not contain the portal score.

---

## Task 1C — Vision-Based Lane Detection

**Primary source:** `Task1C/lane_detection.py`

### Objective

Identify the vehicle's lane and estimate the lane centre from camera frames.

### Key Concepts

- Image processing
- Lane identification
- Lane-centre estimation
- Temporal consistency
- Robustness across video frames
- Computer-vision evaluation on public and private clips

### Files

- `lane_detection.py` — packaged lane-detection implementation
- `NV_Task1C.zip` — submission archive
- `NV_Task1C.json` — public-set evaluation result
- `candidates/lane_detection_V11_TEMPORAL_49_35_PRIVATE.py` — selected candidate version
- `candidates/lane_detection_V12_48_09_PUBLIC.py` — selected candidate version

### Reported Evaluation

**Earlier reported private result: 49.35/50**

- Lane identification: 19.94/20
- Lane centre: 29.41/30
- Lane identification accuracy: 99.9%
- Lane-centre accuracy within tolerance: 98.0%
- Mean centre error: 8.3 pixels
- Frames scored: 3,631 out of 4,000
- Ambiguous frames excluded: 369
- Reported evaluation time: 51 seconds

The accompanying local JSON records a separate public-set result of **45.38/50**.

The current source file and the script extracted from `NV_Task1C.zip` have matching SHA-256 hashes. The local package currently matches the V12 public candidate. The exact link between the V11 candidate and the earlier private portal result has not yet been established.

---

## Task 2A — Lane Following and Stop Handling

**Primary source:** `Task2A/task2a.py`

### Objective

Implement lane-following behaviour in the Niti Vahan simulation, maintaining lane position and responding to the evaluator's stop requests.

### Key Concepts

- Autonomous lane following
- Steering control
- Lane-centre tracking
- Stop-request handling
- Simulation-based evaluation
- Trajectory analysis

### Recorded Result

**39.74/40**

The reported portal evaluation recorded:

- Outer-lane in-lane percentage: 98.7%
- Outer-lane mean error: 1.6 cm
- Inner-lane in-lane percentage: 99.2%
- Inner-lane mean error: 1.3 cm
- Stops: 6/6
- Laps: 8/8

### Files

- `task2a.py` — task implementation
- `NV_Task2A.json` — recorded evaluation result
- `NV_Task2A_run.csv` — recorded run data

The score and metrics above are reproduced from the previously reported portal result.

---

## Task 2B — Road-Sign and Obstacle Detection

**Primary source:** `Task2B/sign_obstacle_detection.py`

### Objective

Detect and classify supported road signs and determine obstacle presence and apparent area from images.

### Supported Road Signs

- School
- Speed40
- Parking
- Hospital

### Key Concepts

- OpenCV image processing
- HSV colour segmentation
- Morphological operations
- Contour analysis
- Geometric shape filtering
- Road-sign classification
- Obstacle-presence detection
- Pixel-area estimation

### Recorded Result

**40/40 — full marks**

| Evaluation Category | Score |
|---|---:|
| Sign labels | 15/15 |
| Sign area accuracy | 10/10 |
| Obstacle presence | 10/10 |
| Obstacle area accuracy | 5/5 |
| **Total** | **40/40** |

### Files

- `sign_obstacle_detection.py` — winning source implementation
- `NV_Task2B.zip` — submission archive
- `SHA256SUMS.txt` — recorded file checksums
- `PORTAL_RESULT.txt` — saved score summary

The final source and packaged submission were checked for ZIP integrity. The reported private portal result was 40/40.

---

## Task 2C — ArUco-Based Autonomous Parking

**Primary source:** `Task2C/task2c.py`

### Objective

Develop a parking controller that guides the Niti Vahan vehicle into a selected parking bay, aligns it with the bay's centre line, and brings it to a complete stop.

The controller uses vehicle-pose information estimated from an ArUco marker by the provided vision pipeline. It returns steering, speed, and parking status to control the manoeuvre.

### Key Concepts

- Autonomous parking
- ArUco marker-based pose estimation
- Lateral position error
- Vehicle heading correction
- Steering control
- Speed regulation
- Marker-visibility handling
- Stopping and final alignment
- CoppeliaSim simulation

### Parking Bays

The task has two parking bays:

- **Bay 0:** The nearer bay, identified by ArUco marker ID 0.
- **Bay 1:** The farther bay, identified by ArUco marker ID 1.

The controller must handle both target bays while keeping the vehicle within the narrow bay boundaries.

### Implementation

The `task2c.py` implementation includes a parking manoeuvre controller and helper logic for steering, speed adjustment, marker-loss handling, and stopping.

The controller is tested in CoppeliaSim using the provided `Task_2C.ttt` scene.

### Manual Simulation Test Results

Both manual simulation runs completed, and the terminal reported that the vehicle parked.

| Test | Result | Frames Processed | Marker Detection |
|---|---|---:|---:|
| Bay 0 | Parked | 158 | 158/158 frames |
| Bay 1 | Parked | 278 | 278/278 frames |

During manual testing, the vehicle was observed to stop inside the bay, straight and centred.

These are **manual simulation results**, not official portal scores. Final parking accuracy and marks must be confirmed through the official e-Yantra evaluation.

### Files

- `task2c.py` — parking controller implementation
- `Task_2C.ttt` — official simulation scene, used locally
- `NV_Task2C_bay0.csv` — official evaluation recording, when generated
- `NV_Task2C_bay1.csv` — official evaluation recording, when generated
- `NV_Task2C.zip` — packaged submission, when generated

The CSV files and submission ZIP should be added only after the official evaluation tool generates them.

**Official portal score:** Pending.

---

## Repository Structure

```text
eyrc-2026-27-niti-vahan/
├── README.md
│
├── Task1A/
│   ├── NV_Task1A.py
│   └── ackermann_steering.py
│
├── Task1B/
│   ├── path_tracking.py
│   ├── NV_Task1B.zip
│   ├── trajectory.csv
│   └── result-2026-NV-1b-20260928.json
│
├── Task1C/
│   ├── lane_detection.py
│   ├── NV_Task1C.zip
│   ├── NV_Task1C.json
│   └── candidates/
│       ├── lane_detection_V11_TEMPORAL_49_35_PRIVATE.py
│       └── lane_detection_V12_48_09_PUBLIC.py
│
├── Task2A/
│   ├── task2a.py
│   ├── NV_Task2A.json
│   └── NV_Task2A_run.csv
│
├── Task2B/
│   ├── sign_obstacle_detection.py
│   ├── NV_Task2B.zip
│   ├── SHA256SUMS.txt
│   └── PORTAL_RESULT.txt
│
└── Task2C/
    └── task2c.py
```

*This tree represents the intended curated repository structure. Supporting files should be checked against the actual GitHub tree.*

## Technology Stack

- **Python** — implementation and control logic
- **OpenCV** — computer-vision operations
- **NumPy** — numerical and array processing
- **CoppeliaSim EDU** — robotics simulation
- **e-Yantra AutoEval** — official task evaluation
- **Git and GitHub** — version control and source-code management

## Reproducibility and Version Control

The repository keeps selected source implementations and supporting submission artifacts together. This helps make it easier to:

- Compare candidate implementations.
- Identify which source file corresponds to a packaged submission.
- Preserve evaluation outputs.
- Verify file identity using SHA-256 checksums.
- Trace changes through Git commit history.

Evaluation results can depend on the test set and grading context. Local public-set results should not be treated as equivalent to private portal scores.

## Competition Integrity

This repository is maintained for personal project documentation, development history, and technical review.

All code sharing and redistribution should follow the applicable e-Yantra competition rules. Keep the repository private unless the competition organizers permit public sharing of the relevant solutions.

## Acknowledgements

Thanks to the e-Yantra Robotics Competition team for providing the Niti Vahan theme, task specifications, and evaluation environment.

---

**Team 4027 — Niti Vahan 2026–27**

*Autonomous systems, computer vision, and robotics software.*
