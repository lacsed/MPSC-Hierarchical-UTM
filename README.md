# MPSC-Hierarchical-UTM

Hierarchical UTM framework for multi-UAV traffic management based on Supervisory Control Theory (SCT), Discrete Event Systems (DES), and Model Predictive Supervisory Control (MPSC) implemented in ROS 2 and Gazebo.

The project integrates:

* Hierarchical DES/SCT supervision
* Distributed UAV supervisors via event relabeling
* Online MPSC optimization using MILP
* Multi-UAV task allocation
* Dynamic geofencing
* Shared-resource mutual exclusion
* Battery-aware operation
* Gazebo-based urban airspace simulation
* Scalability analysis against centralized monolithic supervision




https://github.com/user-attachments/assets/6f5810fa-b866-4322-95fc-ea046b043b89


https://github.com/user-attachments/assets/747e9b3f-a7f0-40e1-9d76-cadace233e04


---

# Overview

This repository implements the hierarchical UTM architecture proposed in the associated research work.

The framework models:

* UAVs
* Airspace corridors
* Vertiports
* Charging stations
* Suppliers
* Clients
* Geofenced regions

as deterministic finite automata (DFA) coordinated through SCT-based supervisory control and local receding-horizon optimization.

The architecture combines:

1. A global UTM supervisory layer responsible for:

   * shared-resource coordination,
   * conflict prevention,
   * geofencing,
   * event authorization.

2. Local UAV supervisors responsible for:

   * admissible event execution,
   * finite-horizon planning,
   * battery-aware decision making,
   * task execution.


---

# Repository Structure

```text
MPSC-Hierarchical-UTM/
├── ros2_ws/
│   ├── src/
│   │   ├── utm_control/
│   │   ├── utm_fleet/
│   │   └── utm_graph/
├── scripts/
│   ├── assets/
│   ├── experiments/
│   └── gz_world_out/
└── README.md
```

---

# Core Packages

## `utm_control`

Contains:

* graph abstractions,
* DES routing conversion,
* problem parameterization,
* optimization interfaces.

---

## `utm_fleet`

Contains:

* local UAV supervisors,
* MPSC planners,
* MILP optimizers,
* UAV hardware abstraction,
* fleet controller,
* UTM supervisor.


---

## `utm_graph`

Contains:

* graph loading,
* graph models,
* node typing,
* CSV parsing utilities.

---

# System Architecture

The framework follows a hierarchical structure:

```text
                +----------------------+
                |    UTM Supervisor    |
                |----------------------|
                | conflict management  |
                | geofencing           |
                | mutex coordination   |
                | prohibited events    |
                +----------+-----------+
                           |
          -----------------------------------------
          |                  |                   |
+---------v------+ +---------v------+ +----------v-----+
| UAV Supervisor | | UAV Supervisor | | UAV Supervisor |
|----------------| |----------------| |----------------|
| local SCT      | | local SCT      | | local SCT      |
| local MPSC     | | local MPSC     | | local MPSC     |
| MILP planning  | | MILP planning  | | MILP planning  |
+----------------+ +----------------+ +----------------+
```

---

# Dependencies

## Operating System

Ubuntu 20.04 recommended.

---

## ROS 2

ROS 2 Galactic.

---

## Gazebo

Gazebo 11.

---

## Python

Python 3.8+.

---

## Required Python Libraries

Install dependencies:

```bash
pip install \
    networkx \
    numpy \
    scipy \
    pandas \
    matplotlib \
    pillow \
    opencv-python \
    gurobipy

pip install https://github.com/lacsed/UltraDES-Python/releases/download/0.0.5/ultrades_python-0.0.5-py3-none-any.whl
```

---

# Build Instructions

## Clone repository

```bash
git clone <repository-url>

cd MPSC-Hierarchical-UTM
```

---

## Build ROS 2 workspace

```bash
cd ros2_ws

colcon build --symlink-install
```

---

## Source workspace

```bash
source install/setup.bash
```

---


# World Generation

The project generates urban environments directly from palette-coded map images. The input map is not interpreted only as a visual texture: its colors define semantic regions of the environment. During generation, the script segments the image by its predefined color palette and extracts masks for buildings, roads, and parks. Therefore, the map image must preserve the expected color code; anti-aliased, compressed, or visually similar colors may produce incorrect segmentation.

The generator uses the following semantic color code in the input map:

```text
gray regions  / RGB (128, 128, 128) / HEX #808080 -> buildings
white regions / RGB (255, 255, 255) / HEX #FFFFFF -> roads
green regions / RGB (0, 255, 0)     / HEX #00FF00 -> parks
black regions / RGB (0, 0, 0)       / HEX #000000 -> background
```

Example:

```bash
cd scripts

python3 gen_world_from_image_gz.py \
  6 1 1 2 4 \
  --map ./assets/finalmap.png \
  --out gz_world_out \
  --res 0.2 \
  --seed 77
```

Generated outputs include:

```text
graph_nodes.csv
graph_edges.csv
utm_world.sdf
heightmap.png
```


# Example Experiment Flow

## 1. Generate world

```bash
python3 gen_world_from_image_gz.py ...
```

---

## 2. Launch Gazebo

```bash
ros2 launch gazebo_ros gazebo.launch.py \
  world:=.../utm_world.sdf
```

---

## 3. Launch UTM supervisor

```bash
ros2 run utm_fleet utm_supervisor ...
```

---

## 4. Launch fleet controller

```bash
ros2 run utm_fleet fleet_controller ...
```

---

## 5. Publish tasks

```bash
ros2 topic pub /task_todo std_msgs/msg/String \
"{data: 'SUPPLIER_000,CLIENT_000'}" --once
```

---

## 6. Analyze results

```bash
python3 scripts/experiments/analyze_computational_performance.py \
  --run-dir <run-dir>
```

---

# Current Limitations

Current implementation limitations include:

* simplified UAV kinematics,
* direct Gazebo state manipulation,
* no full UAV dynamics,
* no realistic communication delays,
* simplified battery model,

---

# Future Work

Planned extensions include:

* full UAV dynamic models,
* ROS 2 typed interfaces,
* launch-based experiment orchestration,
* temporal corridor reservation,


