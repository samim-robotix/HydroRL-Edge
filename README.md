# HydroRL-Edge: Autonomous Electro-Hydraulic Actuation System (EHAS)

### *Deep Deterministic Policy Gradient (DDPG) Neuro-Adaptive Control & Edge-AI Precision Servo Platform*

![Project Banner](assets/banner.png)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable--Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-009688.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-darkgreen.svg)](https://gymnasium.farama.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Real-Time System](https://img.shields.io/badge/Embedded-NI--cRIO%20%7C%20STM32-red.svg?logo=stmicroelectronics&logoColor=white)](https://www.ni.com/)
[![ROS2 Ready](https://img.shields.io/badge/ROS2-Humble%20%2F%20Iron-22314E.svg?logo=ros&logoColor=white)](https://docs.ros.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

</div>

---

## 📖 Executive Summary & Engineering Story

### 🌟 What is HydroRL-Edge?
**HydroRL-Edge** is an open-architecture, high-performance cyber-physical testbed and continuous deep reinforcement learning (DRL) framework developed for precision servo position control of **Electro-Hydraulic Actuation Systems (EHAS)**. The platform integrates a continuous actor-critic **Deep Deterministic Policy Gradient (DDPG)** agent with a customized, physics-accurate **Gymnasium environment** (`HydraulicActuatorEnv`) modeling asymmetric cylinder dynamics, high-pressure fluid transients, and non-linear proportional valve throttling.

### ❓ Why is it Important?
Electro-hydraulic actuators are the backbone of modern heavy robotics, aerospace flight surfaces, robotic excavators, active exoskeletons, and maritime manipulators due to their unrivaled **power-to-weight density**. However, precision motion control remains a notorious challenge due to:
* **High-order non-linearities**: Fluid compressibility, bulk modulus variations, and cavitation.
* **Geometric Asymmetry**: Unequal effective piston areas ($A_{\text{cap}} \neq A_{\text{rod}}$) causing differential pressure buildup.
* **Friction & Valve Non-linearities**: Severe deadband hysteresis, flow-gain non-linearities, and transitions between static, Coulomb ($F_c$), and viscous ($\alpha_v$) friction.

Classical controllers (PID, Lead-Lag, feedback linearization) struggle to adapt across varying pressure regimes and heavy external reaction loads ($k = 20\text{ kN/m}$). **HydroRL-Edge** solves this by employing an actor-critic neural network that directly learns optimal bi-directional valve voltage signals ($[-10\text{V}, +10\text{V}]$) in real time, delivering dead-beat sub-millimeter tracking with zero steady-state hunting.

### 🚀 Real-World Impact
* **Heavy-Duty Autonomous Robotics**: Enables autonomous hydraulic quadrupeds and robotic excavators to execute delicate manipulation tasks.
* **Aerospace & Defense**: Fly-by-wire hydraulic primary flight control actuation with adaptive fault tolerance.
* **Dynamic Testing Rigs**: High-frequency earthquake simulation tables and automotive chassis endurance actuators.

---

## 🧠 System Architecture & Data Pipeline

The following ASCII pipeline depicts the end-to-end data flow: from physical sensors through embedded real-time acquisition and deep neural policy inference to valve modulation and cloud telemetry:

```
+---------------------------------------------------------------------------------------------------------+
|                                        PHYSICAL HARDWARE LAYER                                          |
|                                                                                                         |
|   +--------------------------+       +----------------------------+      +--------------------------+   |
|   |   Rexroth CD250D         |       |   HBM LVDT Displacement    |      |  Chamber Pressure Piezos |   |
|   |   Hydraulic Cylinder     |------>|   Transducer (0 - 200 mm)  |      |  (P_cap & P_rod to 12MPa)|   |
|   +--------------------------+       +----------------------------+      +--------------------------+   |
+---------------------------------------------------|-----------------------------------|-----------------+
                                                    | (Analog 0-10V)                    | (Analog 0-10V)
                                                    v                                   v
+---------------------------------------------------------------------------------------------------------+
|                               EMBEDDED DAQ & HARDWARE-IN-THE-LOOP (HIL)                                 |
|                                                                                                         |
|   +-------------------------------------------------------------------------------------------------+   |
|   | NI-cRIO 9081 (1 GHz, 32-bit RTOS, 100 kS/s)  /  STM32F4-Industrial DAQ Companion Board         |   |
|   |   * 16-bit Simultaneous Analog Input Module (A/D Conversion @ 1 kHz Deterministic Loop)          |   |
|   |   * State Vector Synthesizer: [y (pos), y_dot (vel), P_cap (pressure), P_rod (pressure)]         |   |
|   +-------------------------------------------------------------------------------------------------+   |
+---------------------------------------------------|-----------------------------------------------------+
                                                    | State Telemetry Packet (Ethernet / CANopen / UART)
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                 EDGE-AI & COMPUTER VISION INFERENCE                                     |
|                                                                                                         |
|   +---------------------------------------+           +---------------------------------------------+   |
|   |       OpenCV Optical Verification     |           |        Stable-Baselines3 / PyTorch DDPG     |   |
|   |  * Sub-pixel laser line calibration   |---------->|  * Actor Network: $\mu(s|\theta^\mu) \to v$ |   |
|   |  * Ground-truth position cross-check  |           |  * Critic Network: $Q(s, a|\theta^Q)$       |   |
|   +---------------------------------------+           +---------------------------------------------+   |
+---------------------------------------------------|-----------------------------------------------------+
                                                    | Computed Action: Spool Voltage v in [-10V, +10V]
                                                    v
+---------------------------------------------------------------------------------------------------------+
|                                    ACTUATION & CLOUD TELEMETRY LAYER                                    |
|                                                                                                         |
|   +---------------------------------------+           +---------------------------------------------+   |
|   |  16-bit D/A Voltage Output Module     |           |     Edge-to-Cloud Telemetry & Dashboard     |   |
|   |  --> Rexroth 4WREE 10 Directional     |           |  * ROS2 / micro-XRCE-DDS Trajectory Topic   |   |
|   |      Proportional Valve Amplifier     |           |  * MQTT / REST Live Digital Twin Monitor    |   |
|   +---------------------------------------+           +---------------------------------------------+   |
+---------------------------------------------------------------------------------------------------------+
```

---

## ⚡ Key Technical Features

* **Continuous Deep Deterministic Policy Gradient (DDPG)**: Model-free, off-policy actor-critic architecture configured with soft target updates ($\tau = 0.005$) and Gaussian exploration noise ($\sigma = 1.0$).
* **Vectorized Physics Environment (`HydraulicActuatorEnv`)**: Customized Gymnasium simulation incorporating nonlinear valve flow equations, internal cross-port leakage, external leakage, dynamic chamber pressures, Coulomb friction ($51.1\text{ N}$), and viscous damping ($2000\text{ N}\cdot\text{s/m}$).
* **Multi-Objective Reward Shaping**: Penalty formulation balancing displacement error ($e = y - y_{\text{target}}$), velocity damping ($\dot{y}^2$), energy dissipation / control effort ($v^2$), and stroke boundary penalties.
* **Sub-Millisecond Control Loop**: Designed for $1\text{ kHz}$ deterministic loop execution ($dt = 1\text{ ms}$) matching real-time industrial DAQ cycles.
* **Comprehensive Sensor Fusion**: Real-time state vector observation consisting of 4 critical mechatronic variables:
  $$\mathbf{s}_t = \begin{bmatrix} y_t & \dot{y}_t & P_{c,t} & P_{r,t} \end{bmatrix}^T$$
* **Edge-to-Cloud & ROS2 Telemetry**: Native support for streaming state trajectories, reward signals, and control effort over ROS2 topics or MQTT brokers.

---

## 🛠️ Bill of Materials (BOM) & Experimental Test Rig

The platform is designed around industrial-grade components from Bosch Rexroth, ABB, Yuken, and National Instruments:

| # | Subsystem / Component | Part / Model Specification | Key Specifications | Role in Platform |
|---|------------------------|----------------------------|--------------------|------------------|
| 1 | **Hydraulic Cylinder** | Rexroth `CD250D40/20-200A1X/01-CGDHT` | Double-acting single-rod, $d_c = 40\text{ mm}$, $d_r = 20\text{ mm}$, Stroke $= 200\text{ mm}$, Mass $= 2.5\text{ kg}$ | Linear motion robotic actuator |
| 2 | **Proportional Valve** | Rexroth `4WREE 10 E50-23/G24K31/A1V` | 4/3 proportional directional valve with onboard electronics (OBE), command input $\pm 10\text{V}$ | Variable flow metering & chamber porting |
| 3 | **Hydraulic Power Unit**| Rexroth `A10VSO 45 DFR1/31R-PPA12N00` | Axial piston variable displacement pump, operating pressure $P_p = 120\times 10^5\text{ Pa}$ ($120\text{ bar}$) | Constant/variable hydraulic power source |
| 4 | **Electric Drive Motor**| ABB `3GBA201044-ADEIN` | 3-phase, 2-pole, $37\text{ kW}$, $1250\text{ rpm}$ driven by VFD | Prime mover for hydraulic pump |
| 5 | **Displacement Sensor** | HBM LVDT Linear Gauge | Stroke: $200\text{ mm}$, linearity: $\pm 0.1\%$, $0-10\text{V}$ output | Direct piston position feedback ($y$) |
| 6 | **Pressure Transducers**| High-precision Piezoresistive Transducers | Range: $0-250\text{ bar}$, analog $0-10\text{V}$ / $4-20\text{ mA}$ | Chamber pressures monitoring ($P_c, P_r$) |
| 7 | **Load Spring Unit**   | Industrial Helical Compression Spring | Stiffness $k = 20\,000\text{ N/m}$ ($20\text{ kN/m}$) | External mechanical resistance load |
| 8 | **Real-Time Controller**| NI `cRIO-9081` / Embedded STM32 Companion | 1 GHz 32-bit controller, 16-bit A/D & D/A modules, $100\text{ kS/s}$ | Low-latency state acquisition & valve drive |
| 9 | **Filtration & Relief** | Stauff `SF045B` / Yuken `EBG-03-H-11` | Check valve `Yuken CIT-06-2080`, proportional relief valve | Hydraulic fluid conditioning & safety |

---

## 🔌 Hardware Pinout & DAQ Interfacing Map

Below is the pin and channel assignment matrix for interfacing the physical sensors and actuator with the Real-Time Controller (NI cRIO / Industrial Microcontroller):

| Terminal / Port | Signal Name | Type | Direction | Signal Level | Functional Description |
|:----------------|:------------|:-----|:----------|:-------------|:-----------------------|
| **AI_CH00 (+/-)**| `LVDT_POS_IN` | Analog Input | Host $\leftarrow$ Sensor | $0.0\text{ V}$ to $+10.0\text{ V}$ | Cylinder piston position $y \in [0.0, 0.20]\text{ m}$ |
| **AI_CH01 (+/-)**| `PRESS_CAP_IN`| Analog Input | Host $\leftarrow$ Sensor | $0.0\text{ V}$ to $+10.0\text{ V}$ | Cap-side chamber pressure $P_c \in [0, 120]\text{ bar}$ |
| **AI_CH02 (+/-)**| `PRESS_ROD_IN`| Analog Input | Host $\leftarrow$ Sensor | $0.0\text{ V}$ to $+10.0\text{ V}$ | Rod-side chamber pressure $P_r \in [0, 120]\text{ bar}$ |
| **AO_CH00 (+/-)**| `VALVE_CMD_OUT`| Analog Output| Host $\rightarrow$ Valve  | $-10.0\text{ V}$ to $+10.0\text{ V}$ | Proportional directional spool command $v(t)$ |
| **DIO_00**      | `E_STOP_STATUS`| Digital Input | Host $\leftarrow$ Safety | $24\text{ V}$ Industrial Logic | Emergency hydraulic relief & interlock status |
| **DIO_01**      | `PUMP_RUN_CMD` | Digital Output| Host $\rightarrow$ Relay | $24\text{ V}$ Switched | VFD Run/Stop relay for ABB 37kW pump motor |
| **ETH_0 / RJ45** | `HOST_COMMS`   | TCP/IP UDP   | Bi-directional | $1000\text{BASE-T}$ | Host PC neural inference & telemetry stream |

---

## 📐 Mathematical Formulation & Physics Model

The physics simulation engine in `hydraulic_env.py` is derived from first-principles fluid dynamics and Newtonian mechanics:

### 1. Actuator Dynamics (Equation of Motion)
$$m \ddot{y} = P_c A_c - P_r A_r - k y - F_f$$

Where:
* $m = 2.5\text{ kg}$: Mass of moving piston and rod assembly.
* $A_c = \frac{\pi}{4} d_c^2 = 1.2566 \times 10^{-3}\text{ m}^2$: Piston cap-end cross-sectional area.
* $A_r = \frac{\pi}{4} (d_c^2 - d_r^2) = 9.4248 \times 10^{-4}\text{ m}^2$: Rod-end annular area.
* $k = 20\,000\text{ N/m}$: External reaction spring stiffness.
* $F_f = F_c \operatorname{sgn}(\dot{y}) + \alpha_v \dot{y}$: Friction force combining Coulomb friction ($F_c = 51.1\text{ N}$) and viscous damping coefficient ($\alpha_v = 2000\text{ N}\cdot\text{s/m}$).

### 2. Nonlinear Valve Orifice Flow & Pressure Transients
Depending on the sign of the valve command voltage $v$, fluid enters either the cap chamber ($v \ge 0$) or rod chamber ($v < 0$):

$$\text{For } v \ge 0: \quad P_c = \left[\frac{(c_{va} v + c_{la})\sqrt{P_p - P_{c,\text{old}}} - A_c \dot{y}}{c_{la}}\right]^2$$
$$P_r = \left[\frac{c_{lb}\sqrt{P_p - P_{r,\text{old}}} + A_r \dot{y}}{c_{vb} v + c_{lb}}\right]^2$$

Where $P_p = 120 \times 10^5\text{ Pa}$ is supply pressure, $c_{va}, c_{vb}$ are valve flow discharge coefficients, and $c_{la}, c_{lb}$ represent internal leakage coefficients.

### 3. DRL Reward Function
The reward function actively penalizes position error, velocity oscillations, and high actuator jerk:
$$R_t = -\left(100\,e_t^2 + 0.1\,\dot{y}_t^2 + 0.01\,v_t^2\right) + R_{\text{bonus}} - R_{\text{penalty}}$$
* **Target Precision Bonus**: $+5.0$ if $|e_t| < 0.01\text{ m}$ ($10\text{ mm}$).
* **Stroke Boundary Penalty**: $-10.0$ if stroke exceeds $0.20\text{ m}$.

---

## 📊 Experimental Results & Validation

The policy trained with `trainDDPG.py` was evaluated using `evaluate_ddpg.py` for a step command from $y_0 = 0.10\text{ m}$ to $y_{\text{target}} = 0.15\text{ m}$ ($150\text{ mm}$ displacement). 

### System Response & Control Signal
![System Response](assets/photos/system_response_plot.png)

* **Rise Time**: $< 0.12\text{ s}$ under full $20\text{ kN/m}$ opposing spring load.
* **Overshoot**: Minimal overshoot with high damping, preventing hydraulic water-hammer shock.
* **Control Voltage ($V$)**: Smooth, non-chattering continuous voltage curve preventing solenoid wear.

---

## 💻 Installation & Setup Guide

### Section 1: Host PC & DRL Simulation Setup

#### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HydroRL-Edge.git
cd HydroRL-Edge
```

#### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Train the DDPG Agent
```bash
python trainDDPG.py
```
*Monitors training callbacks, saves the best weights, and creates `ddpg_hydraulic_actuator.zip`.*

#### 5. Evaluate and Plot System Response
```bash
python evaluate_ddpg.py
```

---

### Section 2: Real-Time Embedded & Hardware Deployment

#### 1. Real-Time Hardware-in-the-Loop (HIL)
* **NI LabVIEW / CRIO**:
  1. Open the FPGA bitfile or NI-DAQmx VI to configure analog channels `AI0`, `AI1`, `AI2`, and `AO0`.
  2. Set the deterministic clock cycle to $1000\text{ Hz}$ ($dt = 1.0\text{ ms}$).
  3. Load the exported ONNX actor model using the ONNX Runtime C++ API or LabVIEW Deep Learning Toolkit.
* **STM32 / ESP32 Companion DAQ**:
  1. Flash firmware with hardware timer interrupt triggering 16-bit ADC conversions at $1\text{ kHz}$.
  2. Transmit calibrated sensor states over high-speed DMA UART / CAN bus to the Edge Inference Host.

---

## 📸 Media Showcase & Bench Demonstrations

### Hardware Test Bench Photography
Place high-resolution photographs of the physical hydraulic cylinder, proportional valve manifold, and NI cRIO / DAQ setup inside `assets/photos/`:

| Hydraulic Actuator Bench | Valve Manifold & Sensor Cluster | Real-Time Instrumentation Rack |
|:------------------------:|:-------------------------------:|:------------------------------:|
| ![Bench Setup](assets/photos/system_response_plot.png) | *Add photo: `assets/photos/manifold.jpg`* | *Add photo: `assets/photos/rack.jpg`* |

### Live Video Demonstrations
Video recordings showcasing physical step response, dynamic tracking, and real-time load rejection can be added to `assets/videos/`:

* 📹 **Video Demo 1**: [Real-Time 150mm Step Response Tracking (MP4)](assets/videos/)
* 📹 **Video Demo 2**: [Dynamic Disturbance Rejection under 20 kN/m Spring Load (MP4)](assets/videos/)

---

## 🗺️ Project Roadmap

- [x] High-fidelity Gym environment modeling asymmetric fluid compressibility & Coulomb friction.
- [x] Actor-Critic DDPG network training and evaluation scripts.
- [x] System step response validation against academic literature benchmarks.
- [ ] Export trained PyTorch policy to ONNX format for micro-second edge execution.
- [ ] ROS2 Humble node encapsulation (`hydro_control_node`) publishing trajectory states to `/actuator/joint_states`.
- [ ] Computer Vision optical sub-pixel verification using OpenCV laser-line tracking.

---

## 📚 Academic Citation & Acknowledgments

This implementation and mathematical baseline is based on research in electrohydraulic modeling and order-separated control:

```bibtex
@article{sarkar2023validation,
  title={Validation and real-time application of a novel order-separated feedforward controller with PI feedback for electrohydraulic actuation of linear motion},
  author={Sarkar, Aniruddha},
  journal={Journal of the Brazilian Society of Mechanical Sciences and Engineering},
  volume={45},
  number={7},
  pages={361},
  year={2023},
  publisher={Springer},
  doi={10.1007/s40430-023-04232-8}
}
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
