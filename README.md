# Interference Characterisation and Localisation for FMCW mmWave Radars

![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-blue)
![Platform](https://img.shields.io/badge/Hardware-TI%20IWR6843ISK-orange)
![Thesis](https://img.shields.io/badge/Thesis-Master%27s-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Master's Thesis — Embedded Systems Engineering**  
> Fachhochschule Dortmund | Department of Embedded Systems Engineering 
> Author: Md Abul Khair | Matriculation: 7207060

---

## Overview

This repository contains the MATLAB implementation developed for 
the Master's thesis:

**"Interference Characterization and Localization for FMCW 
mmWave Radars"**

The project develops a complete eight-stage signal processing 
pipeline for blind interference characterisation and target 
classification using the Texas Instruments IWR6843ISK 
millimetre-wave radar platform. The pipeline operates without 
any prior knowledge of the aggressor radar's chirp parameters, 
timing, or spatial position.

### Key Contributions

- ✅ Frame-level interference spike detection using local 
     neighbourhood power comparison
- ✅ Dual-path CA-CFAR detection with DC notch architecture 
     for static object detection
- ✅ Four-class sequential rule-based classification: 
     static / moving / interference / noise
- ✅ Two-tier DBSCAN clustering: confirmed vs weak interference
- ✅ Exploratory bearing estimation from interference cluster 
     azimuth angles

### Key Findings

| Finding | Result |
|---|---|
| Interference declaration accuracy | 100% within FoV, 0 false declarations |
| Static detection (256-sample) | 28--43 dB SNR |
| Interference artefact velocity (128-sample) | ±7.58 m/s |
| Interference artefact velocity (256-sample) | 1.65--3.08 m/s |
| Bearing estimation consistency | Circular std 0.7°--1.4° |

---

## Hardware

| Component | Specification |
|---|---|
| Observer radar | TI IWR6843ISK |
| Data capture | TI DCA1000EVM |
| Interferer radar | TI IWR6843AOPEVM |
| Observer bandwidth | 383.77 MHz (128-sample) / 767.54 MHz (256-sample) |
| Interferer bandwidth | ~2.6 GHz |
| Operating frequency | 60 GHz |

---


## Scripts

### 1. `radar_data_analyser.m` — Parameter Analyser

Run this **first** to study your dataset before running the 
main pipeline.

**Purpose:**
- Inspect Range-Doppler map behaviour
- Observe noise and interference characteristics
- Tune CFAR, SNR, velocity, and DBSCAN parameters
- Evaluate detection behaviour across frames

**Parameters analysed:**
- CFAR sensitivity and training cell count
- SNR thresholds (noise gate, static min, moving min)
- Velocity thresholds (static max, moving range)
- DBSCAN epsilon and MinPts
- Interference spike threshold (δ = 5 dB)
- Range gate limits

---

### 2. `Final_radar_detection_separator_with_angle.m` — Main Pipeline

The complete eight-stage processing pipeline.

**Pipeline stages:**
1. Hardware parameter derivation from config
2. Frame-level interference spike detection
3. CA-CFAR detection with dual-path DC notch
4. Primary rule-based classification (static / moving / noise)
5. Secondary interference separation
6. DBSCAN spatial clustering
7. Cluster centroid and statistics extraction
8. Visualisation and bearing estimation

**Output figures generated:**
- Frame power timeline with spike markers
- Range vs velocity scatter plot
- X/Y spatial detection map
- 3D detection map
- SNR distribution histogram
- Range-Doppler map (single frame)
- Polar bearing estimation plot

---

## How to Use

### Requirements

- MATLAB R2021a or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### Step 1 — Place Radar Data

Put recorded `.bin` file inside the project folder or 
a `data/` subdirectory.

### Step 2 — Run Analyser First

```matlab
radar_data_analyser.m
```

Use the generated plots and printed statistics to understand 
the dataset and tune parameters before running the main script.

### Step 3 — Configure Main Script

Open `Final_radar_detection_separator_with_angle.m` and set 
these parameters at the top of the file:

```matlab
% File and configuration
testFile        = 'your_file.bin';   % path to your .bin file
numADCSamples   = 256;               % 128 or 256
true_bearing_deg = 0.0;              % known interferer bearing
                                     % (for validation only)
```

Update other thresholds if your dataset requires different 
values from the defaults derived in the thesis.

### Step 4 — Run Main Script

```matlab
Final_radar_detection_separator_with_angle.m
```

### Step 5 — View Results

The script prints a full parameter profile to the console 
and displays all visualisation figures automatically.

**Example console output:**
```
=== Radar Parameters ===
ADC samples:         256
Range resolution:    0.1954 m
Velocity resolution: 0.1243 m/s

--- STEP 1: Interference spike detection ---
Interference DETECTED: 65 spikes

--- Final classification ---
Static       : 17
Moving       : 1411
Interference : 48  (confirmed)
Weak Interf. : 11

--- Bearing Estimate ---
Weighted bearing: +25.7 deg
```

---

## Dataset

The experimental datasets used in the thesis were collected 
in a 5 m × 5 m laboratory environment at FH Dortmund. 
A total of **27 datasets** were recorded across **17 unique 
scenarios** covering:

- Clean environments (no targets, no interferer)
- Target-only scenarios (static objects, moving person)
- Mixed scenarios (targets + active interferer at multiple 
  positions: ahead, left, right, out of FoV)

Two hardware configurations were used:
- **128-sample** (16 datasets): BW = 383.77 MHz, ΔR = 0.3909 m
- **256-sample** (11 datasets): BW = 767.54 MHz, ΔR = 0.1954 m

> **Note:** Raw `.bin` files are not included in this 
> repository due to file size. Contact the author for 
> data access.

---

## Classification Thresholds

All thresholds were derived empirically from cross-dataset 
profiling rather than assumed from prior literature.

| Parameter | 128-Sample | 256-Sample |
|---|---|---|
| SNR noise gate | 8 dB | 8 dB |
| SNR static min | 19 dB | 28 dB |
| SNR moving min | 16 dB | 16 dB |
| Velocity static max | 0.15 m/s | 0.15 m/s |
| Velocity moving range | 0.16--1.00 m/s | 0.16--1.50 m/s |
| Min spike count | 1 | 5 |
| Spike threshold δ | 5.0 dB | 5.0 dB |

---

