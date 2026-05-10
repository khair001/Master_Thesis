Interference Characterization and Localization for FMCW mmWave
Radars

This repository contains MATLAB scripts developed for my Master’s thesis in Embedded Systems Engineering using the Texas Instruments IWR6843 radar platform.

The project focuses on:

Radar interference detection
Detection classification
DBSCAN clustering
Interference source bearing estimation
Range–Doppler and spatial visualisation
Files Overview
1. Analyzer Script

This script is used to analyse the radar dataset and observe parameter behaviour before running the final processing pipeline.

Purpose

The analyzer helps to:

Inspect Range–Doppler behaviour
Observe noise and interference characteristics
Tune thresholds and parameters
Evaluate detection behaviour
Select suitable configuration values for the main script
Typical Parameters Analysed
CFAR sensitivity
SNR thresholds
Velocity thresholds
DBSCAN parameters
Interference spike thresholds
Range limits
Usage

Run the analyzer script first:

radar_data_analyser.m

Use the generated plots and statistics to adjust parameters for the main processing script.

2. Main Processing Script

After tuning parameters using the analyzer script, the main script performs the complete radar processing pipeline.

Functions

The script:

Reads recorded radar .bin files
Generates Range–Doppler maps
Applies CFAR detection
Classifies detections into:
Noise
Static
Moving
Interference
Detects interference spikes
Applies DBSCAN clustering
Computes cluster centroids
Estimates interference bearing using antenna phase differences
Generates plots and final statistics
Outputs

The script generates:

Detection statistics
Interference classification results
Bearing estimation results
Visualisation figures:
Spike detection
Range vs velocity
X/Y spatial map
3D detection map
SNR distributions
Range–Doppler map
Polar bearing plot
How to Use
Requirements
MATLAB
Signal Processing Toolbox
Statistics and Machine Learning Toolbox
Workflow
Step 1 — Place Radar Data

Put the recorded radar .bin file inside the project folder.

Step 2 — Run Analyzer Script

Run the analyzer script first to study parameter behaviour:

radar_data_analyser.m

Use the generated plots and statistics to tune thresholds and processing parameters.

Step 3 — Update Parameters

Set the tuned parameters in the main processing script:

testFile = 'your_file.bin';
true_bearing_deg = value;

Update other thresholds if required.

Step 4 — Run Main Script

Run the main processing script:

Final_radar_detection_separator_with_angle.m
Step 5 — View Results

The script will display:

Final detection statistics
Interference classification
Bearing estimation
Visualisation figures
Thesis Objective

The objective of this work is to investigate radar interference detection and interference source localisation using mmWave FMCW radar data from the TI IWR6843 platform.

The implementation focuses on:

Reliable interference detection
Clustering of interference detections
Bearing estimation using antenna phase differences
Statistical and spatial analysis of interference behaviour
