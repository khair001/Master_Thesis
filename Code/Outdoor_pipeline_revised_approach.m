%% =========================================================================
%  Radar Detector + Classifier + DBSCAN + X/Y Spatial Plot +
%  Range-Doppler Map + Interference Bearing Estimation
%  (Outdoor, 256-sample configuration)
%
%  PURPOSE:
%    Full processing pipeline for a TI IWR6843ISK 60 GHz FMCW radar.
%    Reads raw ADC binary data, computes Range-Doppler maps, detects
%    targets using CFAR, classifies them (Static / Moving / Interference),
%    clusters confirmed detections with DBSCAN, and estimates the azimuth
%    bearing of any interference source via coherent RX1/RX4 phase
%    comparison (cross-checked against an independent RX2/RX3 estimate).
%
%  DATA COLLECTION SETUP:
%    Observer PC  : IWR6843ISK + DCA1000EVM, captured via mmWave Studio.
%    Interferer PC: IWR6843AOPEVM, run via Demo Visualizer (no data is
%                   recorded from this radar — it exists only to generate
%                   interference).
%    Open-air outdoor site (no roof); four interferer positions tested
%    (0 deg, +/-42.5 deg, out-of-FoV) at two chirp slopes (see below).
%
%  SUPPORTS:  256-sample ADC configuration, outdoor open-area deployment
%             only (30 MHz/us and 65 MHz/us chirp slopes).
%
%  USAGE:
%    1. Set testFile, numADCSamples (256), freqSlope_MHz, and
%       true_bearing_deg at the top.
%    2. Run the script — all figures and console output are generated.
% =========================================================================
clear; clc; close all;

%% ================================================================
%  CONFIGURATION — only change these lines between datasets
%% ================================================================
% Path to raw binary ADC file captured from the radar sensor
testFile         = 'adc_data_Raw_inter6.bin';

% Number of ADC samples per chirp — must match the capture configuration.
% This outdoor-only pipeline is calibrated for 256 samples only (enforced
% below).
numADCSamples    = 256;

% Known true bearing of the interferer from lab geometry measurements.
% Used only for error computation; does not affect processing.
% Convention: 0 = ahead, negative = left, positive = right.
true_bearing_deg = -42.5;

%% ================================================================
%  HARDWARE PARAMETERS
%% ================================================================
numRX            = 4;      % receive antennas on the IWR6843ISK
numChirps        = 128;    % chirps per frame (slow-time samples)
numFramesToAudit = 1000;   % maximum frames read per recording

% Local-neighbourhood spike sensitivity for frame-power spike detection
% (Step 1): a frame is flagged if its power exceeds both neighbours by at
% least this many dB. Set lower than the indoor configuration (1.0 dB)
% because outdoor interference spikes are less pronounced against the
% quieter outdoor noise floor.
detection_sensitivity = 0.5;   % dB

% Temporal correlation window for interference candidacy (Step 4): a Noise
% detection is considered temporally correlated with interference if its
% frame falls within +/- this many frames of a spike frame. A window of 0
% restricts candidacy to the spike frame itself; a wider window admits
% neighbouring frames as well, trading precision for recall.
interf_frame_window = 0;

% --- Waveform parameters (must match the mmWave Studio radar profile) ---
fc          = 60e9;           % carrier frequency (Hz)

% Chirp slope and ramp timing, verified from mmWave Studio Programmed
% Parameters for the two outdoor slope configurations used in this thesis:
%   Slope-30 dataset: freqSlopeConst = 30.0179 MHz/us, rampEndTime = 60 us
%   Slope-65 dataset: freqSlopeConst = 64.9966 MHz/us, rampEndTime = 60 us
% Set freqSlope_MHz to match the capture; rampEndTime is resolved
% automatically below.
freqSlope_MHz = 64.9966;      % MHz/us

if abs(freqSlope_MHz - 30.0179) < 0.1
    freqSlope   = 30.0179e6 / 1e-6;  % Hz/s — slope-30 outdoor
    rampEndTime = 60e-6;              % s
else
    freqSlope   = 64.9966e6 / 1e-6;  % Hz/s — slope-65 outdoor
    rampEndTime = 60e-6;              % s
end
fs          = 10e6;           % ADC sampling rate (Hz)
idleTime    = 100e-6;         % idle time between chirps (s)
chirpTime   = idleTime + rampEndTime;

% --- Derived radar performance metrics ---
BW          = freqSlope * (numADCSamples / fs); % sweep bandwidth (Hz)
c_light     = 3e8;            % speed of light (m/s)
lambda      = c_light / fc;   % wavelength (m)
rangeRes    = c_light / (2 * BW);          % range bin size (m)
Tc          = numChirps * chirpTime;        % coherent processing interval (s)
velRes      = lambda / (2 * Tc);           % Doppler velocity resolution (m/s)
rangeMax    = (fs * c_light) / (2 * freqSlope); % unambiguous range (m)
velMax      = lambda / (4 * chirpTime);    % unambiguous velocity (m/s)

% Half-wavelength antenna spacing (standard ULA assumption for IWR6843)
d_ant       = lambda / 2;

fprintf('=== Radar Parameters ===\n');
fprintf('File:                %s\n', testFile);
fprintf('ADC samples:         %d\n', numADCSamples);
fprintf('Range resolution:    %.4f m\n', rangeRes);
fprintf('Velocity resolution: %.4f m/s\n', velRes);
fprintf('True bearing:        %.1f deg\n', true_bearing_deg);

%% ================================================================
%  CLASSIFICATION AND CFAR THRESHOLDS
%% ================================================================
% CFAR sliding-window parameters (1D range-axis CFAR)
cfar_guard = 1;   % guard cells each side of the cell under test
cfar_train = 4;   % training cells each side of the cell under test
cfar_pfa   = 1e-3; % target probability of false alarm

first_valid_bin   = cfar_guard + cfar_train + 1;
last_valid_bin    = numADCSamples - cfar_guard - cfar_train;
first_valid_range = (first_valid_bin - 1) * rangeRes;

% Outdoor calibration exists only for the 256-sample ADC configuration.
if numADCSamples ~= 256
    error(['Unsupported numADCSamples: %d. This outdoor-only pipeline is ' ...
           'calibrated for 256 ADC samples only.'], numADCSamples);
end

% ---------------------------------------------------------------
%  Outdoor thresholds, derived from threshold-profiler runs across five
%  outdoor datasets (clean, static, moving, static+moving, clean
%  interference) at both the 30 MHz/us and 65 MHz/us chirp slopes.
%
%    - Range gate extended to 12 m for the open-area measurement.
%    - snr_static_min = 10 dB: outdoor static SNR p10 spans 6.3-7.8 dB
%      across datasets; with no dense structural clutter to raise the
%      false-alarm floor, a 10 dB gate captures the full static
%      distribution while remaining above the noise floor gate.
%    - snr_moving_min = 12 dB, snr_interference_min = 10 dB.
%    - snr_noise_thresh = 8 dB (noise floor gate).
%    - min_spike_count kept at 5 (256-sample frame count).
%    - persist_interf_max kept at 0.05, confirmed by the interference
%      profiler run (11.9% / 3.0% of frames for slope-30 / slope-65).
% ---------------------------------------------------------------
minPhysicalRange     = first_valid_range;
maxPhysicalRange     = 12.0;   % outdoor range gate (m); Static/Moving and all plot axes

% --- Relaxed range gate, used only for interference candidacy (Step 4) ---
% A chirp-collision interference artefact is not a physical-range
% measurement: the beat frequency it produces can land anywhere on the
% range axis depending on the timing offset between the colliding chirps,
% independent of any real reflector distance. Static/Moving classification
% and every plot continue to use minPhysicalRange/maxPhysicalRange above;
% this wider gate is used solely to decide which "Noise" detections are
% eligible for promotion to "Interference" in Step 4. The lower bound
% matches the CFAR near-range floor (a processing limit, not a scene
% choice); the upper bound is the true CFAR far-range ceiling.
minInterfRange       = first_valid_range;
maxInterfRange       = (last_valid_bin-1) * rangeRes;

snr_noise_thresh     = 8;      % noise floor gate (dB)
snr_static_min       = 10;     % outdoor static SNR minimum (dB)
snr_moving_min       = 12;     % outdoor moving SNR minimum (dB)
snr_interference_min = 10;     % outdoor interference SNR minimum (dB)
vel_static_thresh    = 0.25;   % static velocity ceiling (m/s)
vel_moving_min       = 0.26;   % moving velocity floor (m/s)
vel_moving_max       = 1.50;   % moving velocity ceiling (m/s)
persist_interf_max   = 0.05;   % transient persistence gate
min_spike_count      = 1;
dcNotch              = 2;
dbscan_eps_static    = 0.5;
dbscan_eps_moving    = 0.5;
dbscan_eps_interf    = 1.5;
dbscan_minpts        = 10;     % 10-point minimum avoids single-frame false clusters
dbscan_minpts_interf = 5;
config_label         = sprintf('256-sample outdoor (%g MHz/us)', freqSlope_MHz);

% --- Bearing estimation quality thresholds ---
bearing_snr_min      = 16;   % dB — minimum SNR for a detection to contribute to bearing

% Circular standard deviation thresholds for per-cluster reliability labels
bearing_std_reliable = 20;   % deg — below this: "High" confidence
bearing_std_poor     = 35;   % deg — above this: "Low" confidence (between: "Moderate")

fprintf('\n--- Thresholds (%s) ---\n', config_label);
fprintf('Range gate:          %.3f to %.1f m (Static/Moving, plots)\n', minPhysicalRange, maxPhysicalRange);
fprintf('Range gate (interf): %.3f to %.1f m (Noise candidates only)\n', minInterfRange, maxInterfRange);
fprintf('SNR noise gate:      %d dB\n',  snr_noise_thresh);
fprintf('SNR static min:      %d dB\n',  snr_static_min);
fprintf('SNR moving min:      %d dB\n',  snr_moving_min);
fprintf('SNR interf min:      %d dB\n',  snr_interference_min);
fprintf('Vel static max:      %.2f m/s\n', vel_static_thresh);
fprintf('Vel moving:          %.2f to %.2f m/s\n', vel_moving_min, vel_moving_max);
fprintf('Min spike count:     %d\n', min_spike_count);
fprintf('--------------------------------------\n\n');

%% --- Colours used consistently across all figures ---
col_noise        = [0.75 0.75 0.75];  % light grey  — noise / unclassified
col_static       = [0.13 0.55 0.13];  % green       — static targets
col_moving       = [0.12 0.35 0.75];  % blue        — moving targets (pedestrians)
col_interf       = [0.85 0.10 0.10];  % red         — clustered interference
col_interf_weak  = [0.95 0.55 0.55];  % pink/light  — isolated interference

%% --- Hann windows for range (fast-time) and Doppler (slow-time) FFTs ---
% Applied before each FFT to suppress spectral sidelobe leakage.
hS = 0.5*(1 - cos(2*pi*(0:numADCSamples-1)'/(numADCSamples-1))); % range window
hC = 0.5*(1 - cos(2*pi*(0:numChirps-1)'/(numChirps-1)));          % Doppler window

% Bytes per frame: complex sample = 2 x int16 (I, Q); numRX channels;
% numChirps chirps per frame.
bytesPerFrame = numADCSamples * numRX * numChirps * 4;

% Index of the zero-velocity (DC) bin after fftshift on the Doppler axis.
zeroVelBin    = ceil(numChirps/2) + 1;

%% ================================================================
%  STEP 1: FRAME POWER + LOCAL NEIGHBOURHOOD SPIKE DETECTION
%
%  A first, lightweight pass computes the mean Range-Doppler power per
%  frame. Frames whose power forms a local maximum relative to their
%  immediate neighbours are flagged as potentially interference-affected.
%  This is a fast pre-screen ahead of the more expensive per-frame CFAR
%  pass in Step 2.
%% ================================================================
fprintf('--- STEP 1: Frame power + interference spike detection ---\n');
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

frame_pwr = zeros(numFramesToAudit, 1);

for f = 1:numFramesToAudit
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if isempty(raw), break; end

    % --- DCA1000 LVDS complex sample reconstruction ---
    % The DCA1000 LVDS stream packs each ADC sample across 4 RX and 2
    % lanes as groups of 8 int16:
    %   [RX1_I, RX2_I, RX1_Q, RX2_Q, RX3_I, RX4_I, RX3_Q, RX4_Q]
    % i.e. every 4 int16 = [I_lane1, I_lane2, Q_lane1, Q_lane2]. A complex
    % sample for a given RX channel is I_laneN + j*Q_laneN, not a
    % consecutive-pair reconstruction.
    raw_grp  = reshape(raw, 4, []);
    cplx_odd = double(raw_grp(1,:)) + 1i*double(raw_grp(3,:));  % lane 1: RX1,RX3
    cplx_evn = double(raw_grp(2,:)) + 1i*double(raw_grp(4,:));  % lane 2: RX2,RX4
    LVDS_vec = zeros(1, numel(cplx_odd)*2);
    LVDS_vec(1:2:end) = cplx_odd;
    LVDS_vec(2:2:end) = cplx_evn;

    % Reshape to [numADCSamples*numRX x numChirps], then extract RX1 by
    % striding across the 4 interleaved RX channels.
    LVDS_tmp = reshape(LVDS_vec, numADCSamples*numRX, numChirps).';
    rx1_data = LVDS_tmp(:, 1:numRX:numADCSamples*numRX).';

    % 2-D Range-Doppler map (dB) for RX1.
    rd_db    = 20*log10(abs(fftshift( ...
                   fft(fft(rx1_data.*hS,numADCSamples,1).*hC',numChirps,2),2)));

    frame_pwr(f) = mean(rd_db(:));
end
fclose(fid);

Total_Audited = find(frame_pwr ~= 0, 1, 'last');
frame_pwr     = frame_pwr(1:Total_Audited);

% Local-neighbourhood spike detection: frame i is a spike only if it
% exceeds both immediate neighbours by more than the sensitivity margin,
% distinguishing a sharp interference spike from gradual power drift
% caused by a moving target.
raw_spike_flags = false(Total_Audited, 1);
for i = 2:(Total_Audited-1)
    if (frame_pwr(i) > frame_pwr(i-1) + detection_sensitivity) && ...
       (frame_pwr(i) > frame_pwr(i+1) + detection_sensitivity)
        raw_spike_flags(i) = true;
    end
end
raw_spike_count = sum(raw_spike_flags);

if raw_spike_count >= min_spike_count
    interfered_flags = raw_spike_flags;
    Interfered_Count = raw_spike_count;
    fprintf('Interference DETECTED: %d spikes (>= threshold %d)\n\n', ...
        Interfered_Count, min_spike_count);
else
    interfered_flags = false(Total_Audited, 1);
    Interfered_Count = 0;
    fprintf('No interference declared: %d spikes (< threshold %d)\n\n', ...
        raw_spike_count, min_spike_count);
end

%% ================================================================
%  STEP 2: PER-FRAME CFAR DETECTION — all 4 RX + azimuth
%
%  For each frame:
%    1. Compute complex Range-Doppler maps for all four RX channels.
%    2. Apply 1D CA-CFAR along the range axis, per Doppler bin, using a
%       dual-pass scheme (strict pass always; relaxed pass on
%       interference-flagged frames only — see the dual-pass CFAR note
%       below).
%    3. For DC-notch bins, apply a higher SNR gate (snr_static_min) to
%       reduce false alarms from static clutter.
%    4. For confirmed detections, compute azimuth from the RX1/RX4 phase
%       difference and convert to Cartesian X/Y.
%    5. Accumulate RD maps across frames for the averaged Fig 7 display.
%
%  Complex RX1/RX4 and RX2/RX3 values are also stored per detection for
%  bearing estimation in Step 6b.
%% ================================================================
fprintf('--- STEP 2: Processing %d frames ---\n', Total_Audited);
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

% frame_dets{f} columns: [frame, range, vel, snr, peak, az_deg, x, y, isRelaxedOnly]
% isRelaxedOnly: 1 if this detection was only revealed by the relaxed
% interference-frame CFAR threshold (not the strict pass) — see the
% dual-pass CFAR note below. Forced to "Noise" in Step 3 regardless of
% velocity/SNR, so it can only be promoted to "Interference" via Step 4,
% never classified as a real target.
frame_dets = cell(Total_Audited, 1);

% Complex value storage for bearing estimation (preserves phase for
% coherent averaging).
rx1_complex_store = cell(Total_Audited, 1);
rx4_complex_store = cell(Total_Audited, 1);

% RX2/RX3 complex value storage, used only for an independent bearing
% cross-check against the primary RX1/RX4 estimate (Step 6b). RX2/RX3
% share the same physical spacing (lambda/2) and horizontal alignment as
% RX1/RX4, so angle(RX2 . conj(RX3)) should encode the same azimuth as
% angle(RX1 . conj(RX4)) if the array geometry and wiring assumptions are
% correct. No 180-degree correction is applied here: RX2 and RX3 are each
% shifted 180 degrees relative to RX1/RX4 by the same amount, so that
% common offset cancels exactly in the RX2-vs-RX3 phase product.
rx2_complex_store = cell(Total_Audited, 1);
rx3_complex_store = cell(Total_Audited, 1);

% CA-CFAR threshold multiplier from the standard closed-form relation:
%   alpha = N * (Pfa^(-1/N) - 1),  N = number of training cells
numTrain_val = 2 * cfar_train;
alpha_cfar   = numTrain_val * (cfar_pfa^(-1/numTrain_val) - 1);
alpha_dB     = 10*log10(alpha_cfar);

% --- Dual-pass CA-CFAR threshold ---
%  Strict pass — 18 dB, applied to every frame. This matches mmWave
%  Studio's own threshold, eliminates the thermal-noise false-alarm
%  carpet (verified on a clean dataset), and is the ONLY pass that feeds
%  Static/Moving/Noise classification in Step 3.
%
%  Relaxed pass — 13.0 dB (PFA-derived), applied only on interference-
%  flagged frames, and only to cells not already found by the strict
%  pass. Interference raises the entire Range-Doppler noise floor, so the
%  local training-cell mean rises with the interference signal; a cell
%  only 12-15 dB above that elevated local mean would be suppressed by
%  the 18 dB gate, and the interference peak would go undetected with no
%  input left for bearing estimation. The relaxed pass lets weak
%  interference peaks (profiler-observed mean SNR of 7-9 dB above the
%  global floor on interference frames) survive CFAR and reach the
%  interference-candidate pool in Step 4.
alpha_dB_clean  = 18;     % clean outdoor frames
alpha_dB_interf = 13.0;   % interference outdoor frames (relaxed, gated to spike frames only)
fprintf('CA-CFAR threshold:   %.1f dB (clean frames)\n', alpha_dB_clean);
fprintf('CA-CFAR threshold:   %.1f dB (interference frames)\n', alpha_dB_interf);

% Accumulators for the time-averaged Range-Doppler map (Fig 7).
rd_map_accum       = zeros(numADCSamples, numChirps);
frames_accumulated = 0;

for f = 1:Total_Audited
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if numel(raw) < bytesPerFrame/2
        fprintf('End of file at frame %d\n', f);
        Total_Audited = f-1; break;
    end

    % --- DCA1000 LVDS complex reconstruction (same format as Step 1) ---
    raw_grp  = reshape(raw, 4, []);
    cplx_odd = double(raw_grp(1,:)) + 1i*double(raw_grp(3,:));
    cplx_evn = double(raw_grp(2,:)) + 1i*double(raw_grp(4,:));
    LVDS_all = zeros(1, numel(cplx_odd)*2);
    LVDS_all(1:2:end) = cplx_odd;
    LVDS_all(2:2:end) = cplx_evn;
    LVDS_mat = reshape(LVDS_all, numADCSamples*numRX, numChirps).';

    % rx{rxIdx} = [numADCSamples x numChirps] complex fast-time / slow-time matrix
    rx = cell(4,1);
    for rxIdx = 1:numRX
        rx{rxIdx} = LVDS_mat(:, rxIdx:numRX:numADCSamples*numRX).';
    end

    % Complex Range-Doppler maps for RX1 and RX4 (primary bearing pair).
    % fftshift centres zero-velocity at column zeroVelBin.
    rd_complex_rx1 = fftshift( ...
        fft(fft(rx{1}.*hS,numADCSamples,1).*hC',numChirps,2), 2);
    rd_complex_rx4 = fftshift( ...
        fft(fft(rx{4}.*hS,numADCSamples,1).*hC',numChirps,2), 2);

    % RX2/RX3 range-Doppler maps, used only for the independent bearing
    % cross-check in Step 6b; detection and CFAR remain RX1-based.
    rd_complex_rx2 = fftshift( ...
        fft(fft(rx{2}.*hS,numADCSamples,1).*hC',numChirps,2), 2);
    rd_complex_rx3 = fftshift( ...
        fft(fft(rx{3}.*hS,numADCSamples,1).*hC',numChirps,2), 2);

    % Magnitude (dB) map used for CFAR and SNR computation.
    rd_db = 20*log10(abs(rd_complex_rx1) + eps);

    rd_map_accum       = rd_map_accum + rd_db;
    frames_accumulated = frames_accumulated + 1;

    % Per-frame noise floor: mean power across the entire RD map.
    noiseFloor_dB = mean(rd_db(:));

    % DC notch: zero out the zero-velocity strip to suppress static
    % clutter for the non-DC CFAR path, while the unnotched map (rd_db)
    % is retained separately for static-target detection.
    dcLo = max(1, zeroVelBin - dcNotch);
    dcHi = min(numChirps, zeroVelBin + dcNotch);

    rd_mag_notched = abs(rd_complex_rx1);
    rd_mag_notched(:, dcLo:dcHi) = 0;
    rd_db_notched  = 20*log10(rd_mag_notched + eps);

    % --- Dual-pass CFAR ---
    % A single per-frame threshold (relaxed on spike frames, strict
    % otherwise) previously fed both the real-target (Static/Moving) and
    % interference-candidate detections from the same pass. On spike
    % frames this let a genuine moving target's own micro-Doppler smear
    % / window sidelobes — energy sitting just under the strict 18 dB bar
    % in clean frames — clear the relaxed 13 dB bar and be misclassified
    % as interference-adjacent Noise-then-Moving. Splitting the CFAR into
    % two passes fixes this:
    %   STRICT pass — always run at alpha_dB_clean, independent of
    %   interfered_flags. This is the only pass that feeds Step 3
    %   Static/Moving/Noise classification, so real-target counts no
    %   longer depend on whether the frame is spike-flagged.
    %   RELAXED pass — computed only on spike-flagged frames, at
    %   alpha_dB_interf. Its cells are intersected against the strict
    %   pass to keep only what the relaxed threshold uniquely reveals
    %   (cfar_mask_relaxed_extra). These cells are tagged isRelaxedOnly
    %   and forced to "Noise" in Step 3, so they can only ever be
    %   promoted to "Interference" via the normal Step 4 gates.
    cfar_mask_strict = false(numADCSamples, numChirps);
    for d = 1:numChirps
        if d >= dcLo && d <= dcHi, continue; end
        col = rd_db_notched(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB_clean ...
               && (col(r)-noiseFloor_dB) > snr_noise_thresh
                cfar_mask_strict(r,d) = true;
            end
        end
    end
    for d = dcLo:dcHi
        col = rd_db(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB_clean ...
               && (col(r)-noiseFloor_dB) > snr_static_min
                cfar_mask_strict(r,d) = true;
            end
        end
    end

    cfar_mask_relaxed_extra = false(numADCSamples, numChirps);
    if interfered_flags(f)
        cfar_mask_relaxed = false(numADCSamples, numChirps);
        for d = 1:numChirps
            if d >= dcLo && d <= dcHi, continue; end
            col = rd_db_notched(:, d);
            for r = first_valid_bin : last_valid_bin
                rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                        r+cfar_guard+1          : r+cfar_guard+cfar_train];
                if col(r) > mean(col(rWin)) + alpha_dB_interf ...
                   && (col(r)-noiseFloor_dB) > snr_noise_thresh
                    cfar_mask_relaxed(r,d) = true;
                end
            end
        end
        for d = dcLo:dcHi
            col = rd_db(:, d);
            for r = first_valid_bin : last_valid_bin
                rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                        r+cfar_guard+1          : r+cfar_guard+cfar_train];
                if col(r) > mean(col(rWin)) + alpha_dB_interf ...
                   && (col(r)-noiseFloor_dB) > snr_static_min
                    cfar_mask_relaxed(r,d) = true;
                end
            end
        end
        % Keep only cells uniquely revealed by the relaxed threshold.
        cfar_mask_relaxed_extra = cfar_mask_relaxed & ~cfar_mask_strict;
    end

    % Combined mask for detection extraction; the two source masks remain
    % available per-cell to tag provenance (isRelaxedOnly, below).
    cfar_mask = cfar_mask_strict | cfar_mask_relaxed_extra;

    % Convert CFAR-flagged (range bin, Doppler bin) pairs into physical
    % detection features.
    [rBins, dBins] = find(cfar_mask);
    if ~isempty(rBins)
        ranges = (rBins-1) * rangeRes;
        vels   = (dBins-zeroVelBin) * velRes;
        snrs   = rd_db(sub2ind(size(rd_db),rBins,dBins)) - noiseFloor_dB;
        peaks  = rd_db(sub2ind(size(rd_db),rBins,dBins));

        % Provenance tag: true if this cell was only revealed by the
        % relaxed interference-frame threshold.
        isRelaxedOnly = cfar_mask_relaxed_extra(sub2ind(size(cfar_mask_relaxed_extra),rBins,dBins));

        % --- Phase-based azimuth estimation (RX1 vs RX4) ---
        % Delta_phi = angle(X_RX1 . X_RX4*) is proportional to sin(theta):
        %   Delta_phi = 2*pi*d_ant*sin(theta) / lambda
        % With d_ant = lambda/2: Delta_phi = pi*sin(theta), so
        %   theta = asin(Delta_phi / pi)
        rx1_val    = rd_complex_rx1(sub2ind(size(rd_complex_rx1),rBins,dBins));
        rx4_val    = rd_complex_rx4(sub2ind(size(rd_complex_rx4),rBins,dBins));
        phase_diff = angle(rx1_val .* conj(rx4_val));
        sin_az     = max(-1, min(1, phase_diff / pi));
        azimuth    = asin(sin_az);

        % Cartesian coordinates (X = cross-range, Y = forward range),
        % derived directly from (range, azimuth).
        x_pos      = ranges .* sin(azimuth);
        y_pos      = ranges .* cos(azimuth);

        % Column 9 = isRelaxedOnly provenance tag, used by Step 3.
        frame_dets{f} = [repmat(f,numel(rBins),1), ranges, vels, ...
                         snrs, peaks, rad2deg(azimuth), x_pos, y_pos, double(isRelaxedOnly)];

        % RX2/RX3 complex values at the same bins, for the independent
        % bearing cross-check in Step 6b (not used for detection).
        rx2_val    = rd_complex_rx2(sub2ind(size(rd_complex_rx2),rBins,dBins));
        rx3_val    = rd_complex_rx3(sub2ind(size(rd_complex_rx3),rBins,dBins));

        rx1_complex_store{f} = rx1_val;
        rx4_complex_store{f} = rx4_val;
        rx2_complex_store{f} = rx2_val;
        rx3_complex_store{f} = rx3_val;
    else
        frame_dets{f}         = zeros(0,9);
        rx1_complex_store{f}  = [];
        rx4_complex_store{f}  = [];
        rx2_complex_store{f}  = [];
        rx3_complex_store{f}  = [];
    end

    if mod(f,100)==0
        nSoFar = sum(cellfun(@(x) size(x,1), frame_dets(1:f)));
        fprintf('  Frame %d / %d | Detections so far: %d\n', ...
            f, Total_Audited, nSoFar);
    end
end
fclose(fid);

% Time-averaged RD map, used as the Fig 7 background image.
rd_map_avg = rd_map_accum / frames_accumulated;

% Flatten all per-frame detection cells into single column vectors.
allDets    = cell2mat(frame_dets(1:Total_Audited));
det_frame  = allDets(:,1);
det_range  = allDets(:,2);
det_vel    = allDets(:,3);
det_snr    = allDets(:,4);
det_rdPeak = allDets(:,5);
det_az_deg = allDets(:,6);
det_x      = allDets(:,7);
det_y      = allDets(:,8);
det_relaxedOnly = logical(allDets(:,9)); % true = only revealed via relaxed interference-frame threshold
detCount   = size(allDets,1);

% Concatenate complex values in the same row-order as allDets.
rx1_complex_all = cell2mat(rx1_complex_store(1:Total_Audited));
rx4_complex_all = cell2mat(rx4_complex_store(1:Total_Audited));
rx2_complex_all = cell2mat(rx2_complex_store(1:Total_Audited));
rx3_complex_all = cell2mat(rx3_complex_store(1:Total_Audited));

fprintf('Total detections (pre-gate): %d\n', detCount);
fprintf('  of which relaxed-only (interference-candidate-only, spike frames): %d\n\n', ...
    sum(det_relaxedOnly));

%% ================================================================
%  STEP 3: PRIMARY CLASSIFICATION — Static / Moving / Noise
%
%  Each detection is assigned exactly one of three labels based on
%  range gate, SNR, and absolute velocity:
%
%    Noise  : outside the range gate, below the noise floor gate, OR
%             only revealed by the relaxed interference-frame CFAR
%             threshold (isRelaxedOnly — see Step 2). Detections in this
%             last group cannot become Static/Moving regardless of their
%             velocity/SNR, since they are only present because the
%             detection threshold was lowered for that frame, not
%             because they cleared the same bar a real target would need
%             to clear on any other frame.
%    Static : low velocity AND high SNR (e.g. walls, poles, buildings)
%    Moving : velocity in the pedestrian band AND sufficient SNR
%
%  Detections that pass the range/SNR gates but fall outside the Static
%  and Moving velocity windows are also labelled Noise here; they may be
%  re-labelled Interference in Step 4.
%% ================================================================
fprintf('--- STEP 3: Primary classification ---\n');
labels = strings(detCount, 1);

for k = 1:detCount
    rng_k = det_range(k);
    snr_k = det_snr(k);
    vel_k = abs(det_vel(k));

    if rng_k < minPhysicalRange || rng_k > maxPhysicalRange
        labels(k) = "Noise"; continue;
    end

    if snr_k < snr_noise_thresh
        labels(k) = "Noise";
    elseif det_relaxedOnly(k)
        % Only visible because this frame's CFAR threshold was relaxed
        % for interference candidacy — not reliable evidence of a real
        % target; may still qualify as an interference candidate in
        % Step 4.
        labels(k) = "Noise";
    elseif vel_k < vel_static_thresh && snr_k >= snr_static_min
        labels(k) = "Static";
    elseif vel_k >= vel_moving_min && vel_k <= vel_moving_max ...
           && snr_k >= snr_moving_min
        labels(k) = "Moving";
    else
        labels(k) = "Noise";
    end
end

fprintf('After primary classification:\n');
fprintf('  Static : %d\n', sum(labels=="Static"));
fprintf('  Moving : %d\n', sum(labels=="Moving"));
fprintf('  Noise  : %d\n\n', sum(labels=="Noise"));

%% ================================================================
%  STEP 4: SECONDARY LOOP — promote Noise to Interference
%
%  A detection labelled Noise in Step 3 is promoted to "Interference" if
%  and only if all of the following hold:
%    (0) RELAXED RANGE GATE — candidacy is evaluated over
%        minInterfRange/maxInterfRange rather than the stricter
%        minPhysicalRange/maxPhysicalRange used for Static/Moving, since
%        interference beat artefacts are not physical-range measurements.
%    (1) TEMPORAL CORRELATION — the detection's frame falls within
%        +/- interf_frame_window frames of an interference-spike frame.
%    (2) ANOMALOUS VELOCITY (mandatory) — the velocity is implausible
%        for a genuine pedestrian target: either above vel_moving_max,
%        or inside the static/moving dead-zone below vel_moving_min. A
%        detection whose velocity sits inside the normal pedestrian band
%        is never promoted to Interference, regardless of SNR or
%        frame-flag, and remains Noise.
%    (3) SNR — the detection exceeds the minimum interference SNR gate.
%
%  DESIGN RATIONALE:
%    Gating on temporal correlation and SNR alone would let a genuine but
%    SNR-weak moving-target detection be misclassified as Interference
%    whenever it shares a frame with a real interference spike. Making
%    anomalous velocity mandatory removes that failure mode: no
%    detection with a pedestrian-consistent velocity can be labelled
%    Interference. The accepted trade-off is reduced recall — genuine
%    interference whose beat-frequency artefact happens to land inside
%    the pedestrian velocity band is left as Noise. This favours
%    precision over recall for the Interference class.
%
%    Frame persistence was considered as an additional promotion gate but
%    was not adopted: it cannot distinguish a transient interference
%    event from a genuine pedestrian passing through, since both are
%    transient at any single range bin by construction. It is retained
%    below as a descriptive statistic only.
%% ================================================================
if Interfered_Count >= min_spike_count
    fprintf('--- STEP 4: Interference separation (%d spikes) ---\n', ...
        Interfered_Count);

    % Persistence: fraction of total frames in which each range bin is
    % occupied (descriptive only; not used to gate promotion).
    rangeBin    = round(det_range / rangeRes);
    uniqueBins  = unique(rangeBin);
    persistence = zeros(detCount, 1);
    for rb = uniqueBins'
        bm  = rangeBin == rb;
        persistence(bm) = numel(unique(det_frame(bm))) / Total_Audited;
    end

    % Expand spike flags by +/- interf_frame_window frames to catch
    % boundary spillover from the finite duration of a chirp collision.
    expandedFlags = interfered_flags;
    for ifr = find(interfered_flags)'
        loFr = max(1, ifr - interf_frame_window);
        hiFr = min(Total_Audited, ifr + interf_frame_window);
        expandedFlags(loFr:hiFr) = true;
    end

    % Only Noise detections inside the relaxed interference range gate
    % are considered — intentionally wider than the Static/Moving range
    % gate, since restricting candidacy to the scene's real target range
    % would discard genuine interference that aliases outside it.
    noiseIdx = find(labels=="Noise" & ...
                    det_range >= minInterfRange & ...
                    det_range <= maxInterfRange);
    promoted = 0;
    rejected_pedestrianVel = 0;
    promoted_relaxedOnly = 0;

    % Descriptive-only counters, not used to gate promotion.
    n_unusualVel = 0; n_ambiguousVel = 0; n_suddenAppearance = 0;

    for k = noiseIdx'
        fi    = det_frame(k);
        vel_k = abs(det_vel(k));
        snr_k = det_snr(k);
        per_k = persistence(k);

        % Gate 1 — temporal correlation.
        if ~expandedFlags(fi), continue; end

        % Gate 2 — anomalous velocity (mandatory).
        unusualVel   = vel_k > vel_moving_max;
        ambiguousVel = vel_k >= vel_static_thresh && vel_k < vel_moving_min;
        anomalousVel = unusualVel || ambiguousVel;
        if ~anomalousVel
            if vel_k >= vel_moving_min && vel_k <= vel_moving_max
                rejected_pedestrianVel = rejected_pedestrianVel + 1;
            end
            continue;
        end

        % Gate 3 — SNR floor.
        if snr_k < snr_interference_min, continue; end

        labels(k) = "Interference"; promoted = promoted+1;
        if det_relaxedOnly(k), promoted_relaxedOnly = promoted_relaxedOnly + 1; end

        % Descriptive-only signature breakdown.
        suddenAppearance = per_k <= persist_interf_max;
        if unusualVel,       n_unusualVel       = n_unusualVel + 1;       end
        if ambiguousVel,     n_ambiguousVel     = n_ambiguousVel + 1;     end
        if suddenAppearance, n_suddenAppearance = n_suddenAppearance + 1; end
    end

    fprintf('  Noise promoted to Interference: %d\n', promoted);
    fprintf('    of which from relaxed-only CFAR pass: %d (%.0f%%)\n', ...
        promoted_relaxedOnly, 100*promoted_relaxedOnly/max(promoted,1));
    fprintf('  Withheld (frame-flagged, pedestrian-band velocity, excluded by design): %d\n', ...
        rejected_pedestrianVel);
    if promoted > 0
        fprintf('  Of the promoted detections:\n');
        fprintf('    via unusual (too-fast) velocity:  %d (%.0f%%)\n', ...
            n_unusualVel, 100*n_unusualVel/promoted);
        fprintf('    via ambiguous-zone velocity:       %d (%.0f%%)\n', ...
            n_ambiguousVel, 100*n_ambiguousVel/promoted);
        fprintf('    also showing transient persistence (descriptive only): %d (%.0f%%)\n\n', ...
            n_suddenAppearance, 100*n_suddenAppearance/promoted);
    else
        fprintf('\n');
    end
else
    fprintf('--- STEP 4: Skipped (spikes %d < threshold %d) ---\n\n', ...
        raw_spike_count, min_spike_count);
    persistence = zeros(detCount,1);
end

fprintf('Final classification:\n');
fprintf('  Static       : %d\n', sum(labels=="Static"));
fprintf('  Moving       : %d\n', sum(labels=="Moving"));
fprintf('  Interference : %d\n', sum(labels=="Interference"));
fprintf('  Noise        : %d\n\n', sum(labels=="Noise"));

%% ================================================================
%  STEP 5: DBSCAN CLUSTERING
%
%  Groups spatially/spectrally nearby detections into clusters so that
%  centroid positions and bearing estimates are computed on confirmed
%  groups rather than isolated detections.
%
%  Feature space: [range_bin, Doppler_bin], normalised to bin units so
%  eps carries the same meaning in both dimensions.
%
%  Cluster ID encoding (avoids collisions between classes):
%    Static : cluster ID + 1000; Moving: + 2000; Interference: + 3000
%
%  DBSCAN returns -1 for noise/outlier points; these are:
%    Static/Moving : remain in their class but unclustered
%    Interference  : re-labelled "isolated interference" (unreliable for bearing)
%% ================================================================
fprintf('--- STEP 5: DBSCAN Clustering ---\n');

cluster_ids = zeros(detCount, 1);

% --- Static ---
idx = find(labels == "Static");
if numel(idx) >= dbscan_minpts
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_static, dbscan_minpts);
    cluster_ids(idx) = clust + 1000;
    fprintf('  Static  : %d detections -> %d clusters, %d unclustered\n', ...
        numel(idx), max(max(clust),0), sum(clust==-1));
end

% --- Moving ---
idx = find(labels == "Moving");
if numel(idx) >= dbscan_minpts
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_moving, dbscan_minpts);
    cluster_ids(idx) = clust + 2000;
    fprintf('  Moving  : %d detections -> %d clusters, %d unclustered\n', ...
        numel(idx), max(max(clust),0), sum(clust==-1));
end

% --- Interference ---
idx = find(labels == "Interference");
if numel(idx) >= dbscan_minpts_interf
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_interf, dbscan_minpts_interf);
    cluster_ids(idx) = clust + 3000;

    unclustMask = clust == -1;
    labels(idx(unclustMask)) = "isolated interference";
    labels(idx(~unclustMask)) = "clustered interference";

    nClust  = max(clust);
    nWeak   = sum(unclustMask);
    nStrong = sum(~unclustMask);
    fprintf('  Interf  : %d detections -> %d clusters (%d confirmed), %d weak\n', ...
        numel(idx), max(nClust,0), nStrong, nWeak);
elseif numel(idx) > 0
    labels(idx) = "isolated interference";
    fprintf('  Interf  : %d detections — all weak (below DBSCAN minimum)\n', numel(idx));
end

fprintf('  Clustered Interference : %d\n', sum(labels=="clustered interference"));
fprintf('  Isolated Interference  : %d\n\n', sum(labels=="isolated interference"));

%% ================================================================
%  STEP 5b: STATIC CLUSTER CONSOLIDATION (outdoor only)
%
%  A single static outdoor reflector (tree, pole, building face) produces
%  a strong response at exactly one range bin. The Hann window used in
%  the slow-time FFT has a main lobe roughly 2 bins wide, so energy leaks
%  into the +/-1 adjacent Doppler bins at approximately -5 dB below the
%  peak. Both adjacent bins can survive the SNR gate and DBSCAN
%  independently, each forming its own cluster. This step merges clusters
%  that lie within half a range bin of each other (i.e. the same range
%  bin, different Doppler bins) into one cluster, keeping the ID of the
%  highest-SNR member. Clusters at distinct range bins (>= 0.5 x
%  rangeRes apart) are never merged.
%
%  Applied unconditionally here: the extended range gate and sparser
%  clutter in outdoor scenes make Doppler-window smearing the dominant
%  source of cluster inflation for static reflectors.
%% ================================================================
static_mask = labels == "Static";
if sum(static_mask) > 0
    s_idx  = find(static_mask);
    s_cids = cluster_ids(s_idx);

    % Only confirmed clusters are considered (DBSCAN outliers sit at offset-1).
    uC_s = unique(s_cids); uC_s = uC_s(uC_s > 1000);
    nCs  = numel(uC_s);

    if nCs > 1
        cMeanRng = zeros(nCs, 1);
        cMeanSNR = zeros(nCs, 1);
        for ci = 1:nCs
            m = s_cids == uC_s(ci);
            cMeanRng(ci) = mean(det_range(s_idx(m)));
            cMeanSNR(ci) = mean(det_snr(s_idx(m)));
        end

        % Merge pairs whose mean ranges fall within the same range bin.
        % Tolerance = 0.51 x rangeRes: same-bin pairs (diff ~ 0) merge;
        % adjacent-bin pairs (diff ~ rangeRes) do not.
        range_bin_tol = 0.51 * rangeRes;
        merge_map = (1:nCs)';

        for ci = 1:nCs
            for cj = ci+1:nCs
                if abs(cMeanRng(ci) - cMeanRng(cj)) <= range_bin_tol
                    if cMeanSNR(ci) >= cMeanSNR(cj)
                        merge_map(cj) = merge_map(ci);
                    else
                        merge_map(ci) = merge_map(cj);
                    end
                end
            end
        end

        % Propagate transitive merges (A->B, B->C => A->C).
        changed = true;
        while changed
            changed = false;
            for ci = 1:nCs
                target = merge_map(merge_map(ci));
                if target ~= merge_map(ci)
                    merge_map(ci) = target;
                    changed = true;
                end
            end
        end

        % Re-assign cluster_ids: merged siblings take the winner's ID.
        for ci = 1:nCs
            if merge_map(ci) ~= ci
                m = s_cids == uC_s(ci);
                cluster_ids(s_idx(m)) = uC_s(merge_map(ci));
            end
        end

        nAfter = numel(unique(cluster_ids(s_idx(cluster_ids(s_idx) > 1000))));
        fprintf('--- STEP 5b: Static cluster consolidation ---\n');
        fprintf('  Doppler-smear merge: %d clusters -> %d physical reflectors\n\n', ...
            nCs, nAfter);
    end
end

%% ================================================================
%  STEP 6: CLUSTER CENTROIDS
%
%  For each confirmed cluster (Static / Moving / Interference), computes
%  the mean range, velocity, SNR, and X/Y position across all member
%  detections. Results are stored in centroid_list for plotting.
%% ================================================================
fprintf('--- Cluster Centroids ---\n');
fprintf('%-22s %6s %10s %10s %8s %8s %8s %8s\n', ...
    'Class','ClustID','Range(m)','Vel(m/s)','SNR(dB)','X(m)','Y(m)','Count');
fprintf('%s\n', repmat('-',1,80));

% centroid_list columns: [classIndex, clusterID, range, vel, SNR, X, Y, count]
centroid_list = [];

colMap = {col_static, col_moving, col_interf};
lblMap = {'Static centroid','Moving centroid','Clustered Interference centroid'};

clsCluster = {"Static",1000;"Moving",2000;"clustered interference",3000};

for c = 1:size(clsCluster,1)
    cl     = clsCluster{c,1};
    offset = clsCluster{c,2};
    idx    = find(labels == cl);
    if isempty(idx), continue; end

    clust  = cluster_ids(idx);
    uC     = unique(clust); uC = uC(uC > offset);  % exclude DBSCAN outliers

    for uc = uC'
        cidx   = idx(clust == uc);
        cRange = mean(det_range(cidx));
        cVel   = mean(det_vel(cidx));
        cSNR   = mean(det_snr(cidx));
        cX     = mean(det_x(cidx));
        cY     = mean(det_y(cidx));
        cCount = numel(cidx);
        fprintf('%-22s %6d %10.3f %10.3f %8.2f %8.3f %8.3f %8d\n', ...
            cl, uc-offset, cRange, cVel, cSNR, cX, cY, cCount);
        centroid_list(end+1,:) = [c, uc, cRange, cVel, cSNR, cX, cY, cCount];
    end
end

%% ================================================================
%  STEP 6b: BEARING ESTIMATION FROM INTERFERENCE CLUSTERS
%
%  PRINCIPLE:
%    The IWR6843ISK has a ULA with d = lambda/2 spacing. For two antennas
%    separated by d, the inter-element phase difference is:
%      Delta_phi = (2*pi * d * sin(theta)) / lambda = pi * sin(theta)
%    so theta = asin(Delta_phi / pi).
%
%    Rather than averaging per-detection phase angles (which suffer from
%    wrap-around), the complex phase products
%      sum(X_RX1 . X_RX4*)
%    are summed and the angle of the sum is taken (coherent averaging).
%    This is equivalent to maximum-likelihood estimation under additive
%    noise and handles phase wrapping correctly.
%
%  RX2/RX3 CROSS-CHECK:
%    RX2/RX3 form a second physical antenna pair with the same lambda/2
%    horizontal spacing and left/right alignment as RX1/RX4 (per the
%    antenna layout diagram). If the array geometry and wiring
%    assumptions used for RX1/RX4 are correct, sum(X_RX2 . X_RX3*)
%    should yield the same azimuth, computed completely independently.
%    This is used purely as a validation cross-check, not as part of the
%    primary bearing estimate. No 180-degree correction is applied: the
%    common 180-degree offset on RX2 and RX3 relative to RX1/RX4 cancels
%    exactly in the RX2-vs-RX3 phase product.
%
%  PER-CLUSTER OUTPUT:
%    bearing_deg        — estimated azimuth (RX1/RX4, primary)
%    bearing_std         — circular standard deviation (spread indicator)
%    reliability          — "High" / "Moderate" / "Low", from std thresholds
%    bearing_deg_rx23    — independent cross-check estimate (RX2/RX3)
%    bearing_agree_diff — signed angular difference (RX1/RX4 minus RX2/RX3)
%
%  Convention: 0 deg = directly ahead, negative = left, positive = right.
%% ================================================================
nConfirmedInterf = sum(labels=="clustered interference");

if nConfirmedInterf > 0
    fprintf('\n--- STEP 6b: Interference Bearing Estimation ---\n');
    fprintf('Convention: 0 deg = directly ahead, negative = left, positive = right\n\n');
    interfIdx = find(labels=="clustered interference");
    interfClustIDs = cluster_ids(interfIdx);
    uniqueClustIDs = unique(interfClustIDs);
    uniqueClustIDs = uniqueClustIDs(uniqueClustIDs > 3000);
    nClusters      = numel(uniqueClustIDs);

    cluster_bearing_deg  = zeros(nClusters, 1);
    cluster_bearing_std  = zeros(nClusters, 1);
    cluster_snr_linear   = zeros(nClusters, 1);
    cluster_snr_db       = zeros(nClusters, 1);
    cluster_count        = zeros(nClusters, 1);
    cluster_range        = zeros(nClusters, 1);
    cluster_reliability  = strings(nClusters, 1);

    % RX2/RX3 independent cross-check results (parallel arrays).
    cluster_bearing_deg_rx23   = zeros(nClusters, 1);
    cluster_bearing_std_rx23   = zeros(nClusters, 1);
    cluster_bearing_agree_diff = zeros(nClusters, 1);   % RX1/RX4 - RX2/RX3, wrapped to [-180,180]

    fprintf('%-10s %9s %11s %9s %9s %11s %11s %9s\n', ...
        'Cluster','Range(m)','RX14(deg)','Std14','SNR(dB)','RX23(deg)','Std23','Diff(deg)');
    fprintf('%s\n', repmat('-',1,80));

    for ci = 1:nClusters
        cid  = uniqueClustIDs(ci);
        cidx = interfIdx(interfClustIDs == cid);

        % Restrict to high-quality detections for the phase estimate,
        % falling back to all members if too few pass the SNR gate.
        snr_mask      = det_snr(cidx) >= bearing_snr_min;
        cidx_filtered = cidx(snr_mask);
        if numel(cidx_filtered) < 2
            cidx_filtered = cidx;
        end

        % --- Coherent phase averaging (RX1/RX4, primary) ---
        rx1_cluster  = rx1_complex_all(cidx_filtered);
        rx4_cluster  = rx4_complex_all(cidx_filtered);
        phase_products = rx1_cluster .* conj(rx4_cluster);
        coherent_sum   = sum(phase_products);

        mean_phase_diff  = angle(coherent_sum);
        sin_bearing      = max(-1, min(1, mean_phase_diff / pi));
        bearing_rad      = asin(sin_bearing);
        bearing_deg_val  = rad2deg(bearing_rad);

        % Circular standard deviation across per-detection bearings
        % (intra-cluster angular spread), used for reliability rating.
        % sigma_c = sqrt(-2*ln(R)), R = mean resultant length.
        per_det_phase = angle(phase_products);
        per_det_sin   = max(-1, min(1, per_det_phase / pi));
        per_det_bear  = rad2deg(asin(per_det_sin));

        sin_mean = mean(sind(per_det_bear));
        cos_mean = mean(cosd(per_det_bear));
        R_circ   = sqrt(sin_mean^2 + cos_mean^2);
        circ_std = rad2deg(sqrt(-2 * log(max(R_circ, 1e-6))));

        if circ_std < bearing_std_reliable
            reliability = "High";
        elseif circ_std < bearing_std_poor
            reliability = "Moderate";
        else
            reliability = "Low";
        end

        % --- Independent RX2/RX3 cross-check (same method, different pair) ---
        rx2_cluster       = rx2_complex_all(cidx_filtered);
        rx3_cluster       = rx3_complex_all(cidx_filtered);
        phase_products_23 = rx2_cluster .* conj(rx3_cluster);
        coherent_sum_23   = sum(phase_products_23);

        mean_phase_diff_23 = angle(coherent_sum_23);
        sin_bearing_23     = max(-1, min(1, mean_phase_diff_23 / pi));
        bearing_rad_23     = asin(sin_bearing_23);
        bearing_deg_val_23 = rad2deg(bearing_rad_23);

        per_det_phase_23 = angle(phase_products_23);
        per_det_sin_23   = max(-1, min(1, per_det_phase_23 / pi));
        per_det_bear_23  = rad2deg(asin(per_det_sin_23));

        sin_mean_23 = mean(sind(per_det_bear_23));
        cos_mean_23 = mean(cosd(per_det_bear_23));
        R_circ_23   = sqrt(sin_mean_23^2 + cos_mean_23^2);
        circ_std_23 = rad2deg(sqrt(-2 * log(max(R_circ_23, 1e-6))));

        % Signed shortest-arc agreement between the two independent estimates.
        agree_diff = mod((bearing_deg_val - bearing_deg_val_23) + 180, 360) - 180;

        cluster_bearing_deg(ci) = bearing_deg_val;
        cluster_bearing_std(ci) = circ_std;
        cluster_snr_db(ci)      = mean(det_snr(cidx));
        cluster_snr_linear(ci)  = 10^(cluster_snr_db(ci)/10);
        cluster_count(ci)       = numel(cidx);
        cluster_range(ci)       = mean(det_range(cidx));
        cluster_reliability(ci) = reliability;

        cluster_bearing_deg_rx23(ci)   = bearing_deg_val_23;
        cluster_bearing_std_rx23(ci)   = circ_std_23;
        cluster_bearing_agree_diff(ci) = agree_diff;

        fprintf('%-10d %9.2f %11.1f %9.1f %9.2f %11.1f %11.1f %9.1f\n', ...
            ci, cluster_range(ci), bearing_deg_val, circ_std, ...
            cluster_snr_db(ci), bearing_deg_val_23, circ_std_23, agree_diff);
    end

    %% STEP 6c: SCENE-LEVEL BEARING — weighted circular mean
    %
    %  Combines per-cluster bearings into one scene estimate: an
    %  unweighted circular mean (reference) and an SNR-weighted circular
    %  mean (primary), which down-weights weaker/noisier clusters.
    fprintf('\n--- Scene-Level Bearing Estimate ---\n');

    sin_vals = sind(cluster_bearing_deg);
    cos_vals = cosd(cluster_bearing_deg);

    bearing_unweighted = atan2d(mean(sin_vals), mean(cos_vals));

    weights      = cluster_snr_linear / sum(cluster_snr_linear);
    sin_weighted = sum(weights .* sin_vals);
    cos_weighted = sum(weights .* cos_vals);
    bearing_weighted = atan2d(sin_weighted, cos_weighted);

    % Inter-cluster circular std — consistency across clusters.
    R_scene        = sqrt(mean(sin_vals)^2 + mean(cos_vals)^2);
    scene_circ_std = rad2deg(sqrt(-2 * log(max(R_scene, 1e-6))));

    % Angular error relative to known lab geometry; wraps to [-180,+180].
    error_unweighted = mod((bearing_unweighted - true_bearing_deg)+180,360)-180;
    error_weighted   = mod((bearing_weighted   - true_bearing_deg)+180,360)-180;

    % RX2/RX3 scene-level cross-check, using the same SNR weights as
    % RX1/RX4 for a like-for-like comparison (both estimates share the
    % same underlying detections, only the antenna pair differs).
    sin_vals_23      = sind(cluster_bearing_deg_rx23);
    cos_vals_23      = cosd(cluster_bearing_deg_rx23);
    sin_weighted_23  = sum(weights .* sin_vals_23);
    cos_weighted_23  = sum(weights .* cos_vals_23);
    bearing_weighted_rx23 = atan2d(sin_weighted_23, cos_weighted_23);
    error_weighted_rx23   = mod((bearing_weighted_rx23 - true_bearing_deg)+180,360)-180;

    % Scene-level agreement between the two independent antenna-pair estimates.
    scene_agree_diff = mod((bearing_weighted - bearing_weighted_rx23)+180,360)-180;

    fprintf('Number of confirmed clusters used: %d\n', nClusters);
    fprintf('True bearing (from lab geometry): %.1f deg\n', true_bearing_deg);
    fprintf('\nUnweighted circular mean bearing: %+.1f deg  (error: %+.1f deg)\n', ...
        bearing_unweighted, error_unweighted);
    fprintf('SNR-weighted circular mean bearing: %+.1f deg  (error: %+.1f deg)\n', ...
        bearing_weighted, error_weighted);
    fprintf('Inter-cluster circular std: %.1f deg\n', scene_circ_std);

    fprintf('\n--- RX2/RX3 Independent Cross-Check ---\n');
    fprintf('RX2/RX3 SNR-weighted scene bearing: %+.1f deg  (error vs true: %+.1f deg)\n', ...
        bearing_weighted_rx23, error_weighted_rx23);
    fprintf('Agreement (RX1/RX4 - RX2/RX3): %+.1f deg\n', scene_agree_diff);
    if abs(scene_agree_diff) < 5
        fprintf('  -> Estimates agree closely: geometry/wiring assumptions for both\n');
        fprintf('     antenna pairs appear consistent.\n\n');
    elseif abs(scene_agree_diff) < 15
        fprintf('  -> Moderate disagreement: worth checking RX2/RX3 spacing and\n');
        fprintf('     ordering assumptions before treating this as a firm cross-check.\n\n');
    else
        fprintf('  -> Large disagreement: RX2/RX3 spacing, ordering, or a channel-\n');
        fprintf('     specific issue (SNR/calibration) likely needs investigation.\n\n');
    end

    if abs(bearing_weighted) < 15
        dir_str = 'AHEAD (main lobe)';
    elseif bearing_weighted < 0
        dir_str = sprintf('LEFT (%.1f deg)', abs(bearing_weighted));
    else
        dir_str = sprintf('RIGHT (%.1f deg)', bearing_weighted);
    end
    fprintf('Estimated direction: %s\n', dir_str);

    if scene_circ_std < bearing_std_reliable
        conf_str = 'HIGH confidence';
    elseif scene_circ_std < bearing_std_poor
        conf_str = 'MODERATE confidence';
    else
        conf_str = 'LOW confidence';
    end
    fprintf('Estimation confidence: %s (inter-cluster std = %.1f deg)\n\n', ...
        conf_str, scene_circ_std);

    %% STEP 6d: FRAME-BY-FRAME BEARING DISTRIBUTION
    %
    %  Computes an independent bearing estimate for each spike frame,
    %  using only that frame's Interference detections, giving a
    %  time-series view of how stable the bearing estimate is.
    fprintf('--- STEP 6d: Frame-by-frame bearing distribution ---\n');

    spike_frames   = find(interfered_flags(1:Total_Audited));
    frame_bearings = nan(numel(spike_frames), 1);
    frame_counts   = zeros(numel(spike_frames), 1);

    for fi = 1:numel(spike_frames)
        fnum              = spike_frames(fi);
        frame_interf_idx = find(labels=="clustered interference" & det_frame==fnum);
        if numel(frame_interf_idx) < 2, continue; end

        rx1_f     = rx1_complex_all(frame_interf_idx);
        rx4_f     = rx4_complex_all(frame_interf_idx);
        coh_sum_f = sum(rx1_f .* conj(rx4_f));
        ph_f      = angle(coh_sum_f);
        sin_f     = max(-1, min(1, ph_f/pi));
        frame_bearings(fi) = rad2deg(asin(sin_f));
        frame_counts(fi)   = numel(frame_interf_idx);
    end

    valid_frame_bearings = frame_bearings(~isnan(frame_bearings));
    fprintf('Frames with reliable bearing estimate: %d / %d spike frames\n', ...
        numel(valid_frame_bearings), numel(spike_frames));

    if numel(valid_frame_bearings) > 1
        R_frame        = sqrt(mean(sind(valid_frame_bearings))^2 + ...
                              mean(cosd(valid_frame_bearings))^2);
        frame_circ_std = rad2deg(sqrt(-2*log(max(R_frame,1e-6))));
        frame_mean_bearing = atan2d(mean(sind(valid_frame_bearings)), ...
                                    mean(cosd(valid_frame_bearings)));
        fprintf('Frame-level mean bearing: %+.1f deg\n', frame_mean_bearing);
        fprintf('Frame-level circular std: %.1f deg\n\n', frame_circ_std);
    end

else
    fprintf('\n--- STEP 6b: Skipped (no confirmed interference clusters) ---\n');
    fprintf('Bearing estimation requires DBSCAN-confirmed interference clusters.\n\n');

    bearing_weighted   = NaN;
    bearing_unweighted = NaN;
    scene_circ_std     = NaN;
    conf_str           = 'N/A';
    error_weighted     = NaN;
    valid_frame_bearings = [];
    nClusters          = 0;

    bearing_weighted_rx23 = NaN;
    error_weighted_rx23   = NaN;
    scene_agree_diff      = NaN;
end

%% ================================================================
%  STEP 7: STATISTICS SUMMARY
%% ================================================================
fprintf('\n========== FINAL PARAMETER PROFILE ==========\n');
fprintf('File: %s  [%s]\n', testFile, config_label);
fprintf('True bearing: %.1f deg | ', true_bearing_deg);
if ~isnan(bearing_weighted)
    fprintf('Estimated: %+.1f deg | Error: %+.1f deg\n', ...
        bearing_weighted, error_weighted);
    fprintf('RX2/RX3 cross-check: %+.1f deg | Error: %+.1f deg | Agreement vs RX1/RX4: %+.1f deg\n', ...
        bearing_weighted_rx23, error_weighted_rx23, scene_agree_diff);
else
    fprintf('No bearing estimate (no confirmed clusters)\n');
end
fprintf('%-22s %8s %8s %8s %10s %10s\n', ...
    'Class','Count','SNR_mean','SNR_std','|Vel|_mean','Range_mean');
fprintf('%s\n', repmat('-',1,68));
allClasses = ["Noise","Static","Moving","clustered interference","isolated interference"];
for c = 1:numel(allClasses)
    idx = labels == allClasses(c);
    if sum(idx)==0, continue; end
    fprintf('%-22s %8d %8.2f %8.2f %10.3f %10.3f\n', ...
        allClasses(c), sum(idx), ...
        mean(det_snr(idx)), std(det_snr(idx)), ...
        mean(abs(det_vel(idx))), mean(det_range(idx)));
end
fprintf('\nTotal frames         : %d\n', Total_Audited);
fprintf('Raw spikes found     : %d\n', raw_spike_count);
fprintf('Interference declared: %s\n', string(Interfered_Count >= min_spike_count));

%% ================================================================
%  STEP 8: PLOTS
%
%  Fig 1 — Frame power time-series with spike markers
%  Fig 2 — Range vs |Velocity| scatter (classification view)
%  Fig 3 — Polar spatial detection map with centroids
%  Fig 4 — 3D accumulated map (X, Y, SNR)
%  Fig 5 — Per-frame detection count time-series
%  Fig 6 — SNR histogram per class
%  Fig 7 — Averaged Range-Doppler map with detection overlays
%  Fig A — Polar bearing plot (only if interference confirmed)
%% ================================================================

% Standard range gate, used for Static/Moving/Noise plotting.
inRange = det_range >= minPhysicalRange & det_range <= maxPhysicalRange;

% Relaxed range gate for interference classes, matching the wider gate
% used for interference candidacy in Step 4. Static/Moving/Noise
% plotting keeps using inRange; clustered/isolated interference uses
% this instead, so interference detected outside the real-target range
% window still appears in the figures.
inRangeInterf = det_range >= minInterfRange & det_range <= maxInterfRange;

nStatic = sum(labels=="Static");
nMoving = sum(labels=="Moving");
nInterf = sum(labels=="clustered interference");
nWeak   = sum(labels=="isolated interference");

% --- Fig 1: Frame power with spike detection ---
figure('Color','w','Name','Interference Spike Detection');
plot(frame_pwr,'LineWidth',1.2,'Color',[0.2 0.4 0.8],'DisplayName','Frame Power');
hold on;
moving_trend = movmean(frame_pwr, 10);
plot(moving_trend,'k--','LineWidth',1,'DisplayName','Moving avg (trend)');
if raw_spike_count > 0
    scatter(find(raw_spike_flags), frame_pwr(raw_spike_flags), 40, ...
        col_interf,'filled','DisplayName', ...
        sprintf('Spikes (%d found, %d declared)', raw_spike_count, Interfered_Count));
end
title(['Interference Spike Detection — ' testFile],'Interpreter','none');
xlabel('Frame'); ylabel('Avg Power (dB)');
legend('Location','best'); grid on;

% --- Fig 2: Range vs |Velocity| ---
% Noise is subsampled to 1500 points for rendering speed.
figure('Color','w','Name','Range vs |Velocity|');
hold on;
idx = labels=="Noise" & inRange;
if sum(idx)>0
    pIdx=find(idx);
    if numel(pIdx)>1500, pIdx=pIdx(randperm(numel(pIdx),1500)); end
    scatter(abs(det_vel(pIdx)),det_range(pIdx),6,col_noise,'o','filled',...
        'MarkerFaceAlpha',0.15,'DisplayName','Noise');
end
if nStatic>0
    scatter(abs(det_vel(labels=="Static")),det_range(labels=="Static"),...
        40,col_static,'s','filled','MarkerFaceAlpha',0.8,...
        'DisplayName',sprintf('Static (%d)',nStatic));
end
if nMoving>0
    scatter(abs(det_vel(labels=="Moving")),det_range(labels=="Moving"),...
        20,col_moving,'^','filled','MarkerFaceAlpha',0.7,...
        'DisplayName',sprintf('Moving (%d)',nMoving));
end
if nInterf>0
    scatter(abs(det_vel(labels=="clustered interference")),det_range(labels=="clustered interference"),...
        45,col_interf,'d','filled','MarkerFaceAlpha',0.95,...
        'DisplayName',sprintf('Clustered Interference (%d)',nInterf));
end
if nWeak>0
    scatter(abs(det_vel(labels=="isolated interference")),det_range(labels=="isolated interference"),...
        15,col_interf_weak,'o','filled','MarkerFaceAlpha',0.7,...
        'DisplayName',sprintf('Isolated Interference (%d)',nWeak));
end
xline(vel_static_thresh,'g--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_min,   'b:','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_max,   'b-','LineWidth',1.5,'HandleVisibility','off');
yline(minPhysicalRange,'k--','LineWidth',1,'HandleVisibility','off');
yline(maxPhysicalRange,'k-', 'LineWidth',1.5,'HandleVisibility','off');
xlabel('|Velocity| (m/s)'); ylabel('Range (m)');
title(sprintf('Range vs |Velocity| — %s', config_label));
legend('Location','best'); grid on;

% --- Fig 3: Polar spatial detection map ---
% Detections are plotted in polar coordinates (azimuth, range) to
% preserve the radar's natural measurement geometry. Particularly
% informative outdoors, where the wide range gate (12 m) and sparse
% reflector density make the polar view clearer than a Cartesian
% floor-plan.
% Convention: theta (azimuth) = 0 deg at top (boresight), clockwise
% positive; rho (range) = distance from radar in metres. The observer
% radar sits at the pole (origin).
figure('Color','w','Name','Polar Spatial Detection Map',...
    'Position',[150 100 700 650]);
ax_sp = polaraxes;
ax_sp.ThetaZeroLocation = 'top';
ax_sp.ThetaDir          = 'clockwise';
ax_sp.ThetaLim          = [-90 90];
% Radial limit must accommodate the wider interference range gate as
% well as the target range gate, or far interference detections would
% be clipped.
maxDisplayRange         = max(maxPhysicalRange, maxInterfRange);
ax_sp.RLim              = [0, maxDisplayRange * 1.05];
ax_sp.RTickLabel        = compose('%.0f m', ax_sp.RTick);
ax_sp.GridColor         = [0.6 0.6 0.6];
ax_sp.GridAlpha         = 0.5;
hold(ax_sp, 'on');

idx = labels=="Noise" & inRange;
if sum(idx)>0
    pIdx = find(idx);
    if numel(pIdx)>1000, pIdx = pIdx(randperm(numel(pIdx),1000)); end
    polarscatter(ax_sp, deg2rad(det_az_deg(pIdx)), det_range(pIdx), ...
        5, col_noise, 'o', 'filled', 'MarkerFaceAlpha', 0.10, ...
        'DisplayName', 'Noise');
end
if nStatic>0
    polarscatter(ax_sp, deg2rad(det_az_deg(labels=="Static")), ...
        det_range(labels=="Static"), ...
        40, col_static, 's', 'filled', 'MarkerFaceAlpha', 0.85, ...
        'DisplayName', sprintf('Static (%d)', nStatic));
end
if nMoving>0
    polarscatter(ax_sp, deg2rad(det_az_deg(labels=="Moving")), ...
        det_range(labels=="Moving"), ...
        20, col_moving, '^', 'filled', 'MarkerFaceAlpha', 0.65, ...
        'DisplayName', sprintf('Moving (%d)', nMoving));
end
if nInterf>0
    polarscatter(ax_sp, deg2rad(det_az_deg(labels=="clustered interference")), ...
        det_range(labels=="clustered interference"), ...
        55, col_interf, 'd', 'filled', 'MarkerFaceAlpha', 0.95, ...
        'DisplayName', sprintf('Clustered Interference (%d)', nInterf));
end
if nWeak>0
    polarscatter(ax_sp, deg2rad(det_az_deg(labels=="isolated interference")), ...
        det_range(labels=="isolated interference"), ...
        12, col_interf_weak, 'o', 'filled', 'MarkerFaceAlpha', 0.65, ...
        'DisplayName', sprintf('Isolated Interference (%d)', nWeak));
end

% Cluster centroids — pentagram markers, one legend entry per class.
if ~isempty(centroid_list)
    sDone=false; mDone=false; iDone=false;
    for i = 1:size(centroid_list,1)
        cc     = centroid_list(i,1);
        cX_c   = centroid_list(i,6);   % mean X
        cY_c   = centroid_list(i,7);   % mean Y
        cAz_deg = rad2deg(atan2(cX_c, cY_c));  % azimuth recomputed from mean X/Y
        cRng   = centroid_list(i,3);
        col_c  = colMap{cc};  lbl_c = lblMap{cc};
        switch cc
            case 1, showLeg=~sDone; sDone=true; csz=180;
            case 2, showLeg=~mDone; mDone=true; csz=180;
            case 3, showLeg=~iDone; iDone=true; csz=150;
        end
        if showLeg
            polarscatter(ax_sp, deg2rad(cAz_deg), cRng, csz, col_c, 'p', ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'DisplayName', lbl_c);
        else
            polarscatter(ax_sp, deg2rad(cAz_deg), cRng, csz, col_c, 'p', ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5, ...
                'HandleVisibility', 'off');
        end
    end
end

% Range gate boundary arc.
theta_arc = linspace(-pi/2, pi/2, 200);
polarplot(ax_sp, theta_arc, repmat(maxPhysicalRange, 1, 200), ...
    'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');

% Relaxed interference-candidacy range boundary, drawn only if it
% extends beyond the target range gate.
if maxInterfRange > maxPhysicalRange
    polarplot(ax_sp, theta_arc, repmat(maxInterfRange, 1, 200), ...
        'Color', col_interf, 'LineStyle', ':', 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
end

title(ax_sp, ...
    {sprintf('Polar Spatial Map — %s', config_label); ...
     sprintf('%d frames | Static:%d | Moving:%d | Clustered:%d | Isolated:%d | Spikes:%d', ...
     Total_Audited, nStatic, nMoving, nInterf, nWeak, Interfered_Count)}, ...
    'FontSize', 9, 'Interpreter', 'none');

legend(ax_sp, 'Location', 'southoutside', 'NumColumns', 3, 'FontSize', 8);

if Interfered_Count >= min_spike_count
    annotation('textbox', [0.10 0.01 0.80 0.04], ...
        'String', sprintf('Interference spikes: %d/%d (%.1f%%)  |  Clustered: %d  |  Isolated: %d', ...
        Interfered_Count, Total_Audited, 100*Interfered_Count/Total_Audited, nInterf, nWeak), ...
        'FitBoxToText', 'on', 'BackgroundColor', [1 0.93 0.93], ...
        'EdgeColor', col_interf, 'Color', col_interf, 'FontSize', 9);
end

% --- Fig 4: 3D accumulated map ---
% SNR is used as the Z-axis so strong/weak detections separate visually.
% Interference classes are plotted with the relaxed range mask
% (inRangeInterf) since interference beat artefacts are not physical-
% range measurements (see Step 4); Static/Moving/Noise use the standard
% range mask (inRange).
figure('Color','w','Name','3D Accumulated Detection Map');
hold on;
clsMap3D = {"Noise",col_noise,6,0.10,'o',false; ...
            "Static",col_static,50,0.85,'s',false; ...
            "Moving",col_moving,20,0.60,'^',false; ...
            "clustered interference",col_interf,60,0.95,'d',true; ...
            "isolated interference",col_interf_weak,12,0.65,'o',true};
for c=1:size(clsMap3D,1)
    cl=clsMap3D{c,1}; col=clsMap3D{c,2};
    sz=clsMap3D{c,3}; alp=clsMap3D{c,4}; mkr=clsMap3D{c,5};
    useInterfRange = clsMap3D{c,6};
    if useInterfRange
        idx = labels==cl & inRangeInterf;
    else
        idx = labels==cl & inRange;
    end
    if sum(idx)==0, continue; end
    pIdx=find(idx);
    if strcmp(cl,'Noise') && numel(pIdx)>1000
        pIdx=pIdx(randperm(numel(pIdx),1000));
    end
    dispName = cl;
    if cl=="clustered interference", dispName="Clustered Interference"; end
    if cl=="isolated interference",  dispName="Isolated Interference";  end
    scatter3(det_x(pIdx),det_y(pIdx),det_snr(pIdx),sz,col,mkr,...
        'filled','MarkerFaceAlpha',alp,'DisplayName',...
        sprintf('%s (%d)',dispName,sum(idx)));
end
if ~isempty(centroid_list)
    sDone=false; mDone=false; iDone=false;
    for i=1:size(centroid_list,1)
        cc=centroid_list(i,1); cX=centroid_list(i,6); cY=centroid_list(i,7);
        cSNR=centroid_list(i,5); col=colMap{cc}; lbl=lblMap{cc};
        switch cc
            case 1, showLeg=~sDone; sDone=true;
            case 2, showLeg=~mDone; mDone=true;
            case 3, showLeg=~iDone; iDone=true;
        end
        if showLeg
            scatter3(cX,cY,cSNR,200,col,'p','filled','MarkerEdgeColor','k',...
                'LineWidth',1.5,'DisplayName',lbl);
        else
            scatter3(cX,cY,cSNR,200,col,'p','filled','MarkerEdgeColor','k',...
                'LineWidth',1.5,'HandleVisibility','off');
        end
    end
end
scatter3(0,0,0,150,'bs','filled','MarkerEdgeColor','k','DisplayName','Observer Radar');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('SNR (dB)');
title(sprintf('3D Accumulated Map — %s', config_label));
legend('Location','best'); grid on; view(30,30);

% --- Fig 5: Per-frame detection count ---
figure('Color','w','Name','Detections per Frame');
hold on;
frameCount = zeros(Total_Audited, 4);
for f = 1:Total_Audited
    fi = det_frame==f;
    frameCount(f,1) = sum(labels=="Static"       & fi);
    frameCount(f,2) = sum(labels=="Moving"       & fi);
    frameCount(f,3) = sum(labels=="clustered interference" & fi);
    frameCount(f,4) = sum(labels=="isolated interference"   & fi);
end
plot(frameCount(:,1),'Color',col_static,'LineWidth',1.5,'DisplayName','Static');
plot(frameCount(:,2),'Color',col_moving,'LineWidth',1.5,'DisplayName','Moving');
plot(frameCount(:,3),'Color',col_interf,'LineWidth',2.0,'DisplayName','Clustered Interference');
plot(frameCount(:,4),'Color',col_interf_weak,'LineWidth',1.0,'DisplayName','Isolated Interference');
if Interfered_Count > 0
    for ifr = find(interfered_flags(1:Total_Audited))'
        xline(ifr,'Color',col_interf,'Alpha',0.20,'LineWidth',1,'HandleVisibility','off');
    end
end
xlabel('Frame'); ylabel('Count');
title(sprintf('Detections per Frame — %s', config_label));
legend('Location','best'); grid on;

% --- Fig 6: SNR distribution ---
figure('Color','w','Name','SNR Distribution');
hold on;
clsSNR = {"Noise",col_noise,false;"Static",col_static,false;"Moving",col_moving,false; ...
          "clustered interference",col_interf,true;"isolated interference",col_interf_weak,true};
for c=1:size(clsSNR,1)
    cl=clsSNR{c,1}; col=clsSNR{c,2}; useInterfRange=clsSNR{c,3};
    if useInterfRange
        idx = labels==cl & inRangeInterf;
    else
        idx = labels==cl & inRange;
    end
    if sum(idx)<5, continue; end
    histogram(det_snr(idx),50,'FaceColor',col,'FaceAlpha',0.6,...
        'EdgeColor','none','DisplayName',cl);
end
xline(snr_noise_thresh,'k:','LineWidth',1.5,...
    'DisplayName',sprintf('Noise gate (%ddB)',snr_noise_thresh));
xline(snr_static_min,'--','Color',col_static,'LineWidth',1.5,...
    'DisplayName',sprintf('Static min (%ddB)',snr_static_min));
xline(snr_moving_min,'--','Color',col_moving,'LineWidth',1.5,...
    'DisplayName',sprintf('Moving min (%ddB)',snr_moving_min));
xline(snr_interference_min,'--','Color',col_interf,'LineWidth',1.5,...
    'DisplayName',sprintf('Interf min (%ddB)',snr_interference_min));
xlabel('SNR (dB)'); ylabel('Count');
title(sprintf('SNR Distribution per Class — %s', config_label));
legend('Location','northeast'); grid on;

%% ================================================================
%  FIG 7: AVERAGED RANGE-DOPPLER MAP WITH DETECTION OVERLAYS
%
%  Background: time-averaged RD power map (jet colormap, dark axes).
%  Overlay: CFAR detections colour-coded by class, cluster centroids, DC
%  notch boundaries, and velocity threshold lines. Averaging reduces
%  frame-to-frame noise and makes persistent clutter/interference
%  patterns clearly visible.
%% ================================================================
vel_axis   = ((0:numChirps-1) - zeroVelBin + 1) * velRes;
range_axis = (0:numADCSamples-1) * rangeRes;

figure('Color','k','Name','Range-Doppler Map');
ax = axes('Color','k');

imagesc(ax, vel_axis, range_axis, rd_map_avg);
set(ax, 'YDir', 'normal');
colormap(ax, jet);
cb = colorbar(ax);
cb.Color = 'w';
cb.Label.String = 'Power (dB)';
cb.Label.Color  = 'w';

% Colorscale clipped to the 2nd-99th percentile to suppress outliers.
cLims = prctile(rd_map_avg(:), [2 99]);
clim(ax, cLims);

hold(ax, 'on');

dc_vel_lo = (dcLo - zeroVelBin) * velRes;
dc_vel_hi = (dcHi - zeroVelBin) * velRes;
xline(ax, dc_vel_lo, 'w--', 'LineWidth', 0.8, 'Alpha', 0.6, 'HandleVisibility','off');
xline(ax, dc_vel_hi, 'w--', 'LineWidth', 0.8, 'Alpha', 0.6, 'HandleVisibility','off');
yline(ax, minPhysicalRange, 'w:', 'LineWidth', 1.0, 'HandleVisibility','off');
yline(ax, maxPhysicalRange, 'w:', 'LineWidth', 1.0, 'HandleVisibility','off');

idx = labels=="Noise" & inRange;
if sum(idx) > 0
    pIdx = find(idx);
    if numel(pIdx) > 800, pIdx = pIdx(randperm(numel(pIdx), 800)); end
    scatter(ax, det_vel(pIdx), det_range(pIdx), 4, col_noise, 'o', ...
        'filled', 'MarkerFaceAlpha', 0.20, 'DisplayName', 'Noise');
end
if nStatic > 0
    scatter(ax, det_vel(labels=="Static"), det_range(labels=="Static"), ...
        35, col_static, 's', 'filled', 'MarkerFaceAlpha', 0.85, ...
        'MarkerEdgeColor', 'none', 'DisplayName', sprintf('Static (%d)', nStatic));
end
if nMoving > 0
    scatter(ax, det_vel(labels=="Moving"), det_range(labels=="Moving"), ...
        20, col_moving, '^', 'filled', 'MarkerFaceAlpha', 0.80, ...
        'MarkerEdgeColor', 'none', 'DisplayName', sprintf('Moving (%d)', nMoving));
end
if nInterf > 0
    scatter(ax, det_vel(labels=="clustered interference"), det_range(labels=="clustered interference"), ...
        50, col_interf, 'd', 'filled', 'MarkerFaceAlpha', 1.0, ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.5, ...
        'DisplayName', sprintf('Clustered Interference (%d)', nInterf));
end
if nWeak > 0
    scatter(ax, det_vel(labels=="isolated interference"), det_range(labels=="isolated interference"), ...
        12, col_interf_weak, 'o', 'filled', 'MarkerFaceAlpha', 0.70, ...
        'MarkerEdgeColor', 'none', ...
        'DisplayName', sprintf('Isolated Interference (%d)', nWeak));
end

if ~isempty(centroid_list)
    sDone=false; mDone=false; iDone=false;
    for i = 1:size(centroid_list,1)
        cc   = centroid_list(i,1);
        cVel = centroid_list(i,4);
        cRng = centroid_list(i,3);
        col  = colMap{cc};
        lbl  = lblMap{cc};
        switch cc
            case 1, showLeg=~sDone; sDone=true;
            case 2, showLeg=~mDone; mDone=true;
            case 3, showLeg=~iDone; iDone=true;
        end
        if showLeg
            scatter(ax, cVel, cRng, 160, col, 'p', 'filled', ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'DisplayName', lbl);
        else
            scatter(ax, cVel, cRng, 160, col, 'p', 'filled', ...
                'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'HandleVisibility','off');
        end
    end
end

xline(ax,  vel_static_thresh, '--', 'Color', [0.4 1 0.4], 'LineWidth', 1.0, ...
    'Label', sprintf('v_{static}=%.2f m/s', vel_static_thresh), ...
    'LabelColor', [0.4 1 0.4], 'FontSize', 8, 'HandleVisibility', 'off');
xline(ax, -vel_static_thresh, '--', 'Color', [0.4 1 0.4], 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
xline(ax,  vel_moving_max, '-', 'Color', [0.5 0.7 1], 'LineWidth', 1.2, ...
    'Label', sprintf('v_{max}=%.2f m/s', vel_moving_max), ...
    'LabelColor', [0.5 0.7 1], 'FontSize', 8, 'HandleVisibility', 'off');
xline(ax, -vel_moving_max, '-', 'Color', [0.5 0.7 1], 'LineWidth', 1.2, ...
    'HandleVisibility', 'off');
xline(ax, 0, 'w-', 'LineWidth', 0.6, 'Alpha', 0.4, 'HandleVisibility', 'off');

xlabel(ax, 'Velocity (m/s)', 'Color', 'w');
ylabel(ax, 'Range (m)', 'Color', 'w');
set(ax, 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.5 0.5 0.5], ...
    'GridAlpha', 0.3, 'FontSize', 9);
xlim(ax, [-velMax velMax]);
ylim(ax, [0 max(rangeMax, maxInterfRange)]);
grid(ax, 'on');

title(ax, {sprintf('Range-Doppler Map (averaged, %d frames) — %s', ...
           frames_accumulated, config_label); ...
           sprintf('Static:%d | Moving:%d | Interf(clustered):%d | Interf(isolated):%d', ...
           nStatic, nMoving, nInterf, nWeak)}, ...
    'Color', 'w', 'Interpreter', 'none');

lgd = legend(ax, 'Location', 'northeast', 'TextColor', 'w', ...
    'Color', [0.15 0.15 0.15], 'EdgeColor', [0.4 0.4 0.4]);

if Interfered_Count >= min_spike_count
    annotation('textbox', [0.13 0.01 0.60 0.04], ...
        'String', sprintf('Spikes: %d/%d (%.1f%%)  |  DC notch: bins %d–%d  |  Clustered: %d  |  Isolated: %d', ...
        Interfered_Count, Total_Audited, 100*Interfered_Count/Total_Audited, ...
        dcLo, dcHi, nInterf, nWeak), ...
        'FitBoxToText', 'on', 'BackgroundColor', [0.15 0.05 0.05], ...
        'EdgeColor', col_interf, 'Color', col_interf, 'FontSize', 8);
end

%% ================================================================
%  BEARING PLOTS
%  (rendered only when at least one confirmed interference cluster exists)
%% ================================================================
if nConfirmedInterf > 0

    % --- Fig A: Polar bearing plot ---
    % Per-cluster bearing estimates (colour-coded by reliability),
    % alongside the SNR-weighted and unweighted scene estimates, the
    % RX2/RX3 cross-check, and the true bearing from lab geometry.
    figure('Color','w','Name','Interference Bearing Estimation — Polar',...
        'Position',[100 100 700 700]);
    ax_polar = polaraxes;
    ax_polar.ThetaZeroLocation = 'top';
    ax_polar.ThetaDir          = 'clockwise';
    ax_polar.ThetaLim          = [-90 90];
    ax_polar.RLim              = [0 1.2];
    ax_polar.RTick             = [];
    ax_polar.RTickLabel        = {};
    ax_polar.GridColor         = [0.8 0.8 0.8];
    hold(ax_polar, 'on');

    for ci = 1:nClusters
        bear_rad_ci = deg2rad(cluster_bearing_deg(ci));
        if cluster_reliability(ci) == "High"
            lc = [0.85 0.10 0.10];
        elseif cluster_reliability(ci) == "Moderate"
            lc = [0.95 0.55 0.10];
        else
            lc = [0.80 0.80 0.80];
        end
        polarplot(ax_polar, bear_rad_ci, 1.0, 'o', ...
            'MarkerFaceColor', lc, 'MarkerEdgeColor', 'k', ...
            'MarkerSize', 8+cluster_snr_db(ci)/5);
        polarplot(ax_polar, [0 bear_rad_ci], [0 0.9], '-', ...
            'Color', [lc 0.5], 'LineWidth', 1.2);
        text(ax_polar, bear_rad_ci, 1.08, sprintf('C%d', ci), ...
            'HorizontalAlignment','center','FontSize',8,'Color','k');
    end

    % SNR-weighted estimate (primary result).
    bear_w_rad = deg2rad(bearing_weighted);
    polarplot(ax_polar, [0 bear_w_rad], [0 1.0], 'r-', 'LineWidth', 3);
    polarplot(ax_polar, bear_w_rad, 1.0, 'r^', ...
        'MarkerFaceColor','r','MarkerSize',12);

    % True bearing (ground truth from lab geometry).
    true_bear_rad = deg2rad(true_bearing_deg);
    polarplot(ax_polar, [0 true_bear_rad], [0 1.0], 'k--', 'LineWidth', 2);
    polarplot(ax_polar, true_bear_rad, 1.0, 'ks', ...
        'MarkerFaceColor','k','MarkerSize',10);

    % Unweighted estimate (reference).
    bear_uw_rad = deg2rad(bearing_unweighted);
    polarplot(ax_polar, [0 bear_uw_rad], [0 0.85], 'b--', 'LineWidth', 1.5);
    polarplot(ax_polar, bear_uw_rad, 0.85, 'b^', ...
        'MarkerFaceColor','b','MarkerSize',9);

    % RX2/RX3 independent cross-check estimate.
    bear_23_rad = deg2rad(bearing_weighted_rx23);
    polarplot(ax_polar, [0 bear_23_rad], [0 0.95], 'm-.', 'LineWidth', 2);
    polarplot(ax_polar, bear_23_rad, 0.95, 'md', ...
        'MarkerFaceColor','m','MarkerSize',9);

    title(ax_polar, ...
        {sprintf('Interference Source Bearing Estimation — %s', config_label); ...
         sprintf('RX1/RX4: %+.1f deg | RX2/RX3: %+.1f deg | True: %+.1f deg | Agreement: %+.1f deg', ...
         bearing_weighted, bearing_weighted_rx23, true_bearing_deg, scene_agree_diff)}, ...
        'FontSize', 10);

    annotation('textbox',[0.72 0.20 0.25 0.25], ...
        'String', {'\color[rgb]{0.85,0.10,0.10}\bullet Per-cluster (high)', ...
                   '\color[rgb]{0.95,0.55,0.10}\bullet Per-cluster (mod)', ...
                   '\color[rgb]{0.80,0.80,0.80}\bullet Per-cluster (low)', ...
                   '{\color{red}\rightarrow} RX1/RX4 weighted (primary)', ...
                   '{\color{blue}--} RX1/RX4 unweighted', ...
                   '{\color{magenta}-.} RX2/RX3 cross-check', ...
                   '{\color{black}--} True bearing'}, ...
        'FitBoxToText','on','BackgroundColor','w','EdgeColor',[0.8 0.8 0.8],...
        'FontSize',8,'Interpreter','tex');

end

fprintf('\n=== Complete [%s config] ===\n', config_label);
fprintf('Change testFile, numADCSamples, detection_sensitivity and true_bearing_deg at top for next dataset.\n');