%% =========================================================================
%  Radar Detector + Classifier + DBSCAN + X/Y Spatial Plot +
%  Range-Doppler Map + Interference Bearing Estimation
%
%  PURPOSE:
%    Full processing pipeline for a TI IWR6843ISK 60 GHz FMCW radar.
%    Reads raw ADC binary data, computes Range-Doppler maps, detects
%    targets using CFAR, classifies them (Static / Moving / Interference),
%    clusters confirmed detections with DBSCAN, and estimates the azimuth
%    bearing of any interference source via coherent RX1/RX4 phase comparison.
%
%  SUPPORTS:  128-sample and 256-sample ADC configurations (auto-configured)
%
%  USAGE:
%    1. Set testFile, numADCSamples, and true_bearing_deg at the top.
%    2. Run the script — all figures and console output are generated.
% =========================================================================
%  Integrated 128 and 256 ADC sample configurations
clear; clc; close all;

%% ================================================================
%  CONFIGURATION — only change these lines between datasets
%% ================================================================
% Path to raw binary ADC file captured from the radar sensor
testFile         = 'adc_data_with_moving_iinterference0_2_Raw_0.bin';

% Number of ADC samples per chirp — must match the capture configuration
% Supported values: 128 or 256
numADCSamples    = 256;   % set to 128 or 256 to match the dataset

% Known true bearing of the interferer from lab geometry measurements.
% Used only for error computation; does not affect processing.
% Convention: 0 = ahead, negative = left, positive = right
true_bearing_deg = 0;

%% ================================================================
%  HARDWARE PARAMETERS — same for both configs
%% ================================================================
% Number of active receive antennas on the IWR6843ISK
numRX            = 4;

% Number of chirps transmitted per frame (slow-time dimension)
numChirps        = 128;

% Maximum number of frames to read from the file during the audit pass
numFramesToAudit = 1000;

% Sensitivity for local neighbourhood spike detection in the frame-power
% sequence. A frame is flagged if its power exceeds both neighbours by
% at least this many dB.
detection_sensitivity = 1.5;   % local neighbourhood spike sensitivity. 1 for 128 and 1.5 for 256 ADC sample configurations 

% --- Waveform parameters (must match the radar profile) ---
fc          = 60e9;           % carrier frequency (Hz)
freqSlope   = 29.982e6 / 1e-6;% chirp frequency slope (Hz/s)
fs          = 10e6;           % ADC sampling rate (Hz)
idleTime    = 100e-6;         % idle time between chirps (s)
rampEndTime = 57.14e-6;       % active ramp duration (s)
chirpTime   = idleTime + rampEndTime; % total time per chirp (s)

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
%  AUTO-CONFIGURE THRESHOLDS BASED ON ADC SAMPLE COUNT
%% ================================================================
% CFAR sliding-window parameters (1D range-axis CFAR)
cfar_guard = 1;   % guard cells on each side of the cell under test
cfar_train = 4;   % training cells on each side

% Probability of false alarm — drives the CFAR threshold multiplier alpha
cfar_pfa   = 1e-3;

% First and last valid range bins after accounting for guard+training margin
first_valid_bin   = cfar_guard + cfar_train + 1;
last_valid_bin    = numADCSamples - cfar_guard - cfar_train;

% Physical range corresponding to the first valid bin
first_valid_range = (first_valid_bin - 1) * rangeRes;

% All detection and classification thresholds are set per ADC configuration
% so the pipeline works correctly without manual adjustment when switching
% between 128-sample and 256-sample datasets.
if numADCSamples == 128
    % --- 128-sample configuration thresholds ---
    minPhysicalRange     = first_valid_range; % minimum valid target range (m)
    maxPhysicalRange     = 5.0;               % room boundary (m)
    snr_noise_thresh     = 8;                 % minimum SNR to survive noise gate (dB)
    snr_static_min       = 19;                % minimum SNR to be labelled Static (dB)
    snr_moving_min       = 16;                % minimum SNR to be labelled Moving (dB)
    snr_interference_min = 16;                % minimum SNR to be promoted to Interference (dB)
    vel_static_thresh    = 0.15;              % velocity below this → Static (m/s)
    vel_moving_min       = 0.16;              % velocity above this → Moving candidate (m/s)
    vel_moving_max       = 1.00;              % velocity above this → not a pedestrian (m/s)
    persist_interf_max   = 0.05;              % range bins appearing in < this fraction of frames → transient (interference) 
    min_spike_count      = 1;                 % minimum frame-power spikes to declare interference
    dcNotch              = 2;                 % half-width of DC (zero-velocity) notch in bins
    dbscan_eps_static    = 0.5;              % DBSCAN neighbourhood radius for Static (normalised bins)
    dbscan_eps_moving    = 0.5;              % DBSCAN neighbourhood radius for Moving
    dbscan_eps_interf    = 1.5;              % DBSCAN neighbourhood radius for Interference 
    dbscan_minpts        = 10;               % minimum cluster size for Static/Moving
    dbscan_minpts_interf = 5;                % minimum cluster size for Interference
    config_label         = '128-sample';

elseif numADCSamples == 256
    % --- 256-sample configuration thresholds ---
    % Higher range resolution means more bins → stricter static SNR gate
    % and more detections per cluster → higher min_spike_count.
    minPhysicalRange     = first_valid_range;
    maxPhysicalRange     = 5.0;
    snr_noise_thresh     = 8;
    snr_static_min       = 28;   % higher than 128-sample due to better resolution
    snr_moving_min       = 16;
    snr_interference_min = 16;
    vel_static_thresh    = 0.15;
    vel_moving_min       = 0.16;
    vel_moving_max       = 1.50; % slightly relaxed upper bound for 256-sample
    persist_interf_max   = 0.05;
    min_spike_count      = 5;    % more spikes expected at finer resolution
    dcNotch              = 2;
    dbscan_eps_static    = 0.5;
    dbscan_eps_moving    = 0.5;
    dbscan_eps_interf    = 1.5;
    dbscan_minpts        = 10;
    dbscan_minpts_interf = 5;
    config_label         = '256-sample';

else
    error('Unsupported numADCSamples: %d. Use 128 or 256.', numADCSamples);
end

% --- Bearing estimation quality thresholds ---
% A detection contributes to bearing only if its SNR exceeds this gate
bearing_snr_min      = 16;   % dB — minimum SNR for a detection to contribute

% Circular standard deviation thresholds for per-cluster reliability labels
bearing_std_reliable = 20;   % deg — below this → "High" confidence
bearing_std_poor     = 35;   % deg — above this → "Low" confidence (between = "Moderate")

fprintf('\n--- Thresholds (%s) ---\n', config_label);
fprintf('Range gate:          %.3f to %.1f m\n', minPhysicalRange, maxPhysicalRange);
fprintf('SNR noise gate:      %d dB\n',  snr_noise_thresh);
fprintf('SNR static min:      %d dB\n',  snr_static_min);
fprintf('SNR moving min:      %d dB\n',  snr_moving_min);
fprintf('SNR interf min:      %d dB\n',  snr_interference_min);
fprintf('Vel static max:      %.2f m/s\n', vel_static_thresh);
fprintf('Vel moving:          %.2f to %.2f m/s\n', vel_moving_min, vel_moving_max);
fprintf('Min spike count:     %d\n', min_spike_count);
fprintf('--------------------------------------\n\n');

%% --- Colours for classification labels (used consistently across all plots) ---
col_noise        = [0.75 0.75 0.75];  % light grey  — noise / unclassified
col_static       = [0.13 0.55 0.13];  % green       — static targets
col_moving       = [0.12 0.35 0.75];  % blue        — moving targets (pedestrians)
col_interf       = [0.85 0.10 0.10];  % red         — clustered interference
col_interf_weak  = [0.95 0.55 0.55];  % pink/light  — isolated interference

%% --- Hann windows for range (fast-time) and Doppler (slow-time) FFTs ---
% Applied before FFT to suppress spectral sidelobes
hS = 0.5*(1 - cos(2*pi*(0:numADCSamples-1)'/(numADCSamples-1))); % range window (column)
hC = 0.5*(1 - cos(2*pi*(0:numChirps-1)'/(numChirps-1)));          % Doppler window (row)

% Byte count for one complete frame of raw ADC data
% Factor 4: each complex sample = 2 int16 values (I and Q) = 4 bytes
bytesPerFrame = numADCSamples * numRX * numChirps * 4;

% Zero-velocity (DC) bin index in the fftshift-ed Doppler axis
zeroVelBin    = ceil(numChirps/2) + 1;

%% ================================================================
%  STEP 1: FRAME POWER + LOCAL NEIGHBOURHOOD SPIKE DETECTION
%
%  PURPOSE:
%    Do a lightweight first pass over the file to compute the mean
%    Range-Doppler power per frame.  Frames whose power forms a local
%    maximum (spike) relative to their immediate neighbours are flagged
%    as potentially interference-affected.  This is a fast pre-screen
%    before the more expensive per-frame CFAR pass in Step 2.
%% ================================================================
fprintf('--- STEP 1: Frame power + interference spike detection ---\n');
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

% Pre-allocate frame power vector for the audit window
frame_pwr = zeros(numFramesToAudit, 1);

for f = 1:numFramesToAudit
    % Read one frame of raw LVDS data (interleaved I/Q int16)
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if isempty(raw), break; end

    % Reconstruct complex samples: odd indices = I, even = Q
    LVDS     = raw(1:2:end) + 1i*raw(2:2:end);

    % Reshape to [numADCSamples*numRX × numChirps] then extract RX1
    LVDS     = reshape(LVDS, numADCSamples*numRX, numChirps).';
    rx1_data = LVDS(:, 1:numADCSamples).';  % [numADCSamples × numChirps]

    % Compute 2-D Range-Doppler map (dB) for RX1 only
    rd_db    = 20*log10(abs(fftshift( ...
                   fft(fft(rx1_data.*hS,numADCSamples,1).*hC',numChirps,2),2)));

    % Store mean power as a scalar representative of this frame's energy
    frame_pwr(f) = mean(rd_db(:));
end
fclose(fid);

% Trim to actual number of frames read
Total_Audited = find(frame_pwr ~= 0, 1, 'last');
frame_pwr     = frame_pwr(1:Total_Audited);

% Detect local maxima: a frame is a spike if it exceeds both neighbours
% by at least detection_sensitivity dB (simple non-maximum suppression)
raw_spike_flags = false(Total_Audited, 1);
for i = 2:(Total_Audited-1)
    if (frame_pwr(i) > frame_pwr(i-1) + detection_sensitivity) && ...
       (frame_pwr(i) > frame_pwr(i+1) + detection_sensitivity)
        raw_spike_flags(i) = true;
    end
end
raw_spike_count = sum(raw_spike_flags);

% Declare interference only if enough spikes were found (min_spike_count gate)
% This prevents false declarations from isolated noise fluctuations.
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
%  PURPOSE:
%    Main detection pass.  For each frame:
%      1. Compute complex Range-Doppler maps for RX1 and RX4.
%      2. Apply 1D CFAR (cell-averaging) along the range axis for each
%         Doppler bin to produce a detection mask.
%      3. For DC-notch bins, apply a higher SNR gate (snr_static_min)
%         to reduce false alarms from static clutter.
%      4. For confirmed detections, compute azimuth angle from the
%         phase difference between RX1 and RX4, and convert to X/Y.
%      5. Accumulate RD maps across frames for the averaged Fig 7 display.
%
%  Also stores complex RX1/RX4 values per detection for bearing estimation
%% ================================================================
fprintf('--- STEP 2: Processing %d frames ---\n', Total_Audited);
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

% Cell array: frame_dets{f} = [frame, range, vel, snr, peak, az_deg, x, y]
frame_dets = cell(Total_Audited, 1);

% Complex value storage for bearing estimation
% Preserves full phase information for coherent averaging
rx1_complex_store = cell(Total_Audited, 1);
rx4_complex_store = cell(Total_Audited, 1);

% CFAR threshold multiplier alpha derived from the CA-CFAR formula:
%   alpha = N * (PFA^(-1/N) - 1)   where N = number of training cells
numTrain_val = 2 * cfar_train;
alpha_cfar   = numTrain_val * (cfar_pfa^(-1/numTrain_val) - 1);
alpha_dB     = 10*log10(alpha_cfar); % pre-converted to dB for speed

% Accumulators for producing the time-averaged Range-Doppler map (Fig 7)
rd_map_accum       = zeros(numADCSamples, numChirps);
frames_accumulated = 0;

for f = 1:Total_Audited
    % Read one frame of raw binary data
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if numel(raw) < bytesPerFrame/2
        fprintf('End of file at frame %d\n', f);
        Total_Audited = f-1; break;
    end

    % Reconstruct complex LVDS stream and demultiplex into 4 RX channels
    LVDS_all = raw(1:2:end) + 1i*raw(2:2:end);
    LVDS_mat = reshape(LVDS_all, numADCSamples*numRX, numChirps).';

    % rx{rxIdx} = [numADCSamples × numChirps] complex fast-time / slow-time matrix
    rx = cell(4,1);
    for rxIdx = 1:numRX
        rx{rxIdx} = LVDS_mat(:, rxIdx:numRX:numADCSamples*numRX).';
    end

    % Compute complex Range-Doppler maps for RX1 and RX4
    % fftshift centres zero-velocity at column zeroVelBin
    rd_complex_rx1 = fftshift( ...
        fft(fft(rx{1}.*hS,numADCSamples,1).*hC',numChirps,2), 2);
    rd_complex_rx4 = fftshift( ...
        fft(fft(rx{4}.*hS,numADCSamples,1).*hC',numChirps,2), 2);

    % Magnitude (dB) map — used for CFAR and SNR computation
    rd_db = 20*log10(abs(rd_complex_rx1) + eps);

    % Accumulate for frame-averaged display map
    rd_map_accum       = rd_map_accum + rd_db;
    frames_accumulated = frames_accumulated + 1;

    % Per-frame noise floor estimate: mean power across the entire RD map
    noiseFloor_dB = mean(rd_db(:));

    % --- DC notch: zero out the zero-velocity strip to suppress static clutter
    % before spike-based CFAR, but keep original rd_db for static target detection
    dcLo = max(1, zeroVelBin - dcNotch);
    dcHi = min(numChirps, zeroVelBin + dcNotch);

    rd_mag_notched = abs(rd_complex_rx1);
    rd_mag_notched(:, dcLo:dcHi) = 0;    % zero out DC bins
    rd_db_notched  = 20*log10(rd_mag_notched + eps);

    % Pre-allocate detection mask for this frame
    cfar_mask = false(numADCSamples, numChirps);

    % --- 1D CA-CFAR along range for non-DC Doppler bins ---
    % The threshold is: cell_under_test > mean(training cells) + alpha_dB
    % AND SNR above noise gate
    for d = 1:numChirps
        if d >= dcLo && d <= dcHi, continue; end  % skip DC bins here
        col = rd_db_notched(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB ...
               && (col(r)-noiseFloor_dB) > snr_noise_thresh
                cfar_mask(r,d) = true;
            end
        end
    end

    % --- CFAR for DC-notch bins with tighter static SNR gate ---
    % Static clutter fills these bins; only accept very strong peaks
    for d = dcLo:dcHi
        col = rd_db(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB ...
               && (col(r)-noiseFloor_dB) > snr_static_min
                cfar_mask(r,d) = true;
            end
        end
    end

    % --- Extract detection parameters from CFAR hits ---
    [rBins, dBins] = find(cfar_mask);
    if ~isempty(rBins)
        ranges = (rBins-1) * rangeRes;                      % range (m)
        vels   = (dBins-zeroVelBin) * velRes;               % velocity (m/s, signed)
        snrs   = rd_db(sub2ind(size(rd_db),rBins,dBins)) - noiseFloor_dB; % SNR above noise floor
        peaks  = rd_db(sub2ind(size(rd_db),rBins,dBins));   % absolute peak power (dB)

        % --- Phase-based azimuth estimation (RX1 vs RX4) ---
        % Phase difference Δφ = angle(X_RX1 · X_RX4*) is proportional to
        % sin(θ): Δφ = 2π·d_ant·sin(θ) / λ
        % With d_ant = λ/2 → Δφ = π·sin(θ) → sin(θ) = Δφ/π
        rx1_val    = rd_complex_rx1(sub2ind(size(rd_complex_rx1),rBins,dBins));
        rx4_val    = rd_complex_rx4(sub2ind(size(rd_complex_rx4),rBins,dBins));
        phase_diff = angle(rx1_val .* conj(rx4_val));       % wrapped to [-π, π]
        sin_az     = max(-1, min(1, phase_diff / pi));       % clamp for numerical safety
        azimuth    = asin(sin_az);                           % azimuth angle (radians)

        % Convert polar (range, azimuth) to Cartesian (X = cross-range, Y = forward)
        x_pos      = ranges .* sin(azimuth);
        y_pos      = ranges .* cos(azimuth);

        % Store all detection attributes for this frame as one matrix row per hit
        frame_dets{f} = [repmat(f,numel(rBins),1), ranges, vels, ...
                         snrs, peaks, rad2deg(azimuth), x_pos, y_pos];

        % Store complex values for bearing estimation
        rx1_complex_store{f} = rx1_val;
        rx4_complex_store{f} = rx4_val;
    else
        % No detections this frame — store empty placeholders
        frame_dets{f}         = zeros(0,8);
        rx1_complex_store{f}  = [];
        rx4_complex_store{f}  = [];
    end

    % Progress update every 100 frames
    if mod(f,100)==0
        nSoFar = sum(cellfun(@(x) size(x,1), frame_dets(1:f)));
        fprintf('  Frame %d / %d | Detections so far: %d\n', ...
            f, Total_Audited, nSoFar);
    end
end
fclose(fid);

% Compute averaged RD map (used for Fig 7 background image)
rd_map_avg = rd_map_accum / frames_accumulated;

% Flatten all frame detection cells into one matrix for downstream processing
allDets    = cell2mat(frame_dets(1:Total_Audited));
det_frame  = allDets(:,1);   % frame index of each detection
det_range  = allDets(:,2);   % range (m)
det_vel    = allDets(:,3);   % velocity (m/s)
det_snr    = allDets(:,4);   % SNR above noise floor (dB)
det_rdPeak = allDets(:,5);   % absolute RD peak (dB)
det_az_deg = allDets(:,6);   % azimuth angle (deg)
det_x      = allDets(:,7);   % X spatial position (m)
det_y      = allDets(:,8);   % Y spatial position (m)
detCount   = size(allDets,1);

% Concatenate complex values in same row-order as allDets
rx1_complex_all = cell2mat(rx1_complex_store(1:Total_Audited));
rx4_complex_all = cell2mat(rx4_complex_store(1:Total_Audited));

fprintf('Total detections (pre-gate): %d\n\n', detCount);

%% ================================================================
%  STEP 3: PRIMARY CLASSIFICATION — Static / Moving / Noise
%
%  Each detection is assigned exactly one of three labels based on
%  range gate, SNR, and absolute velocity:
%
%    Noise       : outside range gate OR SNR below noise threshold
%    Static      : low velocity AND high SNR  (e.g. walls, furniture)
%    Moving      : velocity in pedestrian band AND sufficient SNR
%
%  Detections that pass the range/SNR gates but fall outside the
%  Static and Moving velocity windows are also labelled Noise here;
%  they may be re-labelled Interference in Step 4.
%% ================================================================
fprintf('--- STEP 3: Primary classification ---\n');
labels = strings(detCount, 1);   % string label per detection, initially empty

for k = 1:detCount
    rng_k = det_range(k);
    snr_k = det_snr(k);
    vel_k = abs(det_vel(k));

    % Hard range gate — discard anything outside the room boundaries
    if rng_k < minPhysicalRange || rng_k > maxPhysicalRange
        labels(k) = "Noise"; continue;
    end

    if snr_k < snr_noise_thresh
        % Below the global noise floor gate — cannot be a real target
        labels(k) = "Noise";
    elseif vel_k < vel_static_thresh && snr_k >= snr_static_min
        % Very low velocity + strong return → structural/static reflector
        labels(k) = "Static";
    elseif vel_k >= vel_moving_min && vel_k <= vel_moving_max ...
           && snr_k >= snr_moving_min
        % Velocity consistent with a pedestrian AND sufficient SNR → moving target
        labels(k) = "Moving";
    else
        % All other detections start as Noise (may be re-promoted in Step 4)
        labels(k) = "Noise";
    end
end

fprintf('After primary classification:\n');
fprintf('  Static : %d\n', sum(labels=="Static"));
fprintf('  Moving : %d\n', sum(labels=="Moving"));
fprintf('  Noise  : %d\n\n', sum(labels=="Noise"));

%% ================================================================
%  STEP 4: SECONDARY LOOP — promote Noise → Interference
%
%  PURPOSE:
%    Detections labelled Noise in Step 3 are reconsidered when there
%    is evidence of interference in the corresponding frame(s).
%    A Noise detection is promoted to "Interference" if ALL of:
%      - It falls in a frame flagged as an interference spike (or its
%        immediate temporal neighbours — 1-frame expand window)
%      - It meets the minimum interference SNR gate
%      - It shows at least one of:
%          * velocity outside the moving-target band (unusualVel)
%          * velocity in the ambiguous band between static and moving
%          * very low persistence (transient appearance → suddenAppearance)
%    OR simply if it appears in a spike frame AND exceeds the SNR gate.
%
%  Temporal expansion (±1 frame): interference often spills into the
%  frame immediately before/after the power spike due to the chirp
%  overlap at frame boundaries.
%% ================================================================
if Interfered_Count >= min_spike_count
    fprintf('--- STEP 4: Interference separation (%d spikes) ---\n', ...
        Interfered_Count);

    % Persistence: fraction of total frames in which each range bin is occupied
    % Low persistence → transient appearance → likely interference, not a target
    rangeBin    = round(det_range / rangeRes);  % map each detection to a range bin index
    uniqueBins  = unique(rangeBin);
    persistence = zeros(detCount, 1);
    for rb = uniqueBins'
        bm  = rangeBin == rb;
        persistence(bm) = numel(unique(det_frame(bm))) / Total_Audited;
    end

    % Expand spike flags ±1 frame to catch boundary spillover
    expandedFlags = interfered_flags;
    for ifr = find(interfered_flags)'
        if ifr > 1,            expandedFlags(ifr-1) = true; end
        if ifr < Total_Audited,expandedFlags(ifr+1) = true; end
    end

    % Only consider Noise detections inside the valid range gate
    noiseIdx = find(labels=="Noise" & ...
                    det_range >= minPhysicalRange & ...
                    det_range <= maxPhysicalRange);
    promoted = 0;
    for k = noiseIdx'
        fi    = det_frame(k);
        vel_k = abs(det_vel(k));
        snr_k = det_snr(k);
        per_k = persistence(k);

        % Skip if this frame has no interference flag (even after expansion)
        if ~expandedFlags(fi), continue; end

        % Kinematic / temporal signatures that suggest interference
        unusualVel       = vel_k > vel_moving_max;           % too fast for a pedestrian
        ambiguousVel     = vel_k >= vel_static_thresh && vel_k < vel_moving_min; % in dead-zone
        suddenAppearance = per_k <= persist_interf_max;       % appears in very few frames

        if (unusualVel || ambiguousVel || suddenAppearance) ...
           && snr_k >= snr_interference_min
            labels(k) = "Interference"; promoted = promoted+1;
        elseif snr_k >= snr_interference_min && expandedFlags(fi)
            % Strong SNR in a spike frame → also promote even without velocity evidence
            labels(k) = "Interference"; promoted = promoted+1;
        end
    end
    fprintf('  Noise promoted to Interference: %d\n\n', promoted);
else
    % No interference declared in Step 1 → skip promotion entirely
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
%  PURPOSE:
%    Group spatially/spectrally nearby detections into clusters so
%    that centroid positions and bearing estimates are computed on
%    confirmed groups rather than isolated detections.
%
%  Feature space: [range_bin, Doppler_bin] (normalised to bin units)
%  so that eps has the same meaning in both dimensions.
%
%  Cluster ID encoding (avoids collisions between classes):
%    Static       : cluster ID + 1000
%    Moving       : cluster ID + 2000
%    Interference : cluster ID + 3000
%
%  DBSCAN returns -1 for noise/outlier points; these are:
%    Static/Moving  → remain in their class but unclustered
%    Interference   → re-labelled "isolated interference" (unreliable for bearing)
%% ================================================================
fprintf('--- STEP 5: DBSCAN Clustering ---\n');

% Initialise cluster IDs to zero (0 = no cluster assigned)
cluster_ids = zeros(detCount, 1);

% --- Cluster Static detections ---
idx = find(labels == "Static");
if numel(idx) >= dbscan_minpts
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_static, dbscan_minpts);
    cluster_ids(idx) = clust + 1000;
    fprintf('  Static  : %d detections → %d clusters, %d unclustered\n', ...
        numel(idx), max(max(clust),0), sum(clust==-1));
end

% --- Cluster Moving detections ---
idx = find(labels == "Moving");
if numel(idx) >= dbscan_minpts
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_moving, dbscan_minpts);
    cluster_ids(idx) = clust + 2000;
    fprintf('  Moving  : %d detections → %d clusters, %d unclustered\n', ...
        numel(idx), max(max(clust),0), sum(clust==-1));
end

% --- Cluster Interference detections ---
idx = find(labels == "Interference");
if numel(idx) >= dbscan_minpts_interf
    feats = [det_range(idx)/rangeRes, det_vel(idx)/velRes];
    clust = dbscan(feats, dbscan_eps_interf, dbscan_minpts_interf);
    cluster_ids(idx) = clust + 3000;

    % Outliers from DBSCAN (clust==-1) → demoted to isolated interference
    unclustMask = clust == -1;
    labels(idx(unclustMask)) = "isolated interference";
    
    % Core cluster members remain assigned to clustered interference
    labels(idx(~unclustMask)) = "clustered interference";

    nClust  = max(clust);
    nWeak   = sum(unclustMask);
    nStrong = sum(~unclustMask);
    fprintf('  Interf  : %d detections → %d clusters (%d confirmed), %d weak\n', ...
        numel(idx), max(nClust,0), nStrong, nWeak);
elseif numel(idx) > 0
    % Too few detections for any cluster → all are isolated interference
    labels(idx) = "isolated interference";
    fprintf('  Interf  : %d detections — all weak (below DBSCAN minimum)\n', numel(idx));
end

fprintf('  Clustered Interference : %d\n', sum(labels=="clustered interference"));
fprintf('  Isolated Interference  : %d\n\n', sum(labels=="isolated interference"));

%% ================================================================
%  STEP 6: CLUSTER CENTROIDS
%
%  For each confirmed cluster (Static / Moving / Interference) compute
%  the mean range, velocity, SNR, X and Y position across all member
%  detections.  Results are stored in centroid_list for plotting.
%% ================================================================
fprintf('--- Cluster Centroids ---\n');
fprintf('%-22s %6s %10s %10s %8s %8s %8s %8s\n', ...
    'Class','ClustID','Range(m)','Vel(m/s)','SNR(dB)','X(m)','Y(m)','Count');
fprintf('%s\n', repmat('-',1,80));

% centroid_list columns: [classIndex, clusterID, range, vel, SNR, X, Y, count]
centroid_list = [];

% Colour and label maps indexed by classIndex (1=Static, 2=Moving, 3=Interference)
colMap = {col_static, col_moving, col_interf};
lblMap = {'Static centroid','Moving centroid','Clustered Interference centroid'};

% Class → offset mapping for extracting cluster IDs added in Step 5
clsCluster = {"Static",1000;"Moving",2000;"clustered interference",3000};

for c = 1:size(clsCluster,1)
    cl     = clsCluster{c,1};
    offset = clsCluster{c,2};
    idx    = find(labels == cl);
    if isempty(idx), continue; end

    clust  = cluster_ids(idx);
    uC     = unique(clust); uC = uC(uC > 0);  % valid cluster IDs only

    for uc = uC'
        cidx   = idx(clust == uc);          % indices of members of this cluster
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
%    The IWR6843ISK has a ULA with d = λ/2 spacing. For two antennas
%    separated by d, the inter-element phase difference is:
%      Δφ = (2π · d · sin θ) / λ = π · sin θ
%    Rearranging: θ = arcsin(Δφ / π)
%
%    Rather than averaging per-detection phase angles (which suffer
%    from wrap-around), we sum the complex phase products
%      Σ (X_RX1 · X_RX4*)
%    and extract the angle of the sum (coherent averaging).  This is
%    equivalent to maximum-likelihood estimation under additive noise
%    and correctly handles phase wrapping.
%
%  PER-CLUSTER OUTPUT:
%    bearing_deg  — estimated azimuth of interference source
%    bearing_std  — circular standard deviation (spread indicator)
%    reliability  — "High" / "Moderate" / "Low" based on std thresholds
%
%  Convention: 0 deg = directly ahead, negative = left, positive = right
%% ================================================================
nConfirmedInterf = sum(labels=="clustered interference");

if nConfirmedInterf > 0
    fprintf('\n--- STEP 6b: Interference Bearing Estimation ---\n');
    fprintf('Convention: 0 deg = directly ahead, negative = left, positive = right\n\n');
    interfIdx = find(labels=="clustered interference");
    interfClustIDs = cluster_ids(interfIdx);
    uniqueClustIDs = unique(interfClustIDs);
    uniqueClustIDs = uniqueClustIDs(uniqueClustIDs > 3000); % only interference clusters
    nClusters      = numel(uniqueClustIDs);

    % Pre-allocate per-cluster bearing results
    cluster_bearing_deg  = zeros(nClusters, 1);
    cluster_bearing_std  = zeros(nClusters, 1);
    cluster_snr_linear   = zeros(nClusters, 1);
    cluster_snr_db       = zeros(nClusters, 1);
    cluster_count        = zeros(nClusters, 1);
    cluster_range        = zeros(nClusters, 1);
    cluster_reliability  = strings(nClusters, 1);

    fprintf('%-16s %10s %12s %12s %10s %12s\n', ...
        'Cluster','Range(m)','Bearing(deg)','Std(deg)','SNR(dB)','Reliability');
    fprintf('%s\n', repmat('-',1,68));

    for ci = 1:nClusters
        cid  = uniqueClustIDs(ci);
        cidx = interfIdx(interfClustIDs == cid);  % indices of this cluster's detections

        % Optionally restrict to high-quality detections for the phase estimate
        % Fall back to all members if too few pass the SNR gate
        snr_mask      = det_snr(cidx) >= bearing_snr_min;
        cidx_filtered = cidx(snr_mask);
        if numel(cidx_filtered) < 2
            cidx_filtered = cidx;  % fall back to all if too few pass SNR gate
        end

        % --- Coherent phase averaging ---
        % Sum RX1 * conj(RX4) across detections; the angle of the sum gives
        % the coherent mean phase difference, weighted by signal amplitude.
        rx1_cluster  = rx1_complex_all(cidx_filtered);
        rx4_cluster  = rx4_complex_all(cidx_filtered);
        phase_products = rx1_cluster .* conj(rx4_cluster);
        coherent_sum   = sum(phase_products);

        % Convert coherent mean phase → bearing angle
        mean_phase_diff  = angle(coherent_sum);
        sin_bearing      = max(-1, min(1, mean_phase_diff / pi));
        bearing_rad      = asin(sin_bearing);
        bearing_deg_val  = rad2deg(bearing_rad);

        % --- Circular standard deviation across per-detection bearings ---
        % Measures intra-cluster angular spread; used for reliability rating.
        % Circular std formula: σ_c = sqrt(-2·ln(R)) where R = mean resultant length
        per_det_phase = angle(phase_products);
        per_det_sin   = max(-1, min(1, per_det_phase / pi));
        per_det_bear  = rad2deg(asin(per_det_sin));

        sin_mean = mean(sind(per_det_bear));
        cos_mean = mean(cosd(per_det_bear));
        R_circ   = sqrt(sin_mean^2 + cos_mean^2);            % mean resultant length [0,1]
        circ_std = rad2deg(sqrt(-2 * log(max(R_circ, 1e-6)))); % circular std (deg)

        % Assign reliability label based on circular std thresholds
        if circ_std < bearing_std_reliable
            reliability = "High";
        elseif circ_std < bearing_std_poor
            reliability = "Moderate";
        else
            reliability = "Low";
        end

        % Store results for scene-level aggregation
        cluster_bearing_deg(ci) = bearing_deg_val;
        cluster_bearing_std(ci) = circ_std;
        cluster_snr_db(ci)      = mean(det_snr(cidx));
        cluster_snr_linear(ci)  = 10^(cluster_snr_db(ci)/10);  % linear for weighted mean
        cluster_count(ci)       = numel(cidx);
        cluster_range(ci)       = mean(det_range(cidx));
        cluster_reliability(ci) = reliability;

        fprintf('%-16d %10.2f %12.1f %12.1f %10.2f %12s\n', ...
            ci, cluster_range(ci), bearing_deg_val, circ_std, ...
            cluster_snr_db(ci), reliability);
    end

    %% STEP 6c: SCENE-LEVEL BEARING — weighted circular mean
    %
    %  Combines per-cluster bearings into one final scene estimate.
    %  Two variants are produced:
    %    Unweighted : simple circular mean across clusters
    %    Weighted   : clusters weighted by their linear SNR
    %  The weighted estimate is preferred as it down-weights weaker/noisier clusters.
    fprintf('\n--- Scene-Level Bearing Estimate ---\n');

    sin_vals = sind(cluster_bearing_deg);
    cos_vals = cosd(cluster_bearing_deg);

    % Unweighted circular mean bearing (reference / cross-check)
    bearing_unweighted = atan2d(mean(sin_vals), mean(cos_vals));

    % SNR-weighted circular mean bearing (primary estimate)
    weights      = cluster_snr_linear / sum(cluster_snr_linear);
    sin_weighted = sum(weights .* sin_vals);
    cos_weighted = sum(weights .* cos_vals);
    bearing_weighted = atan2d(sin_weighted, cos_weighted);

    % Inter-cluster circular std — measures consistency across clusters
    R_scene        = sqrt(mean(sin_vals)^2 + mean(cos_vals)^2);
    scene_circ_std = rad2deg(sqrt(-2 * log(max(R_scene, 1e-6))));

    % Angular error relative to known lab geometry (for evaluation)
    % mod(...+180,360)-180 wraps to [-180,+180] to give the signed shortest arc
    error_unweighted = mod((bearing_unweighted - true_bearing_deg)+180,360)-180;
    error_weighted   = mod((bearing_weighted   - true_bearing_deg)+180,360)-180;

    fprintf('Number of confirmed clusters used: %d\n', nClusters);
    fprintf('True bearing (from lab geometry): %.1f deg\n', true_bearing_deg);
    fprintf('\nUnweighted circular mean bearing: %+.1f deg  (error: %+.1f deg)\n', ...
        bearing_unweighted, error_unweighted);
    fprintf('SNR-weighted circular mean bearing: %+.1f deg  (error: %+.1f deg)\n', ...
        bearing_weighted, error_weighted);
    fprintf('Inter-cluster circular std: %.1f deg\n', scene_circ_std);

    % Human-readable direction string
    if abs(bearing_weighted) < 15
        dir_str = 'AHEAD (main lobe)';
    elseif bearing_weighted < 0
        dir_str = sprintf('LEFT (%.1f deg)', abs(bearing_weighted));
    else
        dir_str = sprintf('RIGHT (%.1f deg)', bearing_weighted);
    end
    fprintf('Estimated direction: %s\n', dir_str);

    % Confidence string based on inter-cluster spread
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
    %  Computes an independent bearing estimate for each spike frame
    %  using only the Interference detections in that frame.
    %  Gives a time-series view of how stable the bearing is across frames.
    fprintf('--- STEP 6d: Frame-by-frame bearing distribution ---\n');

    spike_frames   = find(interfered_flags(1:Total_Audited));
    frame_bearings = nan(numel(spike_frames), 1);   % NaN = insufficient data
    frame_counts   = zeros(numel(spike_frames), 1); % number of detections used

    for fi = 1:numel(spike_frames)
        fnum              = spike_frames(fi);
        frame_interf_idx = find(labels=="clustered interference" & det_frame==fnum);
        if numel(frame_interf_idx) < 2, continue; end  % need at least 2 for phase estimate

        % Coherent phase sum for this frame's interference detections
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
        % Circular statistics across frame-level bearing estimates
        R_frame        = sqrt(mean(sind(valid_frame_bearings))^2 + ...
                              mean(cosd(valid_frame_bearings))^2);
        frame_circ_std = rad2deg(sqrt(-2*log(max(R_frame,1e-6))));
        frame_mean_bearing = atan2d(mean(sind(valid_frame_bearings)), ...
                                    mean(cosd(valid_frame_bearings)));
        fprintf('Frame-level mean bearing: %+.1f deg\n', frame_mean_bearing);
        fprintf('Frame-level circular std: %.1f deg\n\n', frame_circ_std);
    end

else
    % No confirmed interference clusters — bearing estimation not possible
    fprintf('\n--- STEP 6b: Skipped (no confirmed interference clusters) ---\n');
    fprintf('Bearing estimation requires DBSCAN-confirmed interference clusters.\n\n');

    % Set NaN sentinels so downstream plotting code can check safely
    bearing_weighted   = NaN;
    bearing_unweighted = NaN;
    scene_circ_std     = NaN;
    conf_str           = 'N/A';
    error_weighted     = NaN;
    valid_frame_bearings = [];
    nClusters          = 0;
end

%% ================================================================
%  STEP 7: STATISTICS SUMMARY
%
%  Prints a compact per-class breakdown of count, mean SNR, SNR std,
%  mean absolute velocity, and mean range, followed by top-level
%  interference detection metrics.
%% ================================================================
fprintf('\n========== FINAL PARAMETER PROFILE ==========\n');
fprintf('File: %s  [%s]\n', testFile, config_label);
fprintf('True bearing: %.1f deg | ', true_bearing_deg);
if ~isnan(bearing_weighted)
    fprintf('Estimated: %+.1f deg | Error: %+.1f deg\n', ...
        bearing_weighted, error_weighted);
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
%  Fig 3 — X/Y Cartesian spatial map with centroids
%  Fig 4 — 3D accumulated map (X, Y, SNR)
%  Fig 5 — Per-frame detection count time-series
%  Fig 6 — SNR histogram per class
%  Fig 7 — Averaged Range-Doppler map with detection overlays
%  Fig A — Polar bearing plot (only if interference confirmed)
%% ================================================================

% Logical mask: detections within the valid physical range gate
inRange = det_range >= minPhysicalRange & det_range <= maxPhysicalRange;

% Detection counts per class for plot titles and conditional rendering
nStatic = sum(labels=="Static");
nMoving = sum(labels=="Moving");
nInterf = sum(labels=="clustered interference");
nWeak   = sum(labels=="isolated interference");

% --- Fig 1: Frame power with spike detection ---
% Shows the mean RD power time-series and highlights detected spikes.
% A moving average trend line helps visualise slow drift vs. sharp spikes.
figure('Color','w','Name','Interference Spike Detection');
plot(frame_pwr,'LineWidth',1.2,'Color',[0.2 0.4 0.8],'DisplayName','Frame Power');
hold on;
moving_trend = movmean(frame_pwr, 10);  % 10-frame smoothed trend
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
% Classic 2D detection scatter useful for verifying threshold placement.
% Noise is subsampled to 1500 points to keep rendering fast.
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
% Overlay velocity threshold lines for visual verification
xline(vel_static_thresh,'g--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_min,   'b:','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_max,   'b-','LineWidth',1.5,'HandleVisibility','off');
yline(minPhysicalRange,'k--','LineWidth',1,'HandleVisibility','off');
yline(maxPhysicalRange,'k-', 'LineWidth',1.5,'HandleVisibility','off');
xlabel('|Velocity| (m/s)'); ylabel('Range (m)');
title(sprintf('Range vs |Velocity| — %s', config_label));
legend('Location','best'); grid on;

% --- Fig 3: X/Y Cartesian spatial map ---
% Top-down floor-plan view. X = cross-range (left/right), Y = forward range.
% Centroid stars mark the mean position of each confirmed cluster.
% Observer radar is shown at the origin.
figure('Color','w','Name','X/Y Spatial Detection Map');
hold on; axis equal;

idx = labels=="Noise" & inRange;
if sum(idx)>0
    pIdx=find(idx);
    if numel(pIdx)>1000, pIdx=pIdx(randperm(numel(pIdx),1000)); end
    scatter(det_x(pIdx),det_y(pIdx),6,col_noise,'o','filled',...
        'MarkerFaceAlpha',0.10,'DisplayName','Noise');
end
if nStatic>0
    scatter(det_x(labels=="Static"),det_y(labels=="Static"),...
        50,col_static,'s','filled','MarkerFaceAlpha',0.85,...
        'DisplayName',sprintf('Static (%d)',nStatic));
end
if nMoving>0
    scatter(det_x(labels=="Moving"),det_y(labels=="Moving"),...
        25,col_moving,'^','filled','MarkerFaceAlpha',0.65,...
        'DisplayName',sprintf('Moving (%d)',nMoving));
end
if nInterf>0
    scatter(det_x(labels=="clustered interference"),det_y(labels=="clustered interference"),...
        60,col_interf,'d','filled','MarkerFaceAlpha',0.95,...
        'DisplayName',sprintf('Clustered Interference (%d)',nInterf));
end
if nWeak>0
    scatter(det_x(labels=="isolated interference"),det_y(labels=="isolated interference"),...
        12,col_interf_weak,'o','filled','MarkerFaceAlpha',0.65,...
        'DisplayName',sprintf('Isolated Interference (%d)',nWeak));
end

% Overlay cluster centroids as pentagrams; show each class in legend once
if ~isempty(centroid_list)
    sDone=false; mDone=false; iDone=false;
    for i=1:size(centroid_list,1)
        cc=centroid_list(i,1); cX=centroid_list(i,6); cY=centroid_list(i,7);
        col=colMap{cc}; lbl=lblMap{cc};
        switch cc
            case 1, showLeg=~sDone; sDone=true; sz=180;
            case 2, showLeg=~mDone; mDone=true; sz=180;
            case 3, showLeg=~iDone; iDone=true; sz=150;
        end
        if showLeg
            scatter(cX,cY,sz,col,'p','filled','MarkerEdgeColor','k',...
                'LineWidth',1.5,'DisplayName',lbl);
        else
            scatter(cX,cY,sz,col,'p','filled','MarkerEdgeColor','k',...
                'LineWidth',1.5,'HandleVisibility','off');
        end
    end
end

% Observer radar at origin
scatter(0,0,150,'bs','filled','MarkerEdgeColor','k','LineWidth',1.5,...
    'DisplayName','Observer Radar');

xlim([-5 5]); ylim([0 6]);
xlabel('X (m) — Cross-range'); ylabel('Y (m) — Forward range');
title({sprintf('X/Y Spatial Map — %s — %s', config_label, ...
       strrep(testFile,'_Raw_0.bin','')); ...
       sprintf('%d frames | Static:%d | Moving:%d | Interf(clustered):%d | Interf(isolated):%d | Spikes:%d/%d', ...
       Total_Audited, nStatic, nMoving, nInterf, nWeak, ...
       Interfered_Count, Total_Audited)}, 'Interpreter','none');
legend('Location','northeast'); grid on;

% Annotation box summarising interference counts (only when interference found)
if Interfered_Count >= min_spike_count
    annotation('textbox',[0.12 0.01 0.65 0.05],...
        'String',sprintf('Interference spikes: %d/%d (%.1f%%)  |  Clustered: %d  |  Isolated: %d', ...
        Interfered_Count,Total_Audited,100*Interfered_Count/Total_Audited,...
        nInterf,nWeak),...
        'FitBoxToText','on','BackgroundColor',[1 0.93 0.93],...
        'EdgeColor',col_interf,'Color',col_interf,'FontSize',9);
end

% --- Fig 4: 3D accumulated map ---
% Adds SNR as the Z-axis so strong/weak detections are visually separated.
figure('Color','w','Name','3D Accumulated Detection Map');
hold on;
clsMap3D = {"Noise",col_noise,6,0.10,'o'; ...
            "Static",col_static,50,0.85,'s'; ...
            "Moving",col_moving,20,0.60,'^'; ...
            "Clustered Interference",col_interf,60,0.95,'d'; ...
            "Isolated Interference",col_interf_weak,12,0.65,'o'};
for c=1:size(clsMap3D,1)
    cl=clsMap3D{c,1}; col=clsMap3D{c,2};
    sz=clsMap3D{c,3}; alp=clsMap3D{c,4}; mkr=clsMap3D{c,5};
    idx=labels==cl & inRange;
    if sum(idx)==0, continue; end
    pIdx=find(idx);
    if strcmp(cl,'Noise') && numel(pIdx)>1000
        pIdx=pIdx(randperm(numel(pIdx),1000));
    end
    scatter3(det_x(pIdx),det_y(pIdx),det_snr(pIdx),sz,col,mkr,...
        'filled','MarkerFaceAlpha',alp,'DisplayName',...
        sprintf('%s (%d)',cl,sum(idx)));
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
% Useful for correlating interference spikes (vertical shaded lines) with
% changes in the number of classified detections per frame.
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
    % Shade spike frames with light red vertical lines
    for ifr = find(interfered_flags(1:Total_Audited))'
        xline(ifr,'Color',col_interf,'Alpha',0.20,'LineWidth',1,'HandleVisibility','off');
    end
end
xlabel('Frame'); ylabel('Count');
title(sprintf('Detections per Frame — %s', config_label));
legend('Location','best'); grid on;

% --- Fig 6: SNR distribution ---
% Histogram of SNR values per class; useful for checking threshold placement
% and understanding the margin between real targets and interference.
figure('Color','w','Name','SNR Distribution');
hold on;
clsSNR = {"Noise",col_noise;"Static",col_static;"Moving",col_moving; ...
          "clustered interference",col_interf;"isolated interference",col_interf_weak};
for c=1:size(clsSNR,1)
    cl=clsSNR{c,1}; col=clsSNR{c,2};
    idx=labels==cl & inRange;
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
%  Overlaid: CFAR detections colour-coded by class, cluster centroids,
%  DC notch boundaries, and velocity threshold lines.
%  The averaged map reduces frame-to-frame noise and makes persistent
%  clutter / interference patterns clearly visible.
%% ================================================================

% Velocity and range axis vectors for imagesc labelling
vel_axis   = ((0:numChirps-1) - zeroVelBin + 1) * velRes;
range_axis = (0:numADCSamples-1) * rangeRes;

figure('Color','k','Name','Range-Doppler Map');
ax = axes('Color','k');

% Display averaged RD map as background image
imagesc(ax, vel_axis, range_axis, rd_map_avg);
set(ax, 'YDir', 'normal');  % range increases upward
colormap(ax, jet);
cb = colorbar(ax);
cb.Color = 'w';
cb.Label.String = 'Power (dB)';
cb.Label.Color  = 'w';

% Clip colorscale to 2nd–99th percentile to suppress extreme outliers
cLims = prctile(rd_map_avg(:), [2 99]);
clim(ax, cLims);

hold(ax, 'on');

% DC notch boundary lines (white dashed)
dc_vel_lo = (dcLo - zeroVelBin) * velRes;
dc_vel_hi = (dcHi - zeroVelBin) * velRes;
xline(ax, dc_vel_lo, 'w--', 'LineWidth', 0.8, 'Alpha', 0.6, 'HandleVisibility','off');
xline(ax, dc_vel_hi, 'w--', 'LineWidth', 0.8, 'Alpha', 0.6, 'HandleVisibility','off');
yline(ax, minPhysicalRange, 'w:', 'LineWidth', 1.0, 'HandleVisibility','off');
yline(ax, maxPhysicalRange, 'w:', 'LineWidth', 1.0, 'HandleVisibility','off');

% Scatter overlays — same colour scheme as Figs 2–4
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

% Cluster centroid stars
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

% Velocity threshold overlay lines with labels
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
xline(ax, 0, 'w-', 'LineWidth', 0.6, 'Alpha', 0.4, 'HandleVisibility', 'off'); % zero-vel reference

xlabel(ax, 'Velocity (m/s)', 'Color', 'w');
ylabel(ax, 'Range (m)', 'Color', 'w');
set(ax, 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.5 0.5 0.5], ...
    'GridAlpha', 0.3, 'FontSize', 9);
xlim(ax, [-velMax velMax]);
ylim(ax, [0 rangeMax]);
grid(ax, 'on');

title(ax, {sprintf('Range-Doppler Map (averaged, %d frames) — %s', ...
           frames_accumulated, config_label); ...
           sprintf('Static:%d | Moving:%d | Interf(clustered):%d | Interf(isolated):%d', ...
           nStatic, nMoving, nInterf, nWeak)}, ...
    'Color', 'w', 'Interpreter', 'none');

lgd = legend(ax, 'Location', 'northeast', 'TextColor', 'w', ...
    'Color', [0.15 0.15 0.15], 'EdgeColor', [0.4 0.4 0.4]);

% Interference summary annotation
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
%  (only rendered when at least one confirmed interference cluster exists)
%% ================================================================
if nConfirmedInterf > 0

    % --- Fig A: Polar bearing plot ---
    % Shows per-cluster bearing estimates (colour-coded by reliability)
    % alongside the SNR-weighted and unweighted scene estimates,
    % and the true bearing from lab geometry for comparison.
    figure('Color','w','Name','Interference Bearing Estimation — Polar',...
        'Position',[100 100 700 700]);
    ax_polar = polaraxes;
    ax_polar.ThetaZeroLocation = 'top';    % 0 deg at top (forward direction)
    ax_polar.ThetaDir          = 'clockwise'; % positive angles to the right
    ax_polar.ThetaLim          = [-90 90]; % limit to visible forward hemisphere
    ax_polar.RLim              = [0 1.2];  % normalised radial axis
    ax_polar.RTick             = [];
    ax_polar.RTickLabel        = {};
    ax_polar.GridColor         = [0.8 0.8 0.8];
    hold(ax_polar, 'on');

    % Per-cluster markers: colour indicates reliability, size scales with SNR
    for ci = 1:nClusters
        bear_rad_ci = deg2rad(cluster_bearing_deg(ci));
        if cluster_reliability(ci) == "High"
            lc = [0.85 0.10 0.10];     % red — high reliability
        elseif cluster_reliability(ci) == "Moderate"
            lc = [0.95 0.55 0.10];     % orange — moderate reliability
        else
            lc = [0.80 0.80 0.80];     % grey — low reliability
        end
        polarplot(ax_polar, bear_rad_ci, 1.0, 'o', ...
            'MarkerFaceColor', lc, 'MarkerEdgeColor', 'k', ...
            'MarkerSize', 8+cluster_snr_db(ci)/5);
        polarplot(ax_polar, [0 bear_rad_ci], [0 0.9], '-', ...
            'Color', [lc 0.5], 'LineWidth', 1.2);
        text(ax_polar, bear_rad_ci, 1.08, sprintf('C%d', ci), ...
            'HorizontalAlignment','center','FontSize',8,'Color','k');
    end

    % SNR-weighted estimate (primary result) — thick red arrow
    bear_w_rad = deg2rad(bearing_weighted);
    polarplot(ax_polar, [0 bear_w_rad], [0 1.0], 'r-', 'LineWidth', 3);
    polarplot(ax_polar, bear_w_rad, 1.0, 'r^', ...
        'MarkerFaceColor','r','MarkerSize',12);

    % True bearing (ground truth from lab geometry) — black dashed line
    true_bear_rad = deg2rad(true_bearing_deg);
    polarplot(ax_polar, [0 true_bear_rad], [0 1.0], 'k--', 'LineWidth', 2);
    polarplot(ax_polar, true_bear_rad, 1.0, 'ks', ...
        'MarkerFaceColor','k','MarkerSize',10);

    % Unweighted estimate (reference) — blue dashed arrow, slightly shorter
    bear_uw_rad = deg2rad(bearing_unweighted);
    polarplot(ax_polar, [0 bear_uw_rad], [0 0.85], 'b--', 'LineWidth', 1.5);
    polarplot(ax_polar, bear_uw_rad, 0.85, 'b^', ...
        'MarkerFaceColor','b','MarkerSize',9);

    title(ax_polar, ...
        {sprintf('Interference Source Bearing Estimation — %s', config_label); ...
         sprintf('Estimated: %+.1f deg (weighted) | True: %+.1f deg | Error: %+.1f deg', ...
         bearing_weighted, true_bearing_deg, error_weighted)}, ...
        'FontSize', 10);

    % Manual legend box (polaraxes does not support standard legend)
    annotation('textbox',[0.72 0.25 0.22 0.20], ...
        'String', {'\color[rgb]{0.85,0.10,0.10}\bullet Per-cluster (high)', ...
                   '\color[rgb]{0.95,0.55,0.10}\bullet Per-cluster (mod)', ...
                   '\color[rgb]{0.80,0.80,0.80}\bullet Per-cluster (low)', ...
                   '{\color{red}\rightarrow} Weighted est.', ...
                   '{\color{blue}--} Unweighted est.', ...
                   '{\color{black}--} True bearing'}, ...
        'FitBoxToText','on','BackgroundColor','w','EdgeColor',[0.8 0.8 0.8],...
        'FontSize',8,'Interpreter','tex');

end  % end bearing plots block

fprintf('\n=== Complete [%s config] ===\n', config_label);
fprintf('Change testFile, numADCSamples, detection_sensitivity and true_bearing_deg at top for next dataset.\n');