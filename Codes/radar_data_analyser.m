%% ============================================================
%  Radar Detection Parameter Profiler
%  Purpose: Analyse raw ADC data to derive classification
%           thresholds for static, moving, and interference
%           detection classes
%  
%  Run this on each dataset before setting final code thresholds
%  Supports both 128-sample and 256-sample configurations
%
%  Output: SNR and velocity percentile tables per candidate class
%          Frame persistence analysis
%          Interference spike detection summary
%          Distribution histograms for threshold visualisation
%
%% ============================================================
clear; clc; close all;

%% ============================================================
%  CONFIGURATION — change these for each dataset
%% ============================================================

% *** CHANGE THIS FOR EACH DATASET ***
testFile      = '256adc_datacleanstatic.bin';
numADCSamples = 128;   % 128 or 256

% Detection sensitivity for interference spike detection
% A frame is flagged as a spike if its power exceeds both
% neighbours by more than this value in dB
detection_sensitivity = 1.5;   % dB

% Minimum spike count to declare interference present
% 128-sample: 1,  256-sample: 5
min_spike_count = 1;

%% ============================================================
%  HARDWARE PARAMETERS (IWR6843ISK — same for both configs)
%% ============================================================
numRX            = 4;
numChirps        = 128;
numFramesToAudit = 1000;

fc          = 60e9;
freqSlope   = 29.982e6 / 1e-6;   % Hz/s
fs          = 10e6;               % ADC sample rate
idleTime    = 100e-6;             % seconds
rampEndTime = 57.14e-6;           % seconds
chirpTime   = idleTime + rampEndTime;

BW          = freqSlope * (numADCSamples / fs);
c_light     = 3e8;
lambda      = c_light / fc;
rangeRes    = c_light / (2 * BW);
Tc          = numChirps * chirpTime;
velRes      = lambda / (2 * Tc);
rangeMax    = (fs * c_light) / (2 * freqSlope);
velMax      = lambda / (4 * chirpTime);

fprintf('=== Radar Parameters ===\n');
fprintf('File:                %s\n', testFile);
fprintf('ADC samples:         %d\n', numADCSamples);
fprintf('Effective BW:        %.2f MHz\n', BW/1e6);
fprintf('Range resolution:    %.4f m\n', rangeRes);
fprintf('Velocity resolution: %.4f m/s\n', velRes);
fprintf('Max range:           %.2f m\n', rangeMax);
fprintf('Max velocity:        %.2f m/s\n', velMax);
fprintf('========================\n\n');

%% ============================================================
%  CFAR PARAMETERS
%  Guard and training cells chosen for short-range lab coverage
%  first_valid_range = (guard + train + 1) * rangeRes
%  128-sample: (1+4+1)*0.3909 = 1.954 m
%  256-sample: (1+4+1)*0.1954 = 0.977 m
%% ============================================================
cfar_guard = 1;
cfar_train = 4;
cfar_pfa   = 1e-3;

first_valid_bin   = cfar_guard + cfar_train + 1;
last_valid_bin    = numADCSamples - cfar_guard - cfar_train;
first_valid_range = (first_valid_bin - 1) * rangeRes;
last_valid_range  = (last_valid_bin  - 1) * rangeRes;

fprintf('--- CFAR Coverage ---\n');
fprintf('Guard cells:            %d\n',   cfar_guard);
fprintf('Training cells/side:    %d\n',   cfar_train);
fprintf('First detectable range: %.3f m (bin %d)\n', first_valid_range, first_valid_bin);
fprintf('Last detectable range:  %.3f m (bin %d)\n', last_valid_range,  last_valid_bin);
fprintf('---------------------\n\n');

%% ============================================================
%  RANGE GATE
%  Min = CFAR geometry limit
%  Max = physical lab constraint (5x5 m room)
%% ============================================================
minPhysicalRange = first_valid_range;
maxPhysicalRange = 5.0;   % metres

%% ============================================================
%  INITIAL CANDIDATE THRESHOLDS
%  These are starting points — the profiler output will tell
%  where to set the final values
%% ============================================================
snr_noise_floor  = 8;     % dB — minimum SNR gate (both configs)
vel_static_cand  = 0.25;  % m/s — candidate static velocity upper bound
vel_moving_min   = 0.50;  % m/s — candidate moving lower bound
vel_moving_max   = 1.50;  % m/s — candidate moving upper bound (walking speed)

%% ============================================================
%  DC NOTCH
%  Zeroes ±dcNotch Doppler bins around zero-velocity bin
%  to suppress TX-RX leakage in non-DC CFAR path
%% ============================================================
dcNotch    = 2;

%% --- Windows ---
hS = 0.5*(1 - cos(2*pi*(0:numADCSamples-1)'/(numADCSamples-1)));
hC = 0.5*(1 - cos(2*pi*(0:numChirps-1)'/(numChirps-1)));

bytesPerFrame = numADCSamples * numRX * numChirps * 4;
zeroVelBin    = ceil(numChirps/2) + 1;

%% ============================================================
%  STEP 1: FRAME POWER CALCULATION + INTERFERENCE SPIKE DETECTION
%  Local neighbourhood comparison:
%  frame i is a spike if P(i) > P(i-1) + delta AND P(i) > P(i+1) + delta
%  This adapts to gradual power trends from moving targets
%% ============================================================
fprintf('--- STEP 1: Frame power + spike detection ---\n');
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

frame_pwr = zeros(numFramesToAudit, 1);
for f = 1:numFramesToAudit
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if isempty(raw), break; end
    LVDS     = raw(1:2:end) + 1i*raw(2:2:end);
    LVDS     = reshape(LVDS, numADCSamples*numRX, numChirps).';
    rx1_data = LVDS(:, 1:numADCSamples).';
    rd_db    = 20*log10(abs(fftshift( ...
                   fft(fft(rx1_data.*hS,numADCSamples,1).*hC',numChirps,2),2)));
    frame_pwr(f) = mean(rd_db(:));
end
fclose(fid);

Total_Audited = find(frame_pwr ~= 0, 1, 'last');
frame_pwr     = frame_pwr(1:Total_Audited);

% Local neighbourhood spike detection
raw_spike_flags = false(Total_Audited, 1);
for i = 2:(Total_Audited-1)
    if (frame_pwr(i) > frame_pwr(i-1) + detection_sensitivity) && ...
       (frame_pwr(i) > frame_pwr(i+1) + detection_sensitivity)
        raw_spike_flags(i) = true;
    end
end
raw_spike_count = sum(raw_spike_flags);

% Apply minimum spike count gate
if raw_spike_count >= min_spike_count
    interfered_flags = raw_spike_flags;
    Interfered_Count = raw_spike_count;
    fprintf('Interference DECLARED: %d spikes (>= threshold %d)\n', ...
        Interfered_Count, min_spike_count);
else
    interfered_flags = false(Total_Audited, 1);
    Interfered_Count = 0;
    fprintf('No interference declared: %d spikes < threshold %d\n', ...
        raw_spike_count, min_spike_count);
end
fprintf('Total frames audited: %d\n\n', Total_Audited);

%% ============================================================
%  STEP 2: PER-FRAME CFAR DETECTION
%  Two separate CFAR paths:
%  Path A — non-DC bins (notched): detects moving + interference
%  Path B — DC bins (unnotched):   detects static objects
%  SNR gate applied to both paths to suppress weak false alarms
%% ============================================================
fprintf('--- STEP 2: CFAR detection across all frames ---\n');
fid = fopen(testFile, 'r');
if fid == -1, error('File not found: %s', testFile); end

frame_dets = cell(Total_Audited, 1);

numTrain_val = 2 * cfar_train;
alpha_cfar   = numTrain_val * (cfar_pfa^(-1/numTrain_val) - 1);
alpha_dB     = 10*log10(alpha_cfar);

for f = 1:Total_Audited
    raw = fread(fid, bytesPerFrame/2, 'int16');
    if numel(raw) < bytesPerFrame/2
        fprintf('End of file at frame %d\n', f);
        Total_Audited = f-1; break;
    end

    LVDS_all = raw(1:2:end) + 1i*raw(2:2:end);
    LVDS_mat = reshape(LVDS_all, numADCSamples*numRX, numChirps).';
    rx1      = LVDS_mat(:, 1:numRX:numADCSamples*numRX).';

    rd_complex = fftshift( ...
        fft(fft(rx1.*hS,numADCSamples,1).*hC',numChirps,2), 2);
    rd_db = 20*log10(abs(rd_complex) + eps);

    noiseFloor_dB = mean(rd_db(:));

    dcLo = max(1, zeroVelBin - dcNotch);
    dcHi = min(numChirps, zeroVelBin + dcNotch);

    % Path A: notched map for non-DC bins
    rd_mag_notched = abs(rd_complex);
    rd_mag_notched(:, dcLo:dcHi) = 0;
    rd_db_notched  = 20*log10(rd_mag_notched + eps);

    cfar_mask = false(numADCSamples, numChirps);

    % Non-DC CFAR (no SNR gate — we want ALL detections for profiling)
    for d = 1:numChirps
        if d >= dcLo && d <= dcHi, continue; end
        col = rd_db_notched(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB
                cfar_mask(r,d) = true;
            end
        end
    end

    % DC CFAR (Path B — static detection)
    for d = dcLo:dcHi
        col = rd_db(:, d);
        for r = first_valid_bin : last_valid_bin
            rWin = [r-cfar_train-cfar_guard : r-cfar_guard-1, ...
                    r+cfar_guard+1          : r+cfar_guard+cfar_train];
            if col(r) > mean(col(rWin)) + alpha_dB
                cfar_mask(r,d) = true;
            end
        end
    end

    % Extract detection features
    [rBins, dBins] = find(cfar_mask);
    if ~isempty(rBins)
        ranges = (rBins-1) * rangeRes;
        vels   = (dBins-zeroVelBin) * velRes;
        snrs   = rd_db(sub2ind(size(rd_db),rBins,dBins)) - noiseFloor_dB;
        % columns: [frame, range, vel, snr]
        frame_dets{f} = [repmat(f,numel(rBins),1), ranges, vels, snrs];
    else
        frame_dets{f} = zeros(0,4);
    end

    if mod(f,100)==0
        nSoFar = sum(cellfun(@(x) size(x,1), frame_dets(1:f)));
        fprintf('  Frame %d / %d | Detections so far: %d\n', ...
            f, Total_Audited, nSoFar);
    end
end
fclose(fid);

allDets   = cell2mat(frame_dets(1:Total_Audited));
det_frame = allDets(:,1);
det_range = allDets(:,2);
det_vel   = allDets(:,3);
det_snr   = allDets(:,4);
detCount  = size(allDets,1);

% Range gate mask
inGate = det_range >= minPhysicalRange & det_range <= maxPhysicalRange;

fprintf('\nTotal detections (pre-gate): %d (%.1f avg/frame)\n', ...
    detCount, detCount/Total_Audited);
fprintf('Detections inside range gate: %d (%.1f avg/frame)\n\n', ...
    sum(inGate), sum(inGate)/Total_Audited);

%% ============================================================
%  STEP 3: FRAME PERSISTENCE SCORING
%  For each range bin, compute what fraction of total frames
%  it fires in. Used to separate:
%    Persistent noise  (high persistence — DC leakage, multipath)
%    Real static objects (moderate-high persistence)
%    Interference artefacts (low persistence — sudden appearance)
%% ============================================================
fprintf('--- STEP 3: Frame persistence analysis ---\n');

rangeBin    = round(det_range / rangeRes);
uniqueBins  = unique(rangeBin(inGate));
persistence = zeros(detCount, 1);

for rb = uniqueBins'
    binMask   = rangeBin == rb;
    binFrames = unique(det_frame(binMask));
    frac      = numel(binFrames) / Total_Audited;
    persistence(binMask) = frac;
end

% Persistence distribution inside gate
pers_inGate = persistence(inGate);
fprintf('Persistence distribution (in-gate detections):\n');
for p = [5 10 25 50 75 90 95]
    fprintf('  p%2d: %.3f (%.1f%% of frames)\n', p, ...
        prctile(pers_inGate,p), prctile(pers_inGate,p)*100);
end
fprintf('\n');

%% ============================================================
%  STEP 4: CANDIDATE CLASS SEPARATION
%  Assigned provisional labels based on velocity only
%  (no SNR threshold yet — this is what we are trying to find)
%
%  Static candidates:   |v| < vel_static_cand
%  Moving candidates:   vel_moving_min <= |v| <= vel_moving_max
%  High-vel candidates: |v| > vel_moving_max  (likely interference/noise)
%  Ambiguous zone:      vel_static_cand <= |v| < vel_moving_min
%% ============================================================
fprintf('--- STEP 4: Candidate class profiling ---\n');

staticCand  = inGate & abs(det_vel) <  vel_static_cand;
movingCand  = inGate & abs(det_vel) >= vel_moving_min ...
                     & abs(det_vel) <= vel_moving_max;
highVelCand = inGate & abs(det_vel) >  vel_moving_max;
ambigCand   = inGate & abs(det_vel) >= vel_static_cand ...
                     & abs(det_vel) <  vel_moving_min;
interferCand = inGate & interfered_flags(det_frame) == 1;

fprintf('Candidate counts inside gate:\n');
fprintf('  Static candidates   (|v| < %.2f m/s):       %d\n', vel_static_cand,  sum(staticCand));
fprintf('  Moving candidates   (%.2f-%.2f m/s):      %d\n', vel_moving_min, vel_moving_max, sum(movingCand));
fprintf('  Ambiguous zone      (%.2f-%.2f m/s):      %d\n', vel_static_cand, vel_moving_min, sum(ambigCand));
fprintf('  High velocity       (|v| > %.2f m/s):      %d\n', vel_moving_max,    sum(highVelCand));
fprintf('  In interference frames:                     %d\n\n', sum(interferCand));

%% ============================================================
%  STEP 5: SNR AND VELOCITY PERCENTILE TABLES
%  These are the key numbers for setting final thresholds
%  Look for:
%    - SNR gap between noise floor and static/moving peaks
%    - Velocity boundary between static and moving distributions
%    - SNR of interference-frame detections vs clean detections
%% ============================================================
fprintf('--- STEP 5: SNR and velocity percentile tables ---\n\n');

candidates = {staticCand, movingCand, ambigCand, highVelCand, interferCand};
candNames  = {'Static cand','Moving cand','Ambiguous','High-vel','In-interf-frame'};

%% SNR percentiles
fprintf('--- SNR percentiles by candidate class (dB above noise floor) ---\n');
fprintf('%-20s %6s %6s %6s %6s %6s %6s %6s %8s\n', ...
    'Class', 'Count', 'Mean', 'Std', 'p10', 'p25', 'p75', 'p90', 'p95');
fprintf('%s\n', repmat('-',1,80));
for c = 1:numel(candidates)
    mask = candidates{c};
    if sum(mask) < 5
        fprintf('%-20s %6d   (insufficient data)\n', candNames{c}, sum(mask));
        continue;
    end
    s = det_snr(mask);
    fprintf('%-20s %6d %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %8.2f\n', ...
        candNames{c}, sum(mask), mean(s), std(s), ...
        prctile(s,10), prctile(s,25), prctile(s,75), prctile(s,90), prctile(s,95));
end

%% Velocity percentiles
fprintf('\n--- |Velocity| percentiles by candidate class (m/s) ---\n');
fprintf('%-20s %6s %6s %6s %6s %6s %6s %6s %8s\n', ...
    'Class', 'Count', 'Mean', 'Std', 'p10', 'p25', 'p75', 'p90', 'p95');
fprintf('%s\n', repmat('-',1,80));
for c = 1:numel(candidates)
    mask = candidates{c};
    if sum(mask) < 5
        continue;
    end
    v = abs(det_vel(mask));
    fprintf('%-20s %6d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %8.3f\n', ...
        candNames{c}, sum(mask), mean(v), std(v), ...
        prctile(v,10), prctile(v,25), prctile(v,75), prctile(v,90), prctile(v,95));
end

%% Range percentiles
fprintf('\n--- Range percentiles by candidate class (m) ---\n');
fprintf('%-20s %6s %6s %6s %6s %6s\n', 'Class', 'Count', 'Mean', 'p10', 'p50', 'p90');
fprintf('%s\n', repmat('-',1,50));
for c = 1:numel(candidates)
    mask = candidates{c};
    if sum(mask) < 5, continue; end
    r = det_range(mask);
    fprintf('%-20s %6d %6.3f %6.3f %6.3f %6.3f\n', ...
        candNames{c}, sum(mask), mean(r), prctile(r,10), prctile(r,50), prctile(r,90));
end

%% Persistence percentiles
fprintf('\n--- Persistence percentiles by candidate class ---\n');
fprintf('%-20s %6s %6s %6s %6s %6s\n', 'Class', 'Count', 'Mean%', 'p10%', 'p50%', 'p90%');
fprintf('%s\n', repmat('-',1,50));
for c = 1:numel(candidates)
    mask = candidates{c};
    if sum(mask) < 5, continue; end
    p = persistence(mask)*100;
    fprintf('%-20s %6d %6.1f %6.1f %6.1f %6.1f\n', ...
        candNames{c}, sum(mask), mean(p), prctile(p,10), prctile(p,50), prctile(p,90));
end

%% ============================================================
%  STEP 6: THRESHOLD CANDIDATE SUMMARY
%  Based on the percentile tables above, this section prints
%  suggested threshold values with the reasoning
%% ============================================================
fprintf('\n--- STEP 6: Suggested threshold candidates ---\n');
fprintf('(Based on observed distributions — verify across all datasets)\n\n');

% All in-gate detections for noise floor estimate
allInGate = det_snr(inGate);

if sum(staticCand) > 5
    static_snr_p10 = prctile(det_snr(staticCand), 10);
    static_vel_p90 = prctile(abs(det_vel(staticCand)), 90);
    fprintf('SNR static min:    %.1f dB  (static p10 = %.1f dB)\n', ...
        static_snr_p10, static_snr_p10);
    fprintf('Vel static max:    %.3f m/s (static vel p90 = %.3f m/s)\n', ...
        static_vel_p90, static_vel_p90);
else
    fprintf('SNR static min:    insufficient static candidates\n');
end

if sum(movingCand) > 5
    moving_snr_p10 = prctile(det_snr(movingCand), 10);
    moving_vel_p10 = prctile(abs(det_vel(movingCand)), 10);
    moving_vel_p90 = prctile(abs(det_vel(movingCand)), 90);
    fprintf('SNR moving min:    %.1f dB  (moving p10 = %.1f dB)\n', ...
        moving_snr_p10, moving_snr_p10);
    fprintf('Vel moving range:  %.3f to %.3f m/s (moving p10 to p90)\n', ...
        moving_vel_p10, moving_vel_p90);
else
    fprintf('SNR moving min:    insufficient moving candidates\n');
end

noise_snr_p90 = prctile(allInGate, 90);
fprintf('SNR noise gate:    %.1f dB  (all in-gate p90 = %.1f dB)\n', ...
    snr_noise_floor, noise_snr_p90);
fprintf('Persist interf max: 0.05 (interference events = %.1f%% of frames)\n', ...
    100*Interfered_Count/Total_Audited);
fprintf('\n');

%% ============================================================
%  STEP 7: VISUALISATIONS
%% ============================================================

% --- Fig 1: Frame power with spike detection ---
figure('Color','w','Name','Frame Power — Spike Detection');
plot(frame_pwr,'LineWidth',1.2,'Color',[0.2 0.4 0.8],'DisplayName','Frame Power');
hold on;
plot(movmean(frame_pwr,10),'k--','LineWidth',1,'DisplayName','Moving avg (trend)');
if raw_spike_count > 0
    scatter(find(raw_spike_flags), frame_pwr(raw_spike_flags), 40, ...
        [0.85 0.10 0.10],'filled', ...
        'DisplayName',sprintf('Spikes (%d found)',raw_spike_count));
end
title(['Frame Power — ' strrep(testFile,'_Raw_0.bin','')],'Interpreter','none');
xlabel('Frame'); ylabel('Avg Power (dB)');
legend('Location','best'); grid on;

% --- Fig 2: SNR histogram — all in-gate detections ---
% Looked for natural gaps that separate noise from targets
figure('Color','w','Name','SNR Distribution — All In-Gate Detections');
histogram(det_snr(inGate), 80, ...
    'FaceColor',[0.3 0.5 0.8],'FaceAlpha',0.7,'EdgeColor','none');
hold on;
xline(snr_noise_floor,'k:','LineWidth',2,...
    'DisplayName',sprintf('Noise gate candidate (%d dB)',snr_noise_floor));
if sum(staticCand)>5
    xline(prctile(det_snr(staticCand),10),'g--','LineWidth',1.5,...
        'DisplayName',sprintf('Static p10 (%.1f dB)',prctile(det_snr(staticCand),10)));
end
if sum(movingCand)>5
    xline(prctile(det_snr(movingCand),10),'b--','LineWidth',1.5,...
        'DisplayName',sprintf('Moving p10 (%.1f dB)',prctile(det_snr(movingCand),10)));
end
xlabel('SNR (dB above noise floor)'); ylabel('Count');
title(sprintf('SNR Distribution — In-Gate Detections — %d-sample', numADCSamples));
legend('Location','northeast'); grid on;

% --- Fig 3: |Velocity| histogram — all in-gate detections ---
% Looked for natural gap between zero-velocity and walking-speed peaks
figure('Color','w','Name','|Velocity| Distribution — All In-Gate Detections');
histogram(abs(det_vel(inGate)), 80, ...
    'FaceColor',[0.3 0.5 0.8],'FaceAlpha',0.7,'EdgeColor','none');
hold on;
xline(vel_static_cand,'g--','LineWidth',1.5,...
    'DisplayName',sprintf('Static cand max (%.2f m/s)',vel_static_cand));
xline(vel_moving_min, 'b--','LineWidth',1.5,...
    'DisplayName',sprintf('Moving cand min (%.2f m/s)',vel_moving_min));
xline(vel_moving_max, 'r-', 'LineWidth',1.5,...
    'DisplayName',sprintf('Moving cand max (%.2f m/s)',vel_moving_max));
xlabel('|Velocity| (m/s)'); ylabel('Count');
title(sprintf('|Velocity| Distribution — In-Gate Detections — %d-sample', numADCSamples));
legend('Location','northeast'); grid on;

% --- Fig 4: SNR vs |Velocity| scatter — threshold visualisation ---
% Shows the feature space used for classification
figure('Color','w','Name','SNR vs |Velocity| — Feature Space');
hold on;
if sum(staticCand)>0
    pIdx = find(staticCand);
    if numel(pIdx)>3000, pIdx=pIdx(randperm(numel(pIdx),3000)); end
    scatter(abs(det_vel(pIdx)), det_snr(pIdx), 10, ...
        [0.13 0.55 0.13],'o','filled','MarkerFaceAlpha',0.4,...
        'DisplayName',sprintf('Static cand (%d)',sum(staticCand)));
end
if sum(movingCand)>0
    pIdx = find(movingCand);
    if numel(pIdx)>3000, pIdx=pIdx(randperm(numel(pIdx),3000)); end
    scatter(abs(det_vel(pIdx)), det_snr(pIdx), 10, ...
        [0.12 0.35 0.75],'^','filled','MarkerFaceAlpha',0.4,...
        'DisplayName',sprintf('Moving cand (%d)',sum(movingCand)));
end
if sum(highVelCand)>0
    pIdx = find(highVelCand);
    if numel(pIdx)>3000, pIdx=pIdx(randperm(numel(pIdx),3000)); end
    scatter(abs(det_vel(pIdx)), det_snr(pIdx), 10, ...
        [0.85 0.10 0.10],'d','filled','MarkerFaceAlpha',0.3,...
        'DisplayName',sprintf('High-vel (%d)',sum(highVelCand)));
end
xline(vel_static_cand,'g--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_min, 'b--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_max, 'r-', 'LineWidth',1.5,'HandleVisibility','off');
yline(snr_noise_floor,'k:','LineWidth',1.5,'HandleVisibility','off');
xlabel('|Velocity| (m/s)'); ylabel('SNR (dB above noise floor)');
title(sprintf('SNR vs |Velocity| Feature Space — %d-sample', numADCSamples));
legend('Location','best'); grid on;

% --- Fig 5: Range vs |Velocity| ---
figure('Color','w','Name','Range vs |Velocity|');
hold on;
if sum(staticCand)>0
    pIdx=find(staticCand);
    if numel(pIdx)>3000, pIdx=pIdx(randperm(numel(pIdx),3000)); end
    scatter(abs(det_vel(pIdx)), det_range(pIdx), 10,...
        [0.13 0.55 0.13],'o','filled','MarkerFaceAlpha',0.4,...
        'DisplayName','Static cand');
end
if sum(movingCand)>0
    pIdx=find(movingCand);
    if numel(pIdx)>3000, pIdx=pIdx(randperm(numel(pIdx),3000)); end
    scatter(abs(det_vel(pIdx)), det_range(pIdx), 10,...
        [0.12 0.35 0.75],'^','filled','MarkerFaceAlpha',0.4,...
        'DisplayName','Moving cand');
end
xline(vel_static_cand,'g--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_min, 'b--','LineWidth',1,'HandleVisibility','off');
xline(vel_moving_max, 'r-', 'LineWidth',1.5,'HandleVisibility','off');
yline(minPhysicalRange,'k--','LineWidth',1,'HandleVisibility','off');
yline(maxPhysicalRange,'k-', 'LineWidth',1.5,'HandleVisibility','off');
xlabel('|Velocity| (m/s)'); ylabel('Range (m)');
title(sprintf('Range vs |Velocity| — %d-sample', numADCSamples));
legend('Location','best'); grid on;

% --- Fig 6: Persistence distribution ---
figure('Color','w','Name','Frame Persistence Distribution');
histogram(pers_inGate*100, 50, ...
    'FaceColor',[0.3 0.5 0.3],'FaceAlpha',0.7,'EdgeColor','none');
hold on;
xline(5, 'r--','LineWidth',1.5,...
    'DisplayName','Interference threshold (5%)');
xline(30,'b--','LineWidth',1.5,...
    'DisplayName','Persistent noise threshold (30%)');
xlabel('Persistence (% of frames)'); ylabel('Count');
title('Frame Persistence of In-Gate Detections');
legend('Location','best'); grid on;

%% ============================================================
%  FINAL SUMMARY
%% ============================================================
fprintf('========== PROFILER SUMMARY ==========\n');
fprintf('File:          %s\n', testFile);
fprintf('Config:        %d-sample\n', numADCSamples);
fprintf('Range gate:    %.3f to %.1f m\n', minPhysicalRange, maxPhysicalRange);
fprintf('Frames:        %d\n', Total_Audited);
fprintf('Spikes found:  %d  |  Declared: %s\n', ...
    raw_spike_count, string(Interfered_Count >= min_spike_count));
fprintf('In-gate dets:  %d (%.1f/frame)\n', sum(inGate), sum(inGate)/Total_Audited);
fprintf('======================================\n\n');
fprintf('KEY TABLES TO READ:\n');
fprintf('  Step 5 SNR table:  find natural gap between noise and target classes\n');
fprintf('  Step 5 Vel table:  find boundary between static and moving distributions\n');
fprintf('  Step 6 summary:    suggested threshold candidates\n');
fprintf('  Fig 4 (feature space): visualise separability in SNR-velocity space\n\n');
fprintf('Change testFile and numADCSamples at top — run on all datasets.\n');
fprintf('Compare Step 5 tables across datasets to confirm stable thresholds.\n');