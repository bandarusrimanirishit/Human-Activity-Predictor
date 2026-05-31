%% 1. Master Configuration
%==========================================================================
clear;
clc;
close all;

% --- CHOOSE THE SCRIPT'S FUNCTION ---
% 'add_files':               Import and label your raw data files.
% 'plot_individual_files':   Create a separate plot for each data file.
% 'train':                   Train the advanced BiLSTM model with feature engineering.
% 'test_model':              Load the trained model and test it on a new data file.
% 'run_rule_based_analysis': Analyze a file with your refined rule-based logic.
operatingMode = 'test_model'; % <-- Set to 'train' to build the new AI model


% --- Global Parameters ---
targetSampleRate = 100; % Hz (Standard rate for the entire project)
firOrder = 50;
cutoffFreq = 15;      % Hz

% --- Rule-Based Parameters (from your tuning) ---
idle_VM_variance_threshold = 5.0; 
walking_Z_variance_threshold = 11.5; 
stairs_Y_slope_threshold = 3.0; 

% Step counting parameters
minPeakHeight = 11.0; % A global minimum to filter out noise before dynamic thresholding
minPeakDistance = 0.35 * targetSampleRate;
emaSpan = 10;

% AI model parameters
windowSizeSeconds = 2;
windowOverlap = 0.5;


%% 2. Main Controller (Switches between Modes)
%==========================================================================
switch operatingMode
    case 'add_files'
        runFileImportPipeline();
    case 'plot_individual_files'
        runPlottingPipeline();
    case 'train'
        runTrainingPipeline(targetSampleRate, firOrder, cutoffFreq, windowSizeSeconds, windowOverlap);
    case 'test_model'
        runModelTestingPipeline(targetSampleRate, firOrder, cutoffFreq, windowSizeSeconds, windowOverlap, emaSpan, minPeakHeight, minPeakDistance);
    case 'run_rule_based_analysis'
        runRuleBasedAnalysis(targetSampleRate, firOrder, cutoffFreq, emaSpan, ...
            idle_VM_variance_threshold, stairs_Y_slope_threshold, walking_Z_variance_threshold, ...
            minPeakHeight, minPeakDistance);
        
    % --- Threshold Tuning is preserved but commented out as requested ---
    % case 'tune_thresholds'
    %     runThresholdTuner(targetSampleRate, firOrder, cutoffFreq);
        
    otherwise
        error("Invalid 'operatingMode'.");
end


%% 3. Core Functions
%==========================================================================

% --- RULE-BASED ANALYSIS FUNCTION (with IMPROVED logic) ---
function runRuleBasedAnalysis(targetSampleRate, firOrder, cutoffFreq, emaSpan, idle_threshold, stairs_slope_threshold, walking_z_thresh, minPeakHeight, minPeakDistance)
    fprintf('=== RUNNING IN REFINED RULE-BASED ANALYSIS MODE ===\n\n');
    
    [fileName, pathName] = uigetfile({'*.xls;*.csv'}, 'Select a File to Analyze');
    if isequal(fileName, 0), disp('User cancelled.'); return; end
    fullFilePath = fullfile(pathName, fileName);
    
    fprintf('Processing file: %s\n', fileName);
    T = readtable(fullFilePath);
    accelData = T{:, 2:4};

    % --- Pre-processing Pipeline ---
    try
        timeVector = T.Time;
        originalSampleRate = 1 / median(diff(timeVector));
        if abs(originalSampleRate - targetSampleRate) > 1
            fprintf('  Resampling from %.1f Hz to %.1f Hz...\n', originalSampleRate, targetSampleRate);
            accelData = resample(accelData, targetSampleRate, round(originalSampleRate));
        end
    catch
        originalSampleRate = targetSampleRate; % Default if time column is missing
        warning('Could not detect sample rate. Assuming %.1f Hz.', targetSampleRate);
    end

    lpFilt = designfilt('lowpassfir', 'FilterOrder', firOrder, 'CutoffFrequency', cutoffFreq, 'SampleRate', targetSampleRate);
    filteredAccel = filter(lpFilt, accelData);
    vectorMag = sqrt(sum(filteredAccel.^2, 2));
    
    alpha = 2 / (emaSpan + 1);
    ema_b = alpha;
    ema_a = [1, alpha - 1];
    smoothedVectorMag = filter(ema_b, ema_a, vectorMag);
    
    % --- Analysis with a larger 3.5-second window ---
    fprintf('\n--- CHRONOLOGICAL RULE-BASED RESULTS ---\n');
    windowLength = 3.5 * targetSampleRate; % Larger window for activity classification
    stepLength = 1 * targetSampleRate;   % Check every 1 second
    
    for i = 1:stepLength:(size(vectorMag, 1) - windowLength)
        windowEnd = i + windowLength - 1;
        
        xSegment = filteredAccel(i:windowEnd, 1);
        ySegment = filteredAccel(i:windowEnd, 2);
        zSegment = filteredAccel(i:windowEnd, 3);
        vmSegment = vectorMag(i:windowEnd);
        
        % --- REFINED Hierarchical Classification Rules ---
        finalActivity = 'Unknown';
        
        meanAbsX = mean(abs(xSegment));
        meanAbsY = mean(abs(ySegment));
        meanAbsZ = mean(abs(zSegment));
        [~, dominantAxisIdx] = max([meanAbsX, meanAbsY, meanAbsZ]);
        
        variance_VM = var(vmSegment);
        
        if dominantAxisIdx == 3 || variance_VM < idle_threshold
            finalActivity = 'Idle';
        else
            mean_first_half = mean(ySegment(1:floor(end/2)));
            mean_second_half = mean(ySegment(floor(end/2)+1:end));
            y_slope = mean_second_half - mean_first_half;
            
            variance_Z = var(zSegment);

            if y_slope > stairs_slope_threshold
                finalActivity = 'Climbing Upstairs';
            elseif y_slope < -stairs_slope_threshold
                finalActivity = 'Climbing Downstairs';
            elseif variance_Z > walking_z_thresh
                finalActivity = 'Walking';
            else
                finalActivity = 'Walking'; % Fallback is now Walking
            end
        end
        
        startTime = (i-1)/targetSampleRate;
        fprintf('  - Time %.1fs: Activity: %s\n', startTime, finalActivity);
    end
    
    % --- Overall Step Count (with robust percentile-based dynamic threshold) ---
    [pks_total, ~] = findpeaks(smoothedVectorMag, 'MinPeakHeight', minPeakHeight, 'MinPeakDistance', minPeakDistance);
    if length(pks_total) > 1
        dynamicThreshold_total = prctile(pks_total, 75);
        [~, step_locs_final] = findpeaks(smoothedVectorMag, 'MinPeakHeight', dynamicThreshold_total, 'MinPeakDistance', minPeakDistance);
        totalSteps = length(step_locs_final);
    else
        totalSteps = length(pks_total);
    end
    
    adjustmentFactor = originalSampleRate / targetSampleRate;
    if adjustmentFactor > 1.1 
        originalCount = totalSteps;
        totalSteps = round(totalSteps / adjustmentFactor);
        fprintf('\n  NOTE: Applied dynamic step count adjustment (factor of %.2f).\n', adjustmentFactor);
        fprintf('  Original count: %d, Adjusted count: %d\n', originalCount, totalSteps);
    end
    
    fprintf('\n------------------------------------------\n');
    fprintf('Total Steps Detected in File: %d\n', totalSteps);
    fprintf('------------------------------------------\n');
end

% --- FILE IMPORT PIPELINE FUNCTION ---
function runFileImportPipeline()
    fprintf('=== RUNNING IN FILE IMPORT MODE ===\n\n');
    activityOptions = {'idle', 'walking', 'climbing_upstairs', 'climbing_downstairs'};
    while true
        [fileNames, pathName] = uigetfile({'*.xls;*.csv'}, 'Select Raw Data File(s) to Import', 'MultiSelect', 'on');
        if isequal(fileNames, 0), disp('File import cancelled by user.'); break; end
        if ~iscell(fileNames), fileNames = {fileNames}; end
        for i = 1:length(fileNames)
            currentFile = fileNames{i};
            fullSourcePath = fullfile(pathName, currentFile);
            fprintf('Processing file: %s\n', currentFile);
            phoneModel = questdlg('Which phone is this data from?', 'Phone Model', 'iphone', 'nothingphone', 'iphone');
            if isempty(phoneModel), continue; end
            choice = menu('Which activity does this file contain?', activityOptions);
            if choice == 0, continue; end
            activity = activityOptions{choice};
            dataType = questdlg('What type of acceleration data is this?', 'Data Type', 'with_g', 'without_g', 'with_g');
            if isempty(dataType), continue; end
            [~, ~, fileExt] = fileparts(currentFile);
            newFileName = sprintf('Acceleration_%s_%s_%s.xls', dataType, activity, phoneModel);
            destinationPath = fullfile(pwd, newFileName);
            if strcmpi(fileExt, '.csv')
                T = readtable(fullSourcePath);
                writetable(T, destinationPath);
            else
                copyfile(fullSourcePath, destinationPath);
            end
            fprintf('  -> File imported and standardized to: %s\n\n', newFileName);
        end
        addAnother = questdlg('Do you want to add another file?', 'Continue?', 'Yes', 'No', 'Yes');
        if strcmp(addAnother, 'No'), break; end
    end
    fprintf('=== FILE IMPORT COMPLETE ===\n');
end

% --- INDIVIDUAL FILE PLOTTING FUNCTION ---
function runPlottingPipeline()
    fprintf('=== RUNNING IN PLOTTING MODE (INDIVIDUAL PLOTS) ===\n\n');
    activities = {'idle', 'walking', 'climbing_upstairs', 'climbing_downstairs'};
    phoneModels = {'iphone', 'nothingphone'};
    dataTypes = {'with_g', 'without_g'};
    for p = 1:length(phoneModels)
        phoneModel = phoneModels{p};
        for i = 1:length(activities)
            activity = activities{i};
            for d = 1:length(dataTypes)
                dataType = dataTypes{d};
                fileName = sprintf('Acceleration_%s_%s_%s.xls', dataType, activity, phoneModel);
                try
                    T = readtable(fileName);
                    time = T{:,1};
                    accel = T{:, 2:4};
                    figure('Name', fileName, 'NumberTitle', 'off');
                    subplot(3, 1, 1); plot(time, accel(:,1)); title('X-Axis'); ylabel('m/s^2'); grid on;
                    subplot(3, 1, 2); plot(time, accel(:,2)); title('Y-Axis'); ylabel('m/s^2'); grid on;
                    subplot(3, 1, 3); plot(time, accel(:,3)); title('Z-Axis'); ylabel('m/s^2'); xlabel('Time (s)'); grid on;
                    sgtitle(fileName, 'Interpreter', 'none');
                    fprintf('  Plotted: %s\n', fileName);
                catch
                    fprintf('  Could not find or plot: %s\n', fileName);
                end
            end
        end
    end
    fprintf('\n=== PLOTTING COMPLETE ===\n');
end

% --- TRAINING PIPELINE FUNCTION (with Feature Engineering) ---
function runTrainingPipeline(targetSampleRate, firOrder, cutoffFreq, windowSizeSeconds, windowOverlap)
    fprintf('=== RUNNING IN TRAINING MODE (with Feature Engineering) ===\n\n');
    
    activities = {'idle', 'walking', 'climbing_upstairs', 'climbing_downstairs'};
    phoneModels = {'iphone', 'nothingphone'};
    
    fprintf('Step 1: Loading and pre-processing all training data...\n');
    lpFilt = designfilt('lowpassfir', 'FilterOrder', firOrder, 'CutoffFrequency', cutoffFreq, 'SampleRate', targetSampleRate);
    allDataWithG = {};
    allLabels = {};
    
    for p = 1:length(phoneModels)
        phoneModel = phoneModels{p};
        originalSampleRate = 100;
        if strcmp(phoneModel, 'nothingphone'), originalSampleRate = 416.8; end
        
        fprintf('\nProcessing device: %s (Original Rate: %.1f Hz)\n', phoneModel, originalSampleRate);
        
        for i = 1:length(activities)
            activity = activities{i};
            fileName = sprintf('Acceleration_with_g_%s_%s.xls', activity, phoneModel);
            try
                T = readtable(fileName);
                accelData = T{:, 2:4};
                
                if originalSampleRate ~= targetSampleRate
                    fprintf('  Resampling %s data to %.1f Hz...\n', activity, targetSampleRate);
                    accelData = resample(accelData, targetSampleRate, round(originalSampleRate));
                end
                
                filteredAccel = filter(lpFilt, accelData);
                allDataWithG{end+1} = filteredAccel;
                allLabels{end+1} = activity;
                fprintf('  Loaded and processed: %s\n', activity);
            catch ME
                fprintf('  Error loading file for activity "%s": %s\n', activity, ME.message);
            end
        end
    end
    
    fprintf('\nStep 2: Creating windows and engineering features...\n');
    X_rnn = {};
    Y_rnn = {};
    windowLength = floor(windowSizeSeconds * targetSampleRate);
    stepLength = floor(windowLength * (1 - windowOverlap));
    for i = 1:length(allDataWithG)
        sequence = allDataWithG{i};
        label = allLabels{i};
        for j = 1:stepLength:(size(sequence, 1) - windowLength)
            window_raw = sequence(j : j + windowLength - 1, :);
            
            % --- Feature Engineering for the AI ---
            vectorMag = sqrt(sum(window_raw.^2, 2));
            yAxis = window_raw(:, 2);
            zAxis = window_raw(:, 3);
            
            feature_variance_VM = var(vectorMag);
            feature_variance_Z = var(zAxis); % For walking detection
            
            mean_first_half = mean(yAxis(1:floor(end/2)));
            mean_second_half = mean(yAxis(floor(end/2)+1:end));
            feature_y_slope = mean_second_half - mean_first_half;
            
            [~, dominantAxisIdx] = max(mean(abs(window_raw)));
            feature_dominant_axis = dominantAxisIdx;
            
            num_timesteps = size(window_raw, 1);
            feature_matrix = repmat([feature_variance_VM, feature_variance_Z, feature_y_slope, feature_dominant_axis], num_timesteps, 1);
            
            combined_window = [window_raw, feature_matrix];
            
            X_rnn{end+1} = combined_window';
            Y_rnn{end+1} = label;
        end
    end
    Y_rnn = categorical(Y_rnn');
    fprintf('Created %d feature-rich windows before balancing.\n', length(X_rnn));
    if isempty(Y_rnn), error('No data windows created.'); end
    
    fprintf('Step 3: Balancing dataset via oversampling...\n');
    [counts, classes] = groupcounts(Y_rnn);
    maxCount = max(counts);
    X_balanced = X_rnn;
    Y_balanced = Y_rnn;
    fprintf('  Majority class has %d samples. Balancing other classes...\n', maxCount);
    for i = 1:length(classes)
        currentClass = classes(i);
        currentCount = counts(i);
        if currentCount < maxCount
            numToGenerate = maxCount - currentCount;
            minorityIdx = find(Y_rnn == currentClass);
            randIdx = randsample(minorityIdx, numToGenerate, true);
            X_balanced = [X_balanced, X_rnn(randIdx)];
            Y_balanced = [Y_balanced; Y_rnn(randIdx)];
            fprintf('  Oversampled "%s" class with %d new samples.\n', char(currentClass), numToGenerate);
        end
    end
    fprintf('  Created %d windows after balancing.\n\n', length(X_balanced));
    
    fprintf('Step 4: Defining and training the BiLSTM network...\n');
    cv = cvpartition(length(Y_balanced), 'HoldOut', 0.2);
    X_train = X_balanced(training(cv)); Y_train = Y_balanced(training(cv));
    X_validation = X_balanced(test(cv)); Y_validation = Y_balanced(test(cv));
    numFeatures = size(X_train{1}, 1);
    numClasses = length(unique(Y_train));
    layers = [
        sequenceInputLayer(numFeatures)
        bilstmLayer(100, 'OutputMode', 'last')
        dropoutLayer(0.5)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];
    options = trainingOptions('adam', 'MaxEpochs', 30, 'MiniBatchSize', 32, 'ValidationData', {X_validation, Y_validation}, 'InitialLearnRate', 0.001, 'SequenceLength', 'longest', 'Shuffle', 'every-epoch', 'Plots', 'training-progress', 'Verbose', false);
    rnnNet = trainNetwork(X_train, Y_train, layers, options);
    
    fprintf('Step 5: Evaluating model and saving...\n');
    Y_pred = classify(rnnNet, X_validation);
    figure;
    confusionchart(Y_validation, Y_pred, 'Title', 'Validation Confusion Matrix');
    
    minPeakHeight = evalin('base', 'minPeakHeight');
    minPeakDistance = evalin('base', 'minPeakDistance');
    emaSpan = evalin('base', 'emaSpan');
    
    save('rnnNet.mat', 'rnnNet');
    save('model_parameters.mat', 'targetSampleRate', 'firOrder', 'cutoffFreq', 'minPeakHeight', 'minPeakDistance', 'windowSizeSeconds', 'windowOverlap', 'emaSpan', 'lpFilt');
    fprintf('Model and parameters saved successfully.\n');
end

% --- MODEL TESTING PIPELINE FUNCTION ---
function runModelTestingPipeline(targetSampleRate, firOrder, cutoffFreq, windowSizeSeconds, windowOverlap, emaSpan, minPeakHeight, minPeakDistance)
    fprintf('=== RUNNING IN MODEL TESTING MODE ===\n\n');
    
    fprintf('Step 1: Loading trained model and parameters...\n');
    try
        load('rnnNet.mat', 'rnnNet'); 
        load('model_parameters.mat');
        fprintf('Assets loaded successfully.\n\n');
    catch ME
        error('Could not find "rnnNet.mat" or "model_parameters.mat". Please run in ''train'' mode first.');
    end
    
    [fileName, pathName] = uigetfile({'*.xls;*.csv'}, 'Select a File for Testing');
    if isequal(fileName, 0), disp('User cancelled.'); return; end
    fullFilePath = fullfile(pathName, fileName);
    
    fprintf('Step 2: Processing file: %s\n', fileName);
    T = readtable(fullFilePath);
    try
        timeVector = T.Time;
        originalSampleRate = 1 / median(diff(timeVector));
    catch
        originalSampleRate = targetSampleRate;
    end
    
    accelData = T{:, 2:4};
    if abs(originalSampleRate - targetSampleRate) > 1
        accelData = resample(accelData, targetSampleRate, round(originalSampleRate));
    end
    
    filteredAccel = filter(lpFilt, accelData);
    
    fprintf('Step 3: Running inference...\n');
    windowLength = floor(windowSizeSeconds * targetSampleRate);
    stepLength = floor(windowLength * (1 - windowOverlap));
    X_val = {};
    windowStartSamples = [];
    for j = 1:stepLength:(size(filteredAccel, 1) - windowLength)
        window_raw = filteredAccel(j : j + windowLength - 1, :);
        
        % --- Feature Engineering for the AI ---
        vectorMag = sqrt(sum(window_raw.^2, 2));
        yAxis = window_raw(:, 2);
        zAxis = window_raw(:, 3);
        
        feature_variance_VM = var(vectorMag);
        feature_variance_Z = var(zAxis);
        
        mean_first_half = mean(yAxis(1:floor(end/2)));
        mean_second_half = mean(yAxis(floor(end/2)+1:end));
        feature_y_slope = mean_second_half - mean_first_half;
        
        [~, dominantAxisIdx] = max(mean(abs(window_raw)));
        feature_dominant_axis = dominantAxisIdx;
        
        num_timesteps = size(window_raw, 1);
        feature_matrix = repmat([feature_variance_VM, feature_variance_Z, feature_y_slope, feature_dominant_axis], num_timesteps, 1);
        
        combined_window = [window_raw, feature_matrix];
        
        X_val{end+1} = combined_window';
        windowStartSamples(end+1) = j;
    end
    
    if isempty(X_val), fprintf('\n--- TEST RESULTS ---\nError: Not enough data.\n'); return; end
    
    windowPredictions = classify(rnnNet, X_val);
    
    vectorMag_full = sqrt(sum(filteredAccel.^2, 2));
    alpha = 2 / (emaSpan + 1);
    ema_b = alpha;
    ema_a = [1, alpha - 1];
    smoothedVectorMag = filter(ema_b, ema_a, vectorMag_full);
    
    [pks_total, ~] = findpeaks(smoothedVectorMag, 'MinPeakHeight', minPeakHeight, 'MinPeakDistance', minPeakDistance);
    if length(pks_total) > 1
        dynamicThreshold_total = prctile(pks_total, 75);
        [~, stepLocations] = findpeaks(smoothedVectorMag, 'MinPeakHeight', dynamicThreshold_total, 'MinPeakDistance', minPeakDistance);
        totalSteps = length(stepLocations);
    else
        totalSteps = length(pks_total);
    end
    
    fprintf('\n--- CHRONOLOGICAL RESULTS FOR: %s ---\n', fileName);
    currentActivity = windowPredictions(1);
    segmentStartSample = windowStartSamples(1);
    stepsReported = 0;
    for k = 2:length(windowPredictions)
        if windowPredictions(k) ~= currentActivity
            segmentEndSample = windowStartSamples(k-1) + windowLength;
            stepsInSegment = sum(stepLocations >= segmentStartSample & stepLocations < segmentEndSample);
            stepsReported = stepsReported + stepsInSegment;
            startTime = (segmentStartSample-1)/targetSampleRate;
            endTime = (segmentEndSample-1)/targetSampleRate;
            fprintf('  - From %.1fs to %.1fs: Activity: %s, Steps: %d\n', startTime, endTime, string(currentActivity), stepsInSegment);
            currentActivity = windowPredictions(k);
            segmentStartSample = windowStartSamples(k);
        end
    end
    segmentEndSample = size(filteredAccel, 1);
    stepsInSegment = sum(stepLocations >= segmentStartSample & stepLocations < segmentEndSample);
    stepsReported = stepsReported + stepsInSegment;
    startTime = (segmentStartSample-1)/targetSampleRate;
    endTime = (segmentEndSample-1)/targetSampleRate;
    fprintf('  - From %.1fs to %.1fs: Activity: %s, Steps: %d\n', startTime, endTime, string(currentActivity), stepsInSegment);
    
    adjustmentFactor = originalSampleRate / targetSampleRate;
    if adjustmentFactor > 1.1 
        originalCount = totalSteps;
        adjustedTotalSteps = round(totalSteps / adjustmentFactor);
        fprintf('\n  NOTE: Applied dynamic step count adjustment (factor of %.2f).\n', adjustmentFactor);
        fprintf('  Original count: %d, Adjusted count: %d\n', originalCount, adjustedTotalSteps);
    else
        adjustedTotalSteps = totalSteps;
    end
    
    fprintf('\n------------------------------------------\n');
    fprintf('Total Steps Detected: %d\n', adjustedTotalSteps);
    fprintf('------------------------------------------\n');
    
    finalActivity = mode(windowPredictions);
    figure;
    t = (0:length(smoothedVectorMag)-1) / targetSampleRate;
    plot(t, smoothedVectorMag); hold on;
    plot(t(stepLocations), smoothedVectorMag(stepLocations), 'rv', 'MarkerFaceColor', 'r');
    title({['Test Results for: ' fileName], ['Overall Activity: ' char(finalActivity) ', Total Steps: ' num2str(adjustedTotalSteps)]}, 'Interpreter', 'none');
    xlabel('Time (s)'); ylabel('Acceleration Magnitude (m/s^2)');
    legend('Smoothed Vector Magnitude', 'Detected Steps'); grid on;
end

% --- INTERACTIVE THRESHOLD TUNER FUNCTION (Preserved but not called) ---
function runThresholdTuner(targetSampleRate, firOrder, cutoffFreq)
    fprintf('=== RUNNING IN THRESHOLD TUNING MODE ===\n\n');
    phoneModels = {'iphone', 'nothingphone'};
    activities = {'idle', 'walking', 'climbing_upstairs', 'climbing_downstairs'};
    all_VM_Vars = [];
    all_Z_Vars = [];
    all_Labels = categorical();
    lpFilt = designfilt('lowpassfir', 'FilterOrder', firOrder, 'CutoffFrequency', cutoffFreq, 'SampleRate', targetSampleRate);
    for p = 1:length(phoneModels)
        phoneModel = phoneModels{p};
        for i = 1:length(activities)
            activity = activities{i};
            fileName = sprintf('Acceleration_with_g_%s_%s.xls', activity, phoneModel);
            try
                T = readtable(fileName);
                accelData = T{:, 2:4};
                originalSampleRate = 100;
                if strcmp(phoneModel, 'nothingphone'), originalSampleRate = 416.8; end
                if originalSampleRate ~= targetSampleRate
                    accelData = resample(accelData, targetSampleRate, round(originalSampleRate));
                end
                filteredAccel = filter(lpFilt, accelData);
                vectorMag = sqrt(sum(filteredAccel.^2, 2));
                zAxisData = filteredAccel(:, 3);
                windowLength = targetSampleRate;
                for j = 1:windowLength:(size(vectorMag, 1) - windowLength)
                    all_VM_Vars(end+1) = var(vectorMag(j : j + windowLength - 1));
                    all_Z_Vars(end+1) = var(zAxisData(j : j + windowLength - 1));
                    all_Labels(end+1) = categorical({activity});
                end
                fprintf('  Processed: %s\n', fileName);
            catch
                fprintf('  Could not find or plot: %s\n', fileName);
            end
        end
    end
    if isempty(all_Labels), error('No data was processed.'); end
    figure; hold on; grid on;
    plot(all_VM_Vars(all_Labels == 'idle'), 1*ones(1, sum(all_Labels == 'idle')), 'bo', 'MarkerFaceColor', 'b');
    plot(all_VM_Vars(all_Labels == 'walking'), 2*ones(1, sum(all_Labels == 'walking')), 'ro', 'MarkerFaceColor', 'r');
    plot(all_VM_Vars(all_Labels == 'climbing_upstairs'), 3*ones(1, sum(all_Labels == 'climbing_upstairs')), 'go', 'MarkerFaceColor', 'g');
    plot(all_VM_Vars(all_Labels == 'climbing_downstairs'), 4*ones(1, sum(all_Labels == 'climbing_downstairs')), 'mo', 'MarkerFaceColor', 'm');
    set(gca, 'YTick', 1:4, 'YTickLabel', activities);
    xlabel('Vector Magnitude Variance');
    title('STAGE 1: Click to set a threshold between IDLE and ACTIVE states');
    fprintf('\nClick once on the plot to set the IDLE threshold, then press Enter.\n');
    [idle_threshold, ~] = ginput(1);
    if isempty(idle_threshold), disp('Aborting.'); return; end
    line([idle_threshold idle_threshold], ylim, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
    fprintf('  -> Idle Threshold set to: %.4f\n', idle_threshold);
    pause(1);
    figure; hold on; grid on;
    plot(all_Z_Vars(all_Labels == 'walking'), 1*ones(1, sum(all_Labels == 'walking')), 'ro', 'MarkerFaceColor', 'r');
    plot(all_Z_Vars(all_Labels == 'climbing_upstairs'), 2*ones(1, sum(all_Labels == 'climbing_upstairs')), 'go', 'MarkerFaceColor', 'g');
    plot(all_Z_Vars(all_Labels == 'climbing_downstairs'), 3*ones(1, sum(all_Labels == 'climbing_downstairs')), 'mo', 'MarkerFaceColor', 'm');
    set(gca, 'YTick', 1:3, 'YTickLabel', {'Walking', 'Upstairs', 'Downstairs'});
    xlabel('Z-Axis Variance');
    title('STAGE 2: Click to set a threshold between WALKING and STAIRS');
    fprintf('\nClick once on the plot to set the STAIRS threshold, then press Enter.\n');
    [stairs_threshold, ~] = ginput(1);
    if isempty(stairs_threshold), disp('Aborting.'); return; end
    line([stairs_threshold stairs_threshold], ylim, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
    fprintf('  -> Stairs Threshold set to: %.4f\n', stairs_threshold);
    fprintf('\n\n--- YOUR FINAL RULE-BASED CLASSIFICATION LOGIC ---\n');
    fprintf('You can use this logic in a rule-based analysis script:\n\n');
    fprintf('IF variance_VM < %.4f THEN\n', idle_threshold);
    fprintf('    Activity = ''Idle'';\n');
    fprintf('ELSE\n');
    fprintf('    IF variance_Z > %.4f THEN\n', stairs_threshold);
    fprintf('        Activity = ''Stairs'';\n');
    fprintf('    ELSE\n');
    fprintf('        Activity = ''Walking'';\n');
    fprintf('    END\n');
    fprintf('END\n');
end
