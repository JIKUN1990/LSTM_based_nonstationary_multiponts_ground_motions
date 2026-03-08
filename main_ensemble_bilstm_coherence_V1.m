%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Ensembles + BiLSTM Sequence Learning Framework
% For Epistemic Uncertainty Quantification in Nonstationary Spatial Coherence
% Coded by Kun Ji and Pan Wen
% Please Cite  >
% [Kun Ji, Pan Wen.,et al.] (2026). Modeling Nonstationary Seismic Spatial Coherence Using Wavelet Packet Transform 
% and Deep Recurrent Neural Networks. *[CACIE]*. 

% Description:
% This script trains a deep ensemble of Bidirectional LSTM (BiLSTM) networks 
% to learn the non-linear mapping between spatial coherence and its governing 
% parameters (Normalized Intensity En, Frequency, and Spatial Separation).
% It performs sequence-level bootstrap resampling and outputs predictions 
% with 95% confidence intervals.
%
% Expected Input Data Format (Excel/CSV without headers):
% Column 1: Normalized Intensity (En)
% Column 2: Frequency (Hz)
% Column 3: Separation Distance (m)
% Column 4: Target Coherence (0~1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
rng(42); % Fix random seed for reproducibility

%% =========================================================================
% 0. CONFIGURATION PARAMETERS (User-defined)
% =========================================================================
% File settings
TRAIN_DATA_FILE = 'LSTM_train_data.xlsx'; % Replace with your training data file


% Sequence and Model settings
SEQ_LENGTH      = 91;   % Sequence length (e.g., number of En points per frequency block)
NUM_MODELS      = 5;    % Number of ensemble models (5 is recommended for diversity)
NUM_HIDDEN      = 128;  % Number of hidden units in BiLSTM

% Training hyperparameters
MAX_EPOCHS      = 120;
MINI_BATCH_SIZE = 256;
INITIAL_LR      = 0.01;

%% =========================================================================
% 1. DATA PREPARATION & SEQUENCE RECONSTRUCTION
% =========================================================================
fprintf('1. Loading and reconstructing training data...\n');
data = readmatrix(TRAIN_DATA_FILE); 
X_all = data(:, 1:3); % Features: [En, Frequency, Distance]
Y_all = data(:, 4);   % Target: Coherence

% 1.1 Data Normalization (Save mapping parameters for later use)
[X_norm, input_ps] = mapminmax(X_all', 0, 1);
[Y_norm, output_ps] = mapminmax(Y_all', 0, 1);
X_norm = X_norm'; Y_norm = Y_norm';

% 1.2 Physical Sequence Reconstruction
num_samples = size(X_norm, 1);
num_sequences = floor(num_samples / SEQ_LENGTH);

% Truncate incomplete tails
X_norm = X_norm(1:num_sequences * SEQ_LENGTH, :);
Y_norm = Y_norm(1:num_sequences * SEQ_LENGTH, :);

% Assemble into cell arrays required by LSTM [num_sequences x 1]
X_All_Seq = cell(num_sequences, 1);
Y_All_Seq = cell(num_sequences, 1);
for i = 1:num_sequences
    start_idx = (i-1)*SEQ_LENGTH + 1;
    end_idx = i*SEQ_LENGTH;
    X_All_Seq{i} = X_norm(start_idx:end_idx, :)';  % [Num Features(3) x Sequence Length]
    Y_All_Seq{i} = Y_norm(start_idx:end_idx, :)';
end

% 1.3 Split Training Pool and Validation Set (70% Train, 30% Val)
num_train_base = round(0.7 * num_sequences);
idx_shuffle = randperm(num_sequences);
train_idx_base = idx_shuffle(1:num_train_base);
val_idx = idx_shuffle(num_train_base+1:end);

XVal = X_All_Seq(val_idx);
YVal = Y_All_Seq(val_idx);

%% =========================================================================
% 2. DEEP ENSEMBLES TRAINING
% =========================================================================
models = cell(NUM_MODELS, 1); 
inputSize = size(X_all, 2); 

fprintf('\n2. Training Deep Ensemble Models (Total: %d)...\n', NUM_MODELS);

for k = 1:NUM_MODELS
    fprintf('=== Training Model %d / %d ===\n', k, NUM_MODELS);
    
    % Sequence-level Bootstrap Resampling
    % Each model is trained on a slightly different subset of the training pool
    idx_resample = randi(num_train_base, num_train_base, 1); 
    XTrain_Sub = X_All_Seq(train_idx_base(idx_resample));
    YTrain_Sub = Y_All_Seq(train_idx_base(idx_resample));
    
    % Define Enhanced BiLSTM Architecture
    layers = [ ...
        sequenceInputLayer(inputSize, 'Name', 'input')
        bilstmLayer(NUM_HIDDEN, 'OutputMode', 'sequence', 'Name', 'bilstm1')
        dropoutLayer(0.3, 'Name', 'drop1') % Prevent overfitting
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(1, 'Name', 'fc_out')
        regressionLayer('Name', 'output')];
    
    % Training Options
    options = trainingOptions('adam', ...
        'MaxEpochs', MAX_EPOCHS, ...
        'MiniBatchSize', MINI_BATCH_SIZE, ... 
        'InitialLearnRate', INITIAL_LR, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'L2Regularization', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', 30, ...
        'Verbose', 0, ...      % Suppress detailed console output
        'Plots', 'none');      % Suppress pop-up windows
        
    % Train and store the model
    models{k} = trainNetwork(XTrain_Sub, YTrain_Sub, layers, options);
end

% Save ensemble models and normalization parameters
save('seismic_ensemble_bilstm.mat', 'models', 'input_ps', 'output_ps', 'SEQ_LENGTH');
fprintf('Training complete! Models saved to ''seismic_ensemble_bilstm.mat''.\n\n');

%% =========================================================================
% 3. UNCERTAINTY QUANTIFICATION ON UNSEEN EVENT
% =========================================================================
TEST_DATA_FILE  = 'LSTM_test_data.xlsx';  % Replace with your testing/target data file
fprintf('3. Predicting and quantifying uncertainty on test data...\n');
data_pred = readmatrix(TEST_DATA_FILE); 
X_new = data_pred(:, 1:3);
Y_new_true = data_pred(:, 4);

% 3.1 Normalize and sequence new data (strictly maintaining consistency)
X_new_norm = mapminmax('apply', X_new', input_ps)';
num_new_samples = size(X_new_norm, 1);
num_new_sequences = floor(num_new_samples / SEQ_LENGTH);

X_new_norm = X_new_norm(1:num_new_sequences * SEQ_LENGTH, :);
Y_new_true = Y_new_true(1:num_new_sequences * SEQ_LENGTH, :);

X_new_cell = cell(num_new_sequences, 1);
for i = 1:num_new_sequences
    X_new_cell{i} = X_new_norm((i-1)*SEQ_LENGTH + 1 : i*SEQ_LENGTH, :)';
end

% 3.2 Predict using all ensemble models
num_valid_points = num_new_sequences * SEQ_LENGTH;
ensemble_preds = zeros(num_valid_points, NUM_MODELS);

for k = 1:NUM_MODELS
    Y_pred_cell = predict(models{k}, X_new_cell);
    Y_pred_norm_flat = cell2mat(Y_pred_cell')'; 
    Y_pred_real = mapminmax('reverse', Y_pred_norm_flat', output_ps)';
    ensemble_preds(:, k) = Y_pred_real;
end

% 3.3 Calculate Epistemic Uncertainty (Mean and 95% CI)
pred_mean = mean(ensemble_preds, 2);
pred_std = std(ensemble_preds, 0, 2); 

lower_bound = pred_mean - 1.96 * pred_std;
upper_bound = pred_mean + 1.96 * pred_std;

% Physical constraint boundaries (Coherence must be [0, 1])
lower_bound(lower_bound < 0) = 0;
upper_bound(upper_bound > 1) = 1;
pred_mean(pred_mean > 1) = 1;
pred_mean(pred_mean < 0) = 0;

% Evaluate overall Pearson Correlation Coefficient (R)
R_matrix = corrcoef(Y_new_true, pred_mean);
R_val = R_matrix(1,2);
fprintf('====================================\n');
fprintf('Ensemble Mean Prediction R-value = %.4f\n', R_val);
fprintf('====================================\n');

save('prediction_with_uncertainty.mat', 'pred_mean', 'pred_std', 'ensemble_preds', 'Y_new_true', 'X_new');

