%% ============================================================
%  INITIALIZATION
% ============================================================

clc;
clear;

modelname = "cifar";

% Run this script from the directory containing the experiment CSV files.

% Experiment IDs
experiments = [10, 100, 1000];
% experiments = [0, 1, 2, 3, 4, 5, 6, 7, 8];

n_experiments = length(experiments);

% Norm
norm = "E";

% Data type
type = "union";

% Number of points to save for plotting
targetN = 100;


%% ============================================================
%  OUTPUT DIRECTORIES
% ============================================================

baseDir = pwd;

outStatsDir = fullfile(baseDir, "stats");
outPlotDir  = fullfile(baseDir, "plots");

if ~exist(outStatsDir, "dir")
    mkdir(outStatsDir);
end

if ~exist(outPlotDir, "dir")
    mkdir(outPlotDir);
end


%% ============================================================
%  STORAGE
% ============================================================

L_tr = zeros(n_experiments, 1);

t_all    = cell(n_experiments, 1);
data_all = cell(n_experiments, 1);
tr_all   = cell(n_experiments, 1);
un_all   = cell(n_experiments, 1);


%% ============================================================
%  PROCESS EACH EXPERIMENT
% ============================================================

for i = 1:n_experiments

    k = experiments(i);

    fprintf("\n");
    fprintf("============================================\n");
    fprintf("Processing experiment %d\n", k);
    fprintf("============================================\n");


    %% --------------------------------------------------------
    %  READ t / DELTA DATA
    % ---------------------------------------------------------

    deltaFile = sprintf( ...
        "batchdeltas_dmoc_%d_%s.csv", ...
        k, norm);

    fprintf("Reading: %s\n", deltaFile);

    deltas = readmatrix(deltaFile);

    % First column contains t
    t = deltas(:, 1);
    t = t(:);

    nT = length(t);

    fprintf("Number of original points: %d\n", nT);


    %% --------------------------------------------------------
    %  READ DATA
    % ---------------------------------------------------------

    dataFile = sprintf( ...
        "data_batchunion_dmoc_%d_%s.csv", ...
        k, norm);

    fprintf("Reading: %s\n", dataFile);

    data = readmatrix(dataFile);

    % Use the first column
    data = data(:, 1);
    data = data(:);


    %% --------------------------------------------------------
    %  READ TRAINED DATA
    % ---------------------------------------------------------

    trainedFile = sprintf( ...
        "trained_batchunion_dmoc_%d_%s.csv", ...
        k, norm);

    fprintf("Reading: %s\n", trainedFile);

    trained = readmatrix(trainedFile);

    % Use the first column
    trained = trained(:, 1);
    trained = trained(:);


    %% --------------------------------------------------------
    %  READ UNTRAINED DATA
    % ---------------------------------------------------------

    untrainedFile = sprintf( ...
        "untrained_batchunion_dmoc_%d_%s.csv", ...
        k, norm);

    fprintf("Reading: %s\n", untrainedFile);

    untrained = readmatrix(untrainedFile);

    % Use the first column
    untrained = untrained(:, 1);
    untrained = untrained(:);


    %% ========================================================
    %  CHECK DATA SIZES
    % ========================================================

    if length(t) ~= length(data)
        error( ...
            "Data size mismatch for experiment %d: " + ...
            "t has %d rows, data has %d rows.", ...
            k, length(t), length(data));
    end

    if length(t) ~= length(trained)
        error( ...
            "Trained data size mismatch for experiment %d: " + ...
            "t has %d rows, trained has %d rows.", ...
            k, length(t), length(trained));
    end

    if length(t) ~= length(untrained)
        error( ...
            "Untrained data size mismatch for experiment %d: " + ...
            "t has %d rows, untrained has %d rows.", ...
            k, length(t), length(untrained));
    end


    %% ========================================================
    %  CHECK FOR NON-FINITE VALUES
    % ========================================================

    if any(~isfinite(t))
        error( ...
            "Non-finite values found in t-grid for experiment %d.", ...
            k);
    end

    if any(~isfinite(data))
        warning( ...
            "Non-finite values found in data for experiment %d.", ...
            k);
    end

    if any(~isfinite(trained))
        warning( ...
            "Non-finite values found in trained data for experiment %d.", ...
            k);
    end

    if any(~isfinite(untrained))
        warning( ...
            "Non-finite values found in untrained data for experiment %d.", ...
            k);
    end


    %% ========================================================
    %  CHECK THAT t IS INCREASING
    % ========================================================

    if any(diff(t) <= 0)
        warning( ...
            "t-grid is not strictly increasing for experiment %d.", ...
            k);
    end


    %% ========================================================
    %  STORE FULL DATA
    % ========================================================

    t_all{i}    = t;
    data_all{i} = data;
    tr_all{i}   = trained;
    un_all{i}   = untrained;


    %% ========================================================
    %  SELECT POINTS FOR PLOTTING
    % ========================================================

    if nT <= targetN

        % Keep all points
        idx = 1:nT;

    else

        % Select approximately targetN evenly spaced points.
        % Always keep the first and last points.

        idx_mid = round( ...
            linspace(2, nT - 1, targetN - 2));

        idx = [1, idx_mid, nT];

    end


    %% ========================================================
    %  CREATE OUTPUT FOR PLOTTING
    % ========================================================

    % Columns:
    %
    % 1 = t
    % 2 = data
    % 3 = trained
    % 4 = untrained

    output = [
        t(idx), ...
        data(idx), ...
        trained(idx), ...
        untrained(idx)
    ];


    %% ========================================================
    %  SAVE PLOTTING DATA
    % ========================================================

    outFile = fullfile( ...
        outPlotDir, ...
        sprintf( ...
            "%s_%d_%s_%s.txt", ...
            modelname, k, type, norm));

    writematrix( ...
        output, ...
        outFile, ...
        "Delimiter", "space");

    fprintf( ...
        "Saved %d points to:\n%s\n", ...
        size(output, 1), ...
        outFile);


    %% ========================================================
    %  COMPUTE TRAINED LIPSCHITZ RATIO
    % ========================================================

    % Compute:
    %
    %       L = max_{t > 0} trained(t) / t

    valid = t > 0;

    if ~any(valid)
        warning( ...
            "No positive t values for experiment %d. " + ...
            "Lipschitz ratio not calculated.", ...
            k);

        L_tr(i) = NaN;

    else

        ratios = trained(valid) ./ t(valid);

        L_tr(i) = max(ratios);

    end

    fprintf( ...
        "Lipschitz ratio for experiment %d: %.15g\n", ...
        k, L_tr(i));

end


%% ============================================================
%  SAVE LIPSCHITZ RATIOS
% ============================================================

lipschitzOutput = [
    experiments(:), ...
    L_tr
];

lipschitzFile = fullfile( ...
    outStatsDir, ...
    sprintf( ...
        "%s_lipschitz_ratio_%s_%s.txt", ...
        modelname, type, norm));

writematrix( ...
    lipschitzOutput, ...
    lipschitzFile, ...
    "Delimiter", "space");

fprintf("\n");
fprintf("Lipschitz results saved to:\n");
fprintf("%s\n", lipschitzFile);


%% ============================================================
%  CHECK WHETHER ALL EXPERIMENTS HAVE THE SAME t-GRID
% ============================================================

sameGrid = true;

nT_first = length(t_all{1});

for i = 2:n_experiments

    if length(t_all{i}) ~= nT_first
        sameGrid = false;
        break;
    end

    if max(abs(t_all{i} - t_all{1})) > 1e-12
        sameGrid = false;
        break;
    end

end


%% ============================================================
%  COMPUTE MEAN AND STANDARD DEVIATION
% ============================================================

if sameGrid

    fprintf("\nAll experiments have the same t-grid.\n");
    fprintf("Computing mean and standard deviation...\n");


    %% --------------------------------------------------------
    %  CREATE MATRICES
    % ---------------------------------------------------------

    % Rows    = experiments
    % Columns = t values

    data_matrix = zeros(n_experiments, nT_first);
    tr_matrix   = zeros(n_experiments, nT_first);
    un_matrix   = zeros(n_experiments, nT_first);

    for i = 1:n_experiments

        data_matrix(i, :) = data_all{i}';
        tr_matrix(i, :)   = tr_all{i}';
        un_matrix(i, :)   = un_all{i}';

    end


    %% --------------------------------------------------------
    %  MEAN
    % ---------------------------------------------------------

    mean_data = mean(data_matrix, 1);
    mean_tr   = mean(tr_matrix, 1);
    mean_un   = mean(un_matrix, 1);


    %% --------------------------------------------------------
    %  STANDARD DEVIATION
    % ---------------------------------------------------------

    std_data = std(data_matrix, 0, 1);
    std_tr   = std(tr_matrix, 0, 1);
    std_un   = std(un_matrix, 0, 1);


    %% --------------------------------------------------------
    %  COMMON t-GRID
    % ---------------------------------------------------------

    tgrid = t_all{1};


    %% ========================================================
    %  SAVE DATA MEAN / STD
    % ========================================================

    stats_data = [
        tgrid, ...
        mean_data', ...
        std_data'
    ];

    dataStatsFile = fullfile( ...
        outStatsDir, ...
        sprintf( ...
            "%s_stats_%s_data_%s.txt", ...
            modelname, type, norm));

    writematrix( ...
        stats_data, ...
        dataStatsFile, ...
        "Delimiter", "space");


    %% ========================================================
    %  SAVE TRAINED MEAN / STD
    % ========================================================

    stats_tr = [
        tgrid, ...
        mean_tr', ...
        std_tr'
    ];

    trainedStatsFile = fullfile( ...
        outStatsDir, ...
        sprintf( ...
            "%s_stats_%s_trained_%s.txt", ...
            modelname, type, norm));

    writematrix( ...
        stats_tr, ...
        trainedStatsFile, ...
        "Delimiter", "space");


    %% ========================================================
    %  SAVE UNTRAINED MEAN / STD
    % ========================================================

    stats_un = [
        tgrid, ...
        mean_un', ...
        std_un'
    ];

    untrainedStatsFile = fullfile( ...
        outStatsDir, ...
        sprintf( ...
            "%s_stats_%s_untrained_%s.txt", ...
            modelname, type, norm));

    writematrix( ...
        stats_un, ...
        untrainedStatsFile, ...
        "Delimiter", "space");


    fprintf("\nMean/std statistics saved.\n");

else

    warning( ...
        "Experiments do not share the same t-grid. " + ...
        "Mean/std statistics were not saved.");

end


%% ============================================================
%  FINISHED
% ============================================================

fprintf("\n");
fprintf("============================================\n");
fprintf("ALL EXPERIMENTS FINISHED\n");
fprintf("============================================\n");

fprintf("\nGenerated plotting files:\n");

for i = 1:n_experiments

    k = experiments(i);

    fprintf( ...
        "  %s\n", ...
        fullfile( ...
            outPlotDir, ...
            sprintf( ...
                "%s_%d_%s_%s.txt", ...
                modelname, k, type, norm)));

end

fprintf("\nPlotting file columns:\n");
fprintf("  Column 1: t\n");
fprintf("  Column 2: data\n");
fprintf("  Column 3: trained\n");
fprintf("  Column 4: untrained\n");

fprintf("\nFinished.\n");