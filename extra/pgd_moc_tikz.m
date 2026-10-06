clear;
clc;
modelname = "iris";

%enter the related experiments model directory

% Experiment IDs
experiments = [0,4,8];

n_experiments = length(experiments);

% Norm
norm = "L2";

% Data type
type = "union";

% Number of points to save to the output files
targetN = 100;



% Current directory
baseDir = pwd;

% Directory for statistics
outStatsDir = fullfile(baseDir, 'stats');

if ~exist(outStatsDir, 'dir')
    mkdir(outStatsDir);
end

% Directory for plotting data
outPlotDir = fullfile(baseDir, 'plots');

if ~exist(outPlotDir, 'dir')
    mkdir(outPlotDir);
end



L_tr = zeros(n_experiments,1);
t_all  = cell(n_experiments,1);
%data_all = cell(n_experiments,1);
tr_all = cell(n_experiments,1);
un_all = cell(n_experiments,1);



for i = 1:n_experiments

    % Actual experiment ID
    k = experiments(i);


    deltaFile = sprintf( ...
        'pgd_deltas_dmoc_%d_%s.csv', ...
        k, norm);

    fprintf('Reading: %s\n', deltaFile);

    deltas = readmatrix(deltaFile);

    % First column is t
    t = deltas(:,1);

    nT = length(t);

    fprintf('Number of original points: %d\n', nT);


   %  dataFile = sprintf( ...
   %      'pgd_union_data_dmoc_%s.csv', ...
   % norm);
   % 
   %  fprintf('Reading: %s\n', dataFile);
   % 
   %  data = readmatrix(dataFile);


  
    trainedFile = sprintf( ...
        'pgd_trained_dmoc_%d_%s.csv', ...
        k, norm);

    fprintf('Reading: %s\n', trainedFile);

    trained = readmatrix(trainedFile);

    untrainedFile = sprintf( ...
        'pgd_untrained_dmoc_%d_%s.csv', ...
        k,norm);

    fprintf('Reading: %s\n', untrainedFile);

    untrained = readmatrix(untrainedFile);



    t = t(:);

    % data = data(:,1);
    % data = data(:);

    trained = trained(:,1);
    trained = trained(:);

    untrained = untrained(:,1);
    untrained = untrained(:);


    % if length(t) ~= length(data)
    % 
    %     error( ...
    %         'Data size mismatch for experiment %d: t has %d rows, data has %d rows.', ...
    %         k, length(t), length(data));
    % 
    % end


    if length(t) ~= length(trained)

        error( ...
            'Trained data size mismatch for experiment %d: t has %d rows, trained has %d rows.', ...
            k, length(t), length(trained));

    end


    if length(t) ~= length(untrained)

        error( ...
            'Untrained data size mismatch for experiment %d: t has %d rows, untrained has %d rows.', ...
            k, length(t), length(untrained));

    end



    if any(~isfinite(t))

        error( ...
            'Non-finite values found in t-grid for experiment %d.', ...
            k);

    end


    % if any(~isfinite(data))
    % 
    %     warning( ...
    %         'Non-finite values found in data for experiment %d.', ...
    %         k);
    % 
    % end


    if any(~isfinite(trained))

        warning( ...
            'Non-finite values found in trained data for experiment %d.', ...
            k);

    end


    if any(~isfinite(untrained))

        warning( ...
            'Non-finite values found in untrained data for experiment %d.', ...
            k);

    end


    %% ========================================================
    %  CHECK THAT T-GRID IS STRICTLY INCREASING
    % =========================================================

    if any(diff(t) <= 0)

        warning( ...
            't-grid is not strictly increasing for experiment %d.', ...
            k);

    end


    %% ========================================================
    %  STORE FULL DATA
    % =========================================================

    t_all{i}  = t;
    %data_all{i} = data;
    tr_all{i} = trained;
    un_all{i} = untrained;


    %% ========================================================
    %  SELECT 100 POINTS FOR PLOTTING
    % =========================================================

    if nT <= targetN

        % If there are already 100 or fewer points,
        % keep every point.

        idx = 1:nT;

    else

        % Select approximately targetN evenly spaced points.
        %
        % The first point is always included.
        % The last point is always included.

        idx_mid = round( ...
            linspace( ...
                2, ...
                nT-1, ...
                targetN-2));

        idx = [1, idx_mid, nT];

    end


    %% ========================================================
    %  CREATE FOUR-COLUMN OUTPUT
    % =========================================================

    % Column 1: t
    % Column 3: trained
    % Column 4: untrained

    output = [t(idx), trained(idx), untrained(idx)];



    %% ========================================================
    %  OUTPUT FILENAME
    % =========================================================

    outFile = fullfile( ...
        outPlotDir, ...
        sprintf( ...
            '%s_%d_%s_%s_pgd.txt', ...
            modelname,k, type, norm));


    %% ========================================================
    %  WRITE OUTPUT FILE
    % =========================================================

    writematrix( ...
        output, ...
        outFile, ...
        'Delimiter', 'space');


    fprintf( ...
        'Saved %d points to:\n%s\n', ...
        size(output,1), ...
        outFile);


    %% ========================================================
    %  COMPUTE LIPSCHITZ RATIO
    % =========================================================

    % Compute:
    %
    %       L = max_{t > 0} omega(t) / t
    %
    % Here omega(t) is assumed to be the TRAINED curve.

    valid = t > 0;

    ratios = trained(valid) ./ t(valid);

    L_tr(i) = max(ratios);


    fprintf( ...
        'Lipschitz ratio for experiment %d: %.15g\n', ...
        k, ...
        L_tr(i));


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
        '%s_lipschitz_ratio_%s_%s.txt', ...
        modelname, ...
        type, ...
        norm));


writematrix( ...
    lipschitzOutput, ...
    lipschitzFile, ...
    'Delimiter', 'space');


fprintf('\n');
fprintf('============================================\n');
fprintf('Lipschitz results saved to:\n');
fprintf('%s\n', lipschitzFile);
fprintf('============================================\n');


%% ============================================================
%  COMPUTE MEAN AND STANDARD DEVIATION
% ============================================================

% Only possible if all experiments have the same t-grid length.

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


if sameGrid

    % Create matrices:
    %
    % rows    = experiments
    % columns = t values

    data_matrix = zeros(n_experiments, nT_first);
    tr_matrix   = zeros(n_experiments, nT_first);
    un_matrix   = zeros(n_experiments, nT_first);

    for i = 1:n_experiments

        %data_matrix(i,:) = data_all{i}';
        tr_matrix(i,:)   = tr_all{i}';
        un_matrix(i,:)   = un_all{i}';

    end


    % Mean
    mean_data = mean(data_matrix, 1);
    mean_tr   = mean(tr_matrix, 1);
    mean_un   = mean(un_matrix, 1);


    % Standard deviation
    std_data = std(data_matrix, 0, 1);
    std_tr   = std(tr_matrix, 0, 1);
    std_un   = std(un_matrix, 0, 1);


    %% ========================================================
    %  SAVE MEAN / STD STATISTICS
    % =========================================================

    tgrid = t_all{1};


    % Save data statistics
    stats_data = [
        tgrid, ...
        mean_data', ...
        std_data'
    ];

    writematrix( ...
        stats_data, ...
        fullfile( ...
            outStatsDir, ...
            sprintf( ...
                '%s_stats_%s_data_%s.txt', ...
                modelname, ...
                type, ...
                norm)), ...
        'Delimiter', 'space');


    % Save trained statistics
    stats_tr = [
        tgrid, ...
        mean_tr', ...
        std_tr'
    ];

    writematrix( ...
        stats_tr, ...
        fullfile( ...
            outStatsDir, ...
            sprintf( ...
                '%s_stats_%s_trained_%s.txt', ...
                modelname, ...
                type, ...
                norm)), ...
        'Delimiter', 'space');


    % Save untrained statistics
    stats_un = [
        tgrid, ...
        mean_un', ...
        std_un'
    ];

    writematrix( ...
        stats_un, ...
        fullfile( ...
            outStatsDir, ...
            sprintf( ...
                '%s_stats_%s_untrained_%s.txt', ...
                modelname, ...
                type, ...
                norm)), ...
        'Delimiter', 'space');


else

    warning( ...
        'Experiments do not share the same t-grid. Mean/std statistics were not saved.');

end


%% ============================================================
%  FINISHED
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('ALL EXPERIMENTS FINISHED\n');
fprintf('============================================\n');

fprintf('\nGenerated plotting files:\n');

% for i = 1:n_experiments
% 
%     k = experiments(i);
% 
%     fprintf( ...
%         '  %s\n', ...
%         fullfile( ...
%             outPlotDir, ...
%             sprintf( ...
%                 'mnist_%d_%s_%s_.txt', ...
%                 k, type, norm)));
% 
% end

fprintf('\n');
fprintf('Each plotting file has columns:\n');
fprintf('  Column 0: t\n');
%fprintf('  Column 1: data\n');
fprintf('  Column 1: trained\n');
fprintf('  Column 2: untrained\n');