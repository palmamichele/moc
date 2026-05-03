%enter each folder containing initializations for same net
files = dir(pwd);
modelname="imagenet-minibatch"

outStatsDir = fullfile(pwd, 'stats');
% make sure the folder exists
if ~exist(outStatsDir, 'dir')
    mkdir(outStatsDir);
end

outPlotDir = fullfile(pwd, 'plots');
% make sure the folder exists
if ~exist(outPlotDir, 'dir')
    mkdir(outPlotDir);
end

% n_experiments = 0;
% for j = 1:length(files)
%     if ~files(j).isdir
%         %if ~isempty(regexp(files(j).name, '^model_.+', 'once'))
%         if ~isempty(regexp(files(j).name, '^minibatch_.+', 'once'))
%             n_experiments = n_experiments + 1;
%         end
%     end
% end

experiments = [10,100,1000,10000,70000];%[10,100,1000]%[10,100,1000,10000,70000];
n_experiments = length(experiments);
disp(n_experiments)
    
for norm = ["E", "T"]
    
    for type = ["union", "train", "test"]
        L_tr = zeros(n_experiments,1);
        deltas_all = cell(n_experiments,1);
        tr_all     = cell(n_experiments,1);
        un_all     = cell(n_experiments,1);

        for i = 1:n_experiments
            k = experiments(i);
            %load all net mocs (both trained, untrained)
            deltas_all{i} = readmatrix(sprintf('deltas_dmoc_%d_%s.csv', k, norm)); %readmatrix(sprintf('batchdeltas_dmoc_%d_%s.csv', k, norm)); %
            tr_all{i}     = readmatrix(sprintf('%s_trained_dmoc_%d_%s.csv', type, k, norm)); %readmatrix(sprintf('trained_batch%s_dmoc_%d_%s.csv', type, k, norm)); %
            %un_all{k+1}     = readmatrix(sprintf('%s_untrained_dmoc_%d_%s.csv', type, k, norm));
        end 
        %extract grid (same for all experiments)
        

        tgrid = deltas_all{1}(:,1);
        nT = length(tgrid);
        
        targetN = 100;
        idx_mid = round(linspace(2, nT-1, targetN-2));
        idx = [1, idx_mid, nT];

        omega_tr  = zeros(n_experiments, nT);
        omega_un  = zeros(n_experiments, nT);

        for i = 1:n_experiments
            tk = deltas_all{i}(:,1);

            if length(tk) ~= nT || max(abs(tk - tgrid)) > 1e-12
                error("Grid mismatch in experiment %d (ID=%d)", i, experiments(i));
            end

            if size(tr_all{i},1) ~= nT
                error("Train data mismatch in experiment %d (ID=%d)", i, experiments(i));
            end

            omega_tr(i,:) = tr_all{i}(:)';
        end

        for i = 1:n_experiments
            k = experiments(i);
            outFile = fullfile(outPlotDir, ...
                sprintf('%s_%d_%s_%s_%s.txt', modelname, k, type, norm));

            writematrix( ...
                [tgrid(idx), omega_tr(i, idx)'], ...
                outFile, ...
                'Delimiter', 'space');

            t = tgrid;
            w = omega_tr(i,:)';
     
            valid = t > 0;
            
            ratios = w(valid) ./ t(valid);
            
            L_tr(i) = max(ratios);
        end


        writematrix([experiments(:), L_tr], fullfile(outStatsDir, ...
    sprintf('%s_lipschitz_ratio_%s_%s.txt', modelname, type, norm)), ...
    'Delimiter','space');




        % %data_moc = readmatrix(sprintf('%s_data_dmoc_%s.csv', type, norm));
        % for k= experiments %1:n_experiments
        %     writematrix([tgrid(idx), omega_tr(k, idx)',], fullfile(outPlotDir, sprintf('%s_%d_%s_%s_%s.txt', modelname, k, type, norm)), 'Delimiter', 'space');
        %     %writematrix([tgrid(idx), data_moc(idx),omega_tr(k, idx)', omega_un(k, idx)'], fullfile(outPlotDir, sprintf('%s_%d_%s_%s_%s.txt', modelname, k-1, type, norm)), 'Delimiter', 'space');
        % end


        %stat across all runs (be careful, same model initializations not
        %experiments)
        mean_tr = mean(omega_tr, 1);
        std_tr  = std(omega_tr, 0, 1);

        % mean_un = mean(omega_un, 1);
        % std_un  = std(omega_un, 0, 1);
        % 
        % out_tr = [tgrid, mean_tr', std_tr'];
        % out_un = [tgrid, mean_un', std_un'];
        % 
        % %for tikz, select fewer t sites
        % targetN = 50;
        % idx_mid = round(linspace(2, nT-1, targetN-2));
        % idx = [1, idx_mid, nT];
        % 
        % out_tr = [tgrid(idx), mean_tr(idx)', std_tr(idx)'];
        % out_un = [tgrid(idx), mean_un(idx)', std_un(idx)'];
        % 
        % filename = sprintf('%s_stats_%s_trained_%s_%s.txt', modelname, type, norm);
        % 
        % 
        % writematrix(out_tr, fullfile(outStatsDir, sprintf('%s_stats_%s_trained_%s_%s.txt', modelname, type, norm)), 'Delimiter', 'space');
        % writematrix(out_un, fullfile(outStatsDir, sprintf('%s_stats_%s_untrained_%s_%s.txt', modelname, type, norm)), 'Delimiter', 'space');
        % 
        % 
        % disp("done type = " + type + ", norm = " + norm);


    end


end


% targetN = 50;
% idx_mid = round(linspace(2, nT-1, targetN-2));
% idx = [1, idx_mid, nT];
% writematrix([tgrid(idx), (idx)'], sprintf('%s_stats_%s_trained_%s_%s.txt', modelname, type, norm), 'Delimiter', 'space');
% 