% enter the moc folder
save_path = "experiments";
models = ["linear-california", "california", "iris", "MNIST"];
out_path = pwd;

out_csv = fullfile(out_path, "moc_stats.csv");
out_csv_m = fullfile(out_path, "datamoc_stats.csv");

T = table( ...
    strings(0,1), ... % model_path
    strings(0,1), ... % split_type
    strings(0,1), ... % norm
    zeros(0,1), ...   % model_id
    zeros(0,1), ...   % arel_tr
    zeros(0,1), ...   % arel_un
    zeros(0,1), ...   % norm_tr
    zeros(0,1), ...   % norm_un
    zeros(0,1), ...   % s_tr
    zeros(0,1), ...   % s_un
    zeros(0,1), ...   % r_tr
    zeros(0,1), ...   % r_un
    'VariableNames', { ...
        'model_path', ...
        'split_type', ...
        'norm', ...
        'model_id', ...
        'arel_data_tr', ...
        'arel_data_un', ...
        'norm_arel_data_tr', ...
        'norm_arel_data_un', ...
        'srel_data_tr', ...
        'srel_data_un', ...
        'r_data_tr', ...
        'r_data_un' ...
    } ...
);


M = table( ...
    strings(0,1), ... % model_path
    strings(0,1), ... % pairs
    strings(0,1), ... % norm
    zeros(0,1), ...   % arel
    zeros(0,1), ...   % norm
    zeros(0,1), ...   % s
    zeros(0,1), ...   % r
    'VariableNames', { ...
        'model_path', ...
        'pairs', ...
        'norm', ...
        'arel_data', ...
        'norm_arel_data', ...
        'srel_data', ...
        'r_data' ...
    } ...
);


%do this both for (datamoc, trained), (datamoc, untrained)

for mtype = models
    folder_path = fullfile(save_path, mtype);
    files = dir(folder_path);

    n_experiments = 0;
    for j = 1:length(files)
        if ~files(j).isdir
            if ~isempty(regexp(files(j).name, '^model_.+', 'once'))
                n_experiments = n_experiments + 1;
            end
        end
    end
    
    %compute stats for dataset across different splits



    for norm = ["E", "T"]

        %load data moc (no net)
        moc_union = readmatrix(fullfile(folder_path, sprintf('union_data_dmoc_%s.csv', norm)));
        moc_train= readmatrix(fullfile(folder_path, sprintf('train_data_dmoc_%s.csv', norm)));
        moc_test= readmatrix(fullfile(folder_path, sprintf('test_data_dmoc_%s.csv', norm)));
        
        m_data = {moc_union, moc_train, moc_test};

        names = {'union', 'train', 'test'}; 
        n = numel(m_data);
        
        for i = 1:n-1
            for j = i+1:n
                A = m_data{i};
                B = m_data{j};
                
                lbl= "data moc_" + names{i} + " & data moc_" + names{j};

                arel_data= Arel(A, B, 0);
               
                srel_data = exp(-arel_data);
               

  
                D = mean(abs((A)));
           
                newRow = table( ...
                    mtype, ...
                    lbl, ...
                    norm, ...
                    arel_data, ...
                    arel_data/D, ...
                    srel_data, ...
                    corr(A(:), B(:), 'Type', 'Pearson'), ...
                    'VariableNames', M.Properties.VariableNames);

                 M = [M; newRow];

                
                
            end
        end
    
        

    end
    writetable(M, out_csv_m);

  

    for k = 0:n_experiments-1

        for type = ["test", "train", "union"]

   
            for norm = ["E", "T"]
            
                disp(folder_path)
                data_file = fullfile(folder_path, sprintf('%s_data_dmoc_%s.csv', type, norm));
                tr_file   = fullfile(folder_path, sprintf('%s_trained_dmoc_%d_%s.csv', type, k, norm));
                un_file   = fullfile(folder_path, sprintf('%s_untrained_dmoc_%d_%s.csv', type, k, norm));
                
                

                moc_data = readmatrix(data_file);
                moc_tr   = readmatrix(tr_file);
                moc_un   = readmatrix(un_file);
                %grid = readmatrix(fullfile(folder_path, sprintf('deltas_dmoc_%d_%s.csv',k, norm)));

           

                assert(all(isfinite(moc_data)), 'moc_data contains NaN/Inf');
                assert(all(isfinite(moc_tr)),   'moc_tr contains NaN/Inf');
                assert(all(isfinite(moc_un)),   'moc_un contains NaN/Inf');

                assert(all(moc_data >= 0), 'moc_data has negative values');
                assert(all(moc_tr >= 0),   'moc_tr has negative values');
                assert(all(moc_un >= 0),   'moc_un has negative values');


                % disp(std(moc_data))
                % disp(std(moc_tr))
                % disp(std(moc_un))

                lblw = sprintf('%s_%d_%s_%s', folder_path, k, type, norm);

                % assert(numel(unique(moc_data)) > 1, ...
                %     ['moc_data is constant on ', lblw]);
                % 
                % assert(numel(unique(moc_tr)) > 1, ...
                %     ['moc_tr is constant on ', lblw]);
                % 
                % assert(numel(unique(moc_un)) > 1, ...
                %     ['moc_un is constant on ', lblw]);

                if numel(unique(moc_data)) <= 1
                    disp(['moc_data is constant on ', lblw]);
                    disp(moc_data)
                end
                
                if numel(unique(moc_tr)) <= 1
                    disp(['moc_tr is constant on ', lblw]);
                    disp(moc_tr)
                end
                
                if numel(unique(moc_un)) <= 1
                    disp(['moc_un is constant on ', lblw]);
                    disp(moc_un)
                end

                %take as idx the one that comes later and provides nonzero moc in moc of data, tr,
                %untr

                arel_data_tr = Arel(moc_data, moc_tr, 0);
                arel_data_un = Arel(moc_data, moc_un, 0);
                srel_data_tr  = exp(-arel_data_tr);
                srel_data_un  = exp(-arel_data_un);

   
                
                % arel_log_data_tr = Arel(moc_data(idx:end), moc_tr(idx:end), 1);
                % arel_log_data_un = Arel(moc_data(idx:end), moc_un(idx:end), 1);
                % srel_log_data_tr  = exp(-arel_log_data_tr);
                % srel_log_data_un  = exp(-arel_log_data_un);


                D = mean(abs((moc_data)));
           
                

                

            
                newRow = table( ...
                    mtype, ...
                    type, ...
                    norm, ...
                    k, ...
                    arel_data_tr, ...
                    arel_data_un, ...
                    arel_data_tr/D, ...
                    arel_data_un/D, ...
                    srel_data_tr, ...
                    srel_data_un, ...
                    corr(moc_data(:), moc_tr(:), 'Type', 'Pearson'), ...
                    corr(moc_data(:), moc_un(:), 'Type', 'Pearson'), ...
                    'VariableNames', T.Properties.VariableNames);

                 T = [T; newRow];

            end

            
        end 

    end

end 
writetable(T, out_csv);
disp("done")



function A = Arel(moc_data, moc_net, use_log_space)
    moc_data = moc_data(:);
    moc_net  = moc_net(:);

    if use_log_space
        delta = log(moc_net ./ moc_data);
    else
        delta = moc_net - moc_data;
    end

    A = mean(abs(delta));
end