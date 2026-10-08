% enter the moc experiment folder
save_path = pwd;
models = ["california", "iris", "MNIST", "imagenet"];
out_path = pwd;

dataset_map = containers.Map;
dataset_map("california")        = "California housing";
dataset_map("iris")              = "Iris";
dataset_map("MNIST")             = "MNIST";
dataset_map("imagenet")             = "ImageNet";

%to have straighforward table for LaTeX
pair_keys = ["u_tr","u_te","tr_te"];
pair_latex = containers.Map;
pair_latex("u_tr") = "{\bs w}_{X^{\mathrm{u}},f},{\bs w}_{X^{\mathrm{tr}},f}";
pair_latex("u_te") = "{\bs w}_{X^{\mathrm{u}},f},{\bs w}_{X^{\mathrm{te}},f}";
pair_latex("tr_te") = "{\bs w}_{X^{\mathrm{tr}},f},{\bs w}_{X^{\mathrm{te}},f}";

for norm = ["E"]%["E","T"]

    % store results
    results = struct();

    for mtype = models

        folder_path = fullfile(save_path, mtype);

        if mtype=="imagenet"
            moc_union = readmatrix(fullfile(folder_path, sprintf('data_batchunion_dmoc_1000_%s.csv', norm)));
            moc_train = readmatrix(fullfile(folder_path, sprintf('data_batchtrain_dmoc_1000_%s.csv', norm)));
            moc_test  = readmatrix(fullfile(folder_path, sprintf('data_batchtrain_dmoc_1000_%s.csv', norm)));
        else
            moc_union = readmatrix(fullfile(folder_path, sprintf('union_data_dmoc_%s.csv', norm)));
            moc_train = readmatrix(fullfile(folder_path, sprintf('train_data_dmoc_%s.csv', norm)));
            moc_test  = readmatrix(fullfile(folder_path, sprintf('test_data_dmoc_%s.csv', norm)));
        end 

        pairs = {
            moc_union, moc_train, "u_tr";
            moc_union, moc_test , "u_te";
            moc_train, moc_test , "tr_te";
        };

      
        for p = 1:size(pairs,1)
            A = pairs{p,1};
            B = pairs{p,2};
            key = pairs{p,3};

            results.(key).(mtype).arel = Arel(A,B);
            results.(key).(mtype).srel = 1 - results.(key).(mtype).arel;
            results.(key).(mtype).r    = corr(A(:),B(:),'Type','Pearson');
        end
    end

   
    M = table();

    for k = 1:length(pair_keys)
        key = pair_keys(k);
        base = pair_latex(key);

       
        M = [M; table( ...
            "A_{\\mathrm{rel}}(" + base + ")", ...
            results.(key).("california").arel, ...
            results.(key).("iris").arel, ...
            results.(key).("MNIST").arel, ...
            results.(key).("imagenet").arel, ...
            'VariableNames', {'Metric/Dataset','California housing','Iris','MNIST', 'ImageNet'})];

        
        M = [M; table( ...
            "S_{\\mathrm{rel}}(" + base + ")", ...
            results.(key).("california").srel, ...
            results.(key).("iris").srel, ...
            results.(key).("MNIST").srel, ...
            results.(key).("imagenet").srel, ...
            'VariableNames', M.Properties.VariableNames)];

        
        M = [M; table( ...
            "r(" + base + ")", ...
            results.(key).("california").r, ...
            results.(key).("iris").r, ...
            results.(key).("MNIST").r, ...
            results.(key).("imagenet").r, ...
            'VariableNames', M.Properties.VariableNames)];
    end

  
    out_csv = fullfile(out_path, sprintf("datamoc_stats_%s.csv", norm));
    writetable(M, out_csv);

    disp("done")
end

%

norms  = ["E"];
states = ["data","trained","untrained"];

split_pairs = {
    "union","union";
    "union","train";
};

state_pairs = {
    "data","trained";
    "data","untrained";
};

k_index = @(k) k + 1;

dataset_map = containers.Map;
dataset_map("california") = "California housing";
dataset_map("iris")       = "Iris";
dataset_map("MNIST")      = "MNIST";
dataset_map("imagenet")   = "ImageNet";
dataset_map("linear-california")   = "LinearCal";

for norm = norms

for mtype = models

    folder_path = fullfile(save_path, mtype);

    if mtype == "imagenet"
        K = [10,100,1000];
        maxK = max(K);
    elseif mtype=="linear-california"
        K = [0];
        maxK = 0;
    else 
        K = [0,1,2,3,4,5,6,7,8]; %set to 0 for linear-california model 
        maxK = max(K);
    end

    moc = struct();

    for s = states
        for sp = ["union","train","test"]
            moc.(s).(sp) = cell(maxK+1,1);
        end
    end
    for k = K
        idx = k_index(k);

        if mtype == "imagenet"

            moc.data.union{idx} = readmatrix(fullfile(folder_path, sprintf('data_batchunion_dmoc_%d_%s.csv', k,norm)));
            moc.data.train{idx} = readmatrix(fullfile(folder_path, sprintf('data_batchtrain_dmoc_%d_%s.csv', k,norm)));
            moc.data.test{idx}  = readmatrix(fullfile(folder_path, sprintf('data_batchtest_dmoc_%d_%s.csv', k,norm)));

            moc.trained.union{idx} = readmatrix(fullfile(folder_path, sprintf('trained_batchunion_dmoc_%d_%s.csv', k,norm)));
            moc.trained.train{idx} = readmatrix(fullfile(folder_path, sprintf('trained_batchtrain_dmoc_%d_%s.csv', k,norm)));
            moc.trained.test{idx}  = readmatrix(fullfile(folder_path, sprintf('trained_batchtest_dmoc_%d_%s.csv', k,norm)));

            moc.untrained.union{idx} = readmatrix(fullfile(folder_path, sprintf('untrained_batchunion_dmoc_%d_%s.csv', k,norm)));
            moc.untrained.train{idx} = readmatrix(fullfile(folder_path, sprintf('untrained_batchtrain_dmoc_%d_%s.csv', k,norm)));
            moc.untrained.test{idx}  = readmatrix(fullfile(folder_path, sprintf('untrained_batchtest_dmoc_%d_%s.csv', k,norm)));

        else

            % baseline (data)
            moc.data.union{idx} = readmatrix(fullfile(folder_path, sprintf('union_data_dmoc_%s.csv', norm)));
            moc.data.train{idx} = readmatrix(fullfile(folder_path, sprintf('train_data_dmoc_%s.csv', norm)));
            moc.data.test{idx}  = readmatrix(fullfile(folder_path, sprintf('test_data_dmoc_%s.csv', norm)));

            % trained
            moc.trained.union{idx} = readmatrix(fullfile(folder_path, sprintf('union_trained_dmoc_%d_%s.csv', k,norm)));
            moc.trained.train{idx} = readmatrix(fullfile(folder_path, sprintf('train_trained_dmoc_%d_%s.csv', k,norm)));
            moc.trained.test{idx}  = readmatrix(fullfile(folder_path, sprintf('test_trained_dmoc_%d_%s.csv', k,norm)));

            % untrained
            moc.untrained.union{idx} = readmatrix(fullfile(folder_path, sprintf('union_untrained_dmoc_%d_%s.csv', k,norm)));
            moc.untrained.train{idx} = readmatrix(fullfile(folder_path, sprintf('train_untrained_dmoc_%d_%s.csv', k,norm)));
            moc.untrained.test{idx}  = readmatrix(fullfile(folder_path, sprintf('test_untrained_dmoc_%d_%s.csv', k,norm)));

        end
    end

   
    results = struct();

 
   for k = K
        idx = k_index(k);

        for sp = 1:size(split_pairs,1)

            sp1 = split_pairs{sp,1};
            sp2 = split_pairs{sp,2};

            for p = 1:size(state_pairs,1)

                s1 = state_pairs{p,1};
                s2 = state_pairs{p,2};

                key_base = s1 + "_" + s2 + "__" + sp1 + "_" + sp2;

                A = moc.(s1).(sp1){idx};
                B = moc.(s2).(sp2){idx};

                if isempty(A) || isempty(B)
                    continue;
                end

                arel = Arel(A,B);

                results.("Arel_" + key_base)(idx) = arel;
                results.("Srel_" + key_base)(idx) = 1 - arel;
                results.("r_"    + key_base)(idx) = corr(A(:),B(:),'Type','Pearson');

            end
        end
    end

   
    row_names = fieldnames(results);

    M = table();

    for i = 1:length(row_names)

        name = row_names{i};

        row = NaN(1,length(K));

        for j = 1:length(K)
            kk = k_index(K(j));
            row(j) = results.(name)(kk);
        end

        M = [M;
            array2table(row, ...
            'VariableNames', "Model_" + string(K))];

    end

    M.Properties.RowNames = row_names;

   
    out_csv = fullfile(out_path, sprintf("dmoc_stats_%s_%s.csv", mtype, norm));
    writetable(M, out_csv, 'WriteRowNames', true);

    disp("done: " + mtype + " " + norm);

end
end


function A = Arel(moc_a, moc_b)
    delta = moc_a - moc_b;
    A = sum(abs(delta)) / sum(abs(moc_a));
end
