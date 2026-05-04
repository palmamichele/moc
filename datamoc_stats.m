% enter the moc folder
save_path = pwd;
models = ["california", "iris", "MNIST"];
out_path = pwd;

dataset_map = containers.Map;
dataset_map("california")        = "California housing";
dataset_map("iris")              = "Iris";
dataset_map("MNIST")             = "MNIST";

%to have straighforward table for LaTeX
pair_keys = ["u_tr","u_te","tr_te"];
pair_latex = containers.Map;
pair_latex("u_tr") = "{\bs w}_{X^{\mathrm{u}},f},{\bs w}_{X^{\mathrm{tr}},f}";
pair_latex("u_te") = "{\bs w}_{X^{\mathrm{u}},f},{\bs w}_{X^{\mathrm{te}},f}";
pair_latex("tr_te") = "{\bs w}_{X^{\mathrm{tr}},f},{\bs w}_{X^{\mathrm{te}},f}";

for norm = ["E","T"]

    % store results
    results = struct();

   
    for mtype = models

        folder_path = fullfile(save_path, mtype);


        moc_union = readmatrix(fullfile(folder_path, sprintf('union_data_dmoc_%s.csv', norm)));
        moc_train = readmatrix(fullfile(folder_path, sprintf('train_data_dmoc_%s.csv', norm)));
        moc_test  = readmatrix(fullfile(folder_path, sprintf('test_data_dmoc_%s.csv', norm)));


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
            'VariableNames', {'Metric/Dataset','California housing','Iris','MNIST'})];

        
        M = [M; table( ...
            "S_{\\mathrm{rel}}(" + base + ")", ...
            results.(key).("california").srel, ...
            results.(key).("iris").srel, ...
            results.(key).("MNIST").srel, ...
            'VariableNames', M.Properties.VariableNames)];

        
        M = [M; table( ...
            "r(" + base + ")", ...
            results.(key).("california").r, ...
            results.(key).("iris").r, ...
            results.(key).("MNIST").r, ...
            'VariableNames', M.Properties.VariableNames)];
    end

  
    out_csv = fullfile(out_path, sprintf("datamoc_stats_%s.csv", norm));
    writetable(M, out_csv);

    disp("done")
end

function A = Arel(moc_a, moc_b)
    delta = moc_a - moc_b;
    A = sum(abs(delta))/sum(abs(moc_a));
end

