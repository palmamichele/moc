%enter each experiments folder containing initializations for same net
files = dir(pwd);
modelname="linear-california"

n_experiments = 0;
for j = 1:length(files)
    if ~files(j).isdir
        if ~isempty(regexp(files(j).name, '^model_.+', 'once'))
            n_experiments = n_experiments + 1;
        end
    end
end
    
for norm = ["E","T" ]

    for type = ["union", "train", "test"]
        deltas_all = cell(n_experiments,1);
        tr_all     = cell(n_experiments,1);
        un_all     = cell(n_experiments,1);

        for k = 0:n_experiments-1
            %load all net mocs (both trained, untrained)
            deltas_all{k+1} = readmatrix(sprintf('deltas_dmoc_%d_%s.csv', k, norm));
            tr_all{k+1}     = readmatrix(sprintf('%s_trained_dmoc_%d_%s.csv', type, k, norm));
            un_all{k+1}     = readmatrix(sprintf('%s_untrained_dmoc_%d_%s.csv', type, k, norm));
        end 
        %extract grid (same for all experiments)
        tgrid = deltas_all{1}(:,1);
        
        nT = length(tgrid);
        omega_tr  = zeros(n_experiments, nT);
        omega_un  = zeros(n_experiments, nT);

        for k = 1:n_experiments

            tk = deltas_all{k}(:,1);
            if length(tk) ~= nT || max(abs(tk - tgrid)) > 1e-12
                error("Grid mismatch in experiment %d", k-1);
            end

            if length(tr_all{k}) ~= nT
                error("Mismatch between grid and modulus length at experiment %d", k-1);
            end

            % trained
            omega_tr(k,:) = tr_all{k}(:)';

            % untrained
            omega_un(k,:) = un_all{k}(:)';

        end
        %stat across runs 
        mean_tr = mean(omega_tr, 1);
        std_tr  = std(omega_tr, 0, 1);

        mean_un = mean(omega_un, 1);
        std_un  = std(omega_un, 0, 1);

        out_tr = [tgrid, mean_tr', std_tr'];
        out_un = [tgrid, mean_un', std_un'];


        targetN = 50;
        idx_mid = round(linspace(2, nT-1, targetN-2));
        idx = [1, idx_mid, nT];
        
        out_tr = [tgrid(idx), mean_tr(idx)', std_tr(idx)'];
        out_un = [tgrid(idx), mean_un(idx)', std_un(idx)'];

        writematrix(out_tr, sprintf('%s_stats_%s_trained_%s_%s.txt', modelname, type, norm), 'Delimiter', 'space');
        writematrix(out_un, sprintf('%s_stats_%s_untrained_%s_%s.txt', modelname, type, norm), 'Delimiter', 'space');

        disp("done type = " + type + ", norm = " + norm);

        %plot each model using same filtered tgrid 
        
        writematrix([tgrid(idx), (idx)'], sprintf('%s_stats_%s_trained_%s_%s.txt', modelname, type, norm), 'Delimiter', 'space');

    end


end