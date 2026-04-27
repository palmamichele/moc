% enter the desired model-experiment directory 
modelname = "mnist";
nModels = 9; % 100
out_path = sprintf('%s/plots',pwd);



for type = ["union"]%["union", "train", "test"]
        
    for norm = ["E"]%["E", "T"]

        data_moc = readmatrix(sprintf('%s_data_dmoc_%s.csv', type, norm));
        for k = 0:nModels-1
            deltas_all{k+1} = readmatrix(sprintf('deltas_dmoc_%d_%s.csv', k, norm));
            data_all{k+1}   = data_moc;
            tr_all{k+1}     = readmatrix(sprintf('%s_trained_dmoc_%d_%s.csv', type, k, norm));
            un_all{k+1}     = readmatrix(sprintf('%s_untrained_dmoc_%d_%s.csv', type, k, norm));
        end
        
        nCols = ceil(sqrt(nModels));
        nRows = ceil(nModels / nCols);


        for plot_Lip =[false] %[false true]
        
            for log_plots = [true]%[false true]
                
                clf
    
                figure;
                tiledlayout(nRows, nCols);
                
                for k = 1:nModels
                    x  = deltas_all{k};
                    y1 = data_all{k};
                    y2 = tr_all{k};
                    y3 = un_all{k};
                    
                
                
                    % Check lengths
                    if length(x) ~= length(y1) || length(x) ~= length(y2) || length(x) ~= length(y3)
                        error('Model %d: vectors must all have the same length.', k);
                    end
                
                    % Log plots require positive values
                    % if any(x <= 0) || any(y1 <= 0) || any(y2 <= 0) || any(y3 <= 0)
                    %     error('Model %d: log plots require all x and y values to be positive.', k);
                    % end
                    
                
                    nexttile;
                
                    if log_plots
                        loglog(x, y1, '-o', 'DisplayName', strcat(type, ' data')); hold on;
                        loglog(x, y2, '-s', 'DisplayName', 'trained model');
                        loglog(x, y3, '-^', 'DisplayName', 'untrained model');
                
                    else 
                        plot(x, y1, '-o', 'DisplayName', strcat(type, ' data')); hold on;
                        plot(x, y2, '-s', 'DisplayName', 'trained model');
                        plot(x, y3, '-^', 'DisplayName', 'untrained model');
                
                    end
                
                    if plot_Lip
                        L_tr = max(y2(x > 0) ./ x(x > 0));
                        lip_line = L_tr * x;
                
                        if log_plots
                            loglog(x, lip_line, 'k--', ...
                            'LineWidth', 2, ...
                            'DisplayName', sprintf('trained Lip line, L = %.3g', L_tr));
                        else 
                            plot(x, lip_line, 'k--', ...
                            'LineWidth', 2, ...
                            'DisplayName', sprintf('trained Lip line, L = %.3g', L_tr));
                        end 
                
                    end
                    hold off;
                
                    xlabel('t');
                    ylabel('dmoc(t)');
                    title(sprintf('%s %d', modelname,k-1));
                    legend;
                    grid on;
                    
                    if log_plots
                        loglb = "logmocplots";
                    else
                        loglb = "mocplots";
                    end

                    if plot_Lip
                        liplb = "L";
                    else
                        liplb = "";
                    end
    
                    plot_path = sprintf('%s/%s-%s_%s_%s_%s%d.png', out_path, modelname, type, loglb, norm, liplb, nModels);
                    %exportgraphics(gcf, plot_path, 'Resolution', 300)

                   
    
                end
            end 
    
        end 
    end 

end 


