
% enter the desired model-experiment directory 
modelname = "alexnet";
Models = [10, 100, 1000];
out_path = sprintf('%s/plots',pwd);
type="union";
log_plots = true;

for norm = ["E", "T"]
    figure; % reset for each norm

    for k = Models
        x = readmatrix(sprintf('batchdeltas_dmoc_%d_%s.csv', k, norm));
        y = readmatrix(sprintf('batch%s_dmoc_%d_%s.csv', type, k, norm));

        if log_plots
            loglog(x, y, '-o', 'DisplayName', sprintf('%s k=%d', type, k)); 
        else 
            plot(x, y, '-o', 'DisplayName', sprintf('%s k=%d', type, k));
        end
        hold on;
    end

    xlabel('t');
    ylabel('dmoc(t)');
    title(sprintf('%s (%s norm)', modelname, norm));
    legend;
    grid on;
    hold off;

    plot_path = sprintf('%s/minibatch%s_%s.png', out_path, modelname, norm);
    exportgraphics(gcf, plot_path, 'Resolution', 300)
end
