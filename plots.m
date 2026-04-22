%enter the desired model-experiment directory 

% Load the CSV files
type= "train";
data_moc = readmatrix(sprintf('%s_data_dmoc.csv', type));
%deltas = readmatrix('train_deltas_dmoc_0.csv');
%plot(deltas, data_moc, '-o', 'DisplayName', 'train data');

nModels=1; %100

for k=0:nModels-1
    deltas_all{k+1} = readmatrix(sprintf('%s_deltas_dmoc_%d.csv', type, k));
    data_all{k+1} = data_moc;
    tr_all{k+1} = readmatrix(sprintf('%s_trained_dmoc_%d.csv', type, k));
    un_all{k+1} = readmatrix(sprintf('%s_untrained_dmoc_%d.csv', type,k));
end 

nCols = ceil(sqrt(nModels));
nRows = ceil(nModels / nCols);
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

    nexttile;
    plot(x, y1, '-o', 'DisplayName', strcat(type,' data')); hold on;
    plot(x, y2, '-s', 'DisplayName', 'trained model');
    plot(x, y3, '-^', 'DisplayName', 'untrained model');
    hold off;

    xlabel('t');
    ylabel('dmoc(t)');
    title(sprintf('Model %d', k));
    legend;
    grid on;
end