%%  shuffle CSV dataset (useful for mini-batch) 
function shuffleTrainingSet(datasetDir, srcFileName, destFileName)
% datasetDir = 'C:/Develop/src/pavlikovskiy/chn/data/dataset-100_3276_468_400_200_grayscale-cnn2/';
% srcFileName = 'coin.tr.csv';
% destFileName = 'coin.tr.shuffled.csv';

    csvdata = csvread(strcat(datasetDir, srcFileName));

    shuffledOrder = randperm(size(csvdata,1))';
    shuffled_csvdata = csvdata(shuffledOrder, :);

    dlmwrite(strcat(datasetDir, destFileName),shuffled_csvdata, 'precision',15);

end
