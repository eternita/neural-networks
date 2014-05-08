function [prediction] = testPrediction(imageDir, datasetFile, imgW, imgH, Theta1, Theta2, maxTestSamples, maxTopPredictions)

%TESTPREDICTION Implements the test on specific dataset
%
%   imageDir - dir with dataset images
%   datasetFile - CSV file with dataset
%   imgW - image width
%   imgH - image height
%   Theta1 - L1 matrix
%   Theta2 - L2 matrix
%

csvdata = csvread(datasetFile);
m = size(csvdata,1); % amount of test samples
% if test dataset is huge -> shuffle and get random maxTestSamples records
if m > maxTestSamples
    shuffledOrder = randperm(m)';
    shuffled_csvdata = csvdata(shuffledOrder, :);
    csvdata = shuffled_csvdata(1:maxTestSamples, :);    
end
sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
%m = size(csvdata, 1); % amount of training examples
numTestImages = size(csvdata, 1); % amount of training examples


fprintf('\nRunning prediction test ...  ');
fprintf('\n    dataset %s ', datasetFile);
fprintf('\n    image dir %s ', imageDir);
fprintf('\n    image size %u X %u ', imgW, imgH);
fprintf('\n    %u items in dataset ', numTestImages);

[X] = loadImageSet(sampleId, imageDir, imgW, imgH);

pred = mlpPredict(Theta1, Theta2, X, maxTopPredictions);

prediction = [sampleId, pred];


% accumulate predictions over maxTopPredictions
acc = zeros(numTestImages, 1);
for i = 1:maxTopPredictions
    acc = acc + (pred(:, i) == y(:));
end

acc = sum(acc) / size(acc, 1);
fprintf('\nAccuracy: %2.3f%%\n', acc * 100);

end