function [prediction] = testPrediction(imageDir, datasetFile, imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions)

%TESTPREDICTION Implements the test on specific dataset
%
%   imageDir - dir with dataset images
%   datasetFile - CSV file with dataset
%   imgW - image width
%   imgH - image height
%   patchSize
%   poolSize
%   sae1OptTheta
%   meanPatch
%   hiddenSizeL1
%   softmaxTheta
%   maxTestSamples
%   maxTopPredictions

csvdata = csvread(datasetFile);

m = size(csvdata,1); % amount of test samples


% if test dataset is huge -> shuffle and get random maxTestSamples records
if m > maxTestSamples
    shuffledOrder = randperm(m)';
    shuffled_csvdata = csvdata(shuffledOrder, :);
    csvdata = shuffled_csvdata(1:maxTestSamples, :);    
end

sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
softmaxY = csvdata(:, 2); % second column is coinIdx
numTestImages = size(csvdata, 1); % amount of training examples

fprintf('\nRunning prediction test ...  ');
fprintf('\n    dataset %s ', datasetFile);
fprintf('\n    image dir %s ', imageDir);
fprintf('\n    image size %u X %u ', imgW, imgH);
fprintf('\n    %u items in dataset ', numTestImages);
fprintf('\n    sae1OptTheta size %u X %u ', size(sae1OptTheta, 1), size(sae1OptTheta, 2));
fprintf('\n    meanPatch size %u X %u ', size(meanPatch, 1), size(meanPatch, 2));

[X] = loadImageSet(sampleId, imageDir, imgW, imgH); % images

[pred] = netPredict(X, imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTopPredictions);

prediction = [sampleId, pred];


% accumulate predictions over maxTopPredictions
acc = zeros(numTestImages, 1);
for i = 1:maxTopPredictions
    acc = acc + (pred(:, i) == softmaxY(:));
end

acc = sum(acc) / size(acc, 1);
fprintf('\nAccuracy: %2.3f%%\n', acc * 100);

%% -----------------------------------------------------
% show correct prediction value and amount of correct samples
%{
i = double(pred(:) == softmaxY(:));
yind = find(i == 1); % index where prediction correct

vals = softmaxY(yind)'; % value for correct prediction

uniqval = unique(vals); % unique values for correct prediction

%uniqvalamnt = size(uniqval, 1);

% show correct prediction value and amount of correct samples
for i = 1:size(uniqval, 2)
    v = uniqval(i);
    v = repmat(v, size(vals, 1), 1);
    amnt = sum(double(vals == v));
    fprintf('\n  %u - %u ', uniqval(i), amnt);
end
%}
%-----------------------------------------------------

end