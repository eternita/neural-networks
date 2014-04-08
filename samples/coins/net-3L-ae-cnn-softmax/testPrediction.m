function testPrediction(imageDir, datasetFile, imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL1, softmaxTheta, convolutionsStepSize)

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
%
%

csvdata = csvread(datasetFile);
%csvdata = csvdata(1:50, :);

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

[X] = loadImageSet(sampleId, imageDir, imgW, imgH);

%visibleSize = patchSize * patchSize;

pooledFeaturesTest = convolveAndPool(X, sae1OptTheta, hiddenSizeL1, imgH, imgW, patchSize, meanPatch, poolSize, convolutionsStepSize);


softmaxX = permute(pooledFeaturesTest, [1 3 4 2]);
inputSize = numel(pooledFeaturesTest) / numTestImages;
fprintf('\n    inputSize %u ', inputSize);

softmaxX = reshape(softmaxX, inputSize, numTestImages);

fprintf('\n    softmaxTheta size %u X %u ', size(softmaxTheta, 1), size(softmaxTheta, 2));
fprintf('\n    softmaxX size %u X %u ', size(softmaxX, 1), size(softmaxX, 2));

[pred] = softmaxPredict(softmaxTheta, softmaxX);

acc = (pred(:) == softmaxY(:));
acc = sum(acc) / size(acc, 1);
fprintf('\nAccuracy: %2.3f%%\n', acc * 100);

end