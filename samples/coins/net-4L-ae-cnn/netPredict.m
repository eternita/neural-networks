function [prediction] = netPredict(X, imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTopPredictions)

%NETPREDICT Implements the test on specific dataset
%
%   X - images  N x M
%   imgW - image width
%   imgH - image height
%   patchSize
%   poolSize
%   sae1OptTheta
%   meanPatch
%   hiddenSizeL2
%   softmaxTheta
%   convolutionsStepSize
%   maxTopPredictions
%
%   prediction - output prediction with size M x maxTopPredictions


numTestImages = size(X, 2); % amount of training examples

pooledFeaturesTest = convolveAndPool(X, sae1OptTheta, hiddenSizeL2, imgH, imgW, patchSize, meanPatch, poolSize, convolutionsStepSize);


softmaxX = permute(pooledFeaturesTest, [1 3 4 2]);
inputSize = numel(pooledFeaturesTest) / numTestImages;
fprintf('\n    inputSize %u ', inputSize);

softmaxX = reshape(softmaxX, inputSize, numTestImages);

fprintf('\n    softmaxTheta size %u X %u ', size(softmaxTheta, 1), size(softmaxTheta, 2));
fprintf('\n    softmaxX size %u X %u \n', size(softmaxX, 1), size(softmaxX, 2));

%softmaxY(:) % labels

[prediction] = softmaxPredict(softmaxTheta, softmaxX, maxTopPredictions);


%% -----------------------------------------------------
% show (in)correct prediction value and amount of correct samples
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