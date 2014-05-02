function [prediction] = netPredict(X, cnn, softmaxTheta, convolutionsStepSize, maxTopPredictions)

%NETPREDICT Implements the test on specific dataset
%
%   X - images  N x M
%   cnn
%   softmaxTheta
%   convolutionsStepSize
%   maxTopPredictions
%
%   prediction - output prediction with size M x maxTopPredictions


numTestImages = size(X, 2); % amount of training examples

fprintf('\nL2  (%u X %u X %u) -> (%u X %u X %u) \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputChannels, cnn{1}.outputWidth, cnn{1}.outputHeight, cnn{1}.outputChannels);
cpFeaturesL2 = convolveAndPool(X, cnn{1}.theta, cnn{1}.features, ...
                cnn{1}.inputHeight, cnn{1}.inputWidth, cnn{1}.inputChannels, ...
                cnn{1}.patchSize, cnn{1}.meanPatch, cnn{1}.poolSize, convolutionsStepSize);

outL2 = permute(cpFeaturesL2, [4 3 1 2]);
X = reshape(outL2, cnn{1}.outputSize, numTestImages);


fprintf('\nL3  (%u X %u X %u) -> (%u X %u X %u) \n', cnn{2}.inputWidth, cnn{2}.inputHeight, cnn{2}.inputChannels, cnn{2}.outputWidth, cnn{2}.outputHeight, cnn{2}.outputChannels);
cpFeaturesL3 = convolveAndPool(X, cnn{2}.theta, cnn{2}.features, ...
                cnn{2}.inputHeight, cnn{2}.inputWidth, cnn{2}.inputChannels, ...
                cnn{2}.patchSize, cnn{2}.meanPatch, cnn{2}.poolSize, convolutionsStepSize);

X = permute(cpFeaturesL3, [4 3 1 2]); % W x H x Ch x tr_num

inputSizeL4 = cnn{2}.outputSize; 
fprintf('\n    inputSize %u ', inputSizeL4);

X = reshape(X, inputSizeL4, numTestImages);

fprintf('\nL4  %u -> %u \n', size(softmaxTheta, 2), size(softmaxTheta, 1));
%fprintf('\n    softmaxX size %u X %u \n', size(X, 1), size(X, 2));

[prediction] = softmaxPredict(softmaxTheta, X, maxTopPredictions);


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