function pooledFeaturesTrain = convolveAndPool(shuffledX, sae1OptTheta, hiddenSizeL1, img_h, img_w, img_channels, patchDim, meanPatch, poolDim, convolutionsStepSize)
%convolveAndPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%

visibleSize = patchDim * patchDim * img_channels;

W = reshape(sae1OptTheta(1:visibleSize * hiddenSizeL1), hiddenSizeL1, visibleSize);
b = sae1OptTheta(2*hiddenSizeL1*visibleSize+1:2*hiddenSizeL1*visibleSize+hiddenSizeL1);

trainImages = zeros(img_h, img_w, img_channels, size(shuffledX, 2));
%convImages(:, :, 1, :) = reshape(shuffledX(:, 1:8), img_h, img_w, 8);
trainImages(:, :, :, :) = reshape(shuffledX, img_h, img_w, img_channels, size(shuffledX, 2));

numTrainImages = size(shuffledX, 2);

pooledFeaturesTrain = zeros(hiddenSizeL1, numTrainImages, ...
    floor((img_h - patchDim + 1) / poolDim), ...
    floor((img_w - patchDim + 1) / poolDim) );

%tic();
fprintf('Convolving and pooling train images. (%u features are used) \n', hiddenSizeL1);

for convPart = 1:(hiddenSizeL1 / convolutionsStepSize)

    featureStart = (convPart - 1) * convolutionsStepSize + 1;
    featureEnd = convPart * convolutionsStepSize;

    fprintf('Step %d: Convolving and pooling features %d to %d\n', convPart, featureStart, featureEnd);
    Wt = W(featureStart:featureEnd, :);
    bt = b(featureStart:featureEnd);

%    fprintf('Convolving and pooling train images\n');
    convolvedFeaturesThis = convolve(patchDim, convolutionsStepSize, ...
        trainImages, Wt, bt, meanPatch);
    pooledFeaturesThis = pool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTrain(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;
%    toc();
    clear convolvedFeaturesThis pooledFeaturesThis;

end

end

