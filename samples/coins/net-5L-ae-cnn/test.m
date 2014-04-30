% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

datasetDir = 'C:/Develop/_n4j-nn-data/dataset-30_400_200_x7/'; % dataset root dir

imageDir = strcat(datasetDir, 'img_grayscale/');
tempDir = 'temp/'; % for prediction export

maxTestSamples = 60; % if test set is large - create subset 
maxTopPredictions = 3;

imgW = 400; % image width
imgH = 200; % image height

cnn = cell(2, 1);

% L2
cnn{1}.inputWidth = imgW;
cnn{1}.inputHeight = imgH;
cnn{1}.inputChannels = 1;
cnn{1}.features = 100;
cnn{1}.patchSize = 6;
cnn{1}.poolSize = 5;
cnn{1}.numPatches = 10000;
cnn{1}.inputVisibleSize = cnn{1}.patchSize * cnn{1}.patchSize * cnn{1}.inputChannels;

cnn{1}.outputWidth = floor((cnn{1}.inputWidth - cnn{1}.patchSize + 1) / cnn{1}.poolSize);
cnn{1}.outputHeight = floor((cnn{1}.inputHeight - cnn{1}.patchSize + 1) / cnn{1}.poolSize);
cnn{1}.outputChannels = cnn{1}.features;
cnn{1}.outputSize = cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels;

% L3
cnn{2}.inputWidth = cnn{1}.outputWidth;
cnn{2}.inputHeight = cnn{1}.outputHeight;
cnn{2}.inputChannels = cnn{1}.outputChannels;
cnn{2}.features = 200;
cnn{2}.patchSize = 3;
cnn{2}.poolSize = 3;
cnn{2}.numPatches = 10000;
cnn{2}.inputVisibleSize = cnn{2}.patchSize * cnn{2}.patchSize * cnn{2}.inputChannels;

cnn{2}.outputWidth = floor((cnn{2}.inputWidth - cnn{2}.patchSize + 1) / cnn{2}.poolSize);
cnn{2}.outputHeight = floor((cnn{2}.inputHeight - cnn{2}.patchSize + 1) / cnn{2}.poolSize);
cnn{2}.outputChannels = cnn{2}.features;
cnn{2}.outputSize = cnn{2}.outputWidth * cnn{2}.outputHeight * cnn{2}.outputChannels;

convolutionsStepSize = 50;

addpath ../libs/         % load libs

%% ========================
% loadinng matrixes
fprintf('\nLoading L2 features (sae2OptTheta, meanPatchL2) from %s  \n', strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));
load(strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));
fprintf('\nLoading L3 features (sae3OptTheta, meanPatchL3) from %s  \n', strcat(datasetDir, tempDir, 'L3_SAE_FEATURES.mat'));
load(strcat(datasetDir, tempDir, 'L3_SAE_FEATURES.mat'));
fprintf('\nLoading L4 softmaxTheta from %s  \n', strcat(datasetDir, tempDir, 'L4_SOFTMAX_THETA.mat'));
load(strcat(datasetDir, tempDir, 'L4_SOFTMAX_THETA.mat'));

W = reshape(sae2OptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
fprintf('sae2OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatchL2, 2), size(meanPatchL2, 1));

W = reshape(sae3OptTheta(1 : cnn{2}.inputVisibleSize * cnn{2}.features), cnn{2}.features, cnn{2}.inputVisibleSize);
fprintf('sae3OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatchL3, 2), size(meanPatchL3, 1));

cnn{1}.theta = sae2OptTheta;
cnn{1}.meanPatch = meanPatchL2;
cnn{2}.theta = sae3OptTheta;
cnn{2}.meanPatch = meanPatchL3;

fprintf('softmaxTheta: %u x %u \n', size(softmaxTheta, 2), size(softmaxTheta, 1));


% prediction test on training dataset 
%prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, patchSizeL2, poolSizeL2, sae2OptTheta, meanPatchL2, sae3OptTheta, meanPatchL3, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, 1);
%dlmwrite(strcat(datasetDir, tempDir, 'coin.tr_predict.csv'), prediction, 'precision',15); % export prediction

prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), cnn, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);

% prediction test on cross validation dataset
%prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, patchSizeL2, poolSizeL2, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
%dlmwrite(strcat(datasetDir, tempDir, 'coin.cv_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on test dataset
%prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, patchSizeL2, poolSizeL2, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
%dlmwrite(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'), prediction, 'precision',15); % export prediction

