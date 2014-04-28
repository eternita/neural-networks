% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

datasetDir = 'C:/Develop/_n4j-nn-data/dataset-30_400_200_x31/'; % dataset root dir

imageDir = strcat(datasetDir, 'img_grayscale/');
tempDir = 'temp/'; % for prediction export
mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist

maxTestSamples = 60; % if test set is large - create subset 
maxTopPredictions = 3;

imgW = 400; % image width
imgH = 200; % image height

patchSize = 8; % patch size/dimention for L2 feature extraction (using auto-encodes)
visibleSizeL1 = patchSize * patchSize; % number of input units for the patch
poolSize = 15; % used for pooling convolved features
convolutionsStepSize = 50;
hiddenSizeL2 = 100;     % L2 hidden layer size

addpath ../libs/         % load libs

%% ========================
% loadinng matrixes
fprintf('\nLoading L2 features (sae1OptTheta, meanPatch) from %s  \n', strcat(datasetDir, 'SAE1_FEATURES.mat'));
load(strcat(datasetDir, 'SAE1_FEATURES.mat'));
fprintf('\nLoading L3 softmaxTheta from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
load(strcat(datasetDir, 'SOFTMAX_THETA.mat'));


W = reshape(sae1OptTheta(1:visibleSizeL1 * hiddenSizeL2), hiddenSizeL2, visibleSizeL1);
b = sae1OptTheta(2*hiddenSizeL2*visibleSizeL1+1:2*hiddenSizeL2*visibleSizeL1+hiddenSizeL2);

fprintf('sae1OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatch, 2), size(meanPatch, 1));
fprintf('softmaxTheta: %u x %u \n', size(softmaxTheta, 2), size(softmaxTheta, 1));


% prediction test on training dataset 
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, 1);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tr_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on cross validation dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.cv_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on test dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'), prediction, 'precision',15); % export prediction

