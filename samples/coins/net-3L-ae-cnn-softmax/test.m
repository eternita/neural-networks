% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

%datasetDir = 'E:/nn4coins/dataset-3_924_14_200_100_gau/';
datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-5_25_400_200/'; % dataset root dir

imageDir = strcat(datasetDir, 'img_grayscale/');

maxTestSamples = 200; % if test set is large - create subset 

imgW = 400; % image width
imgH = 200; % image height

patchSize = 6; % patch size/dimention for L2 feature extraction (using auto-encodes)
visibleSizeL1 = patchSize * patchSize; % number of input units for the patch
poolSize = 15; % used for pooling convolved features
convolutionsStepSize = 50;
hiddenSizeL2 = 600;     % L2 hidden layer size

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
%testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples);

% prediction test on cross validation dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples);

% prediction test on test dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTestSamples);
