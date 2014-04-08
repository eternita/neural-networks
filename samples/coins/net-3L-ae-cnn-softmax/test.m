% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

%datasetDir = 'E:/nn4coins/dataset-3_924_14_200_100_gau/';
datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-5_50_25_400_200_grayscale-cnn/';
%datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-100_936_468_400_200_grayscale-cnn/';

imageDir = strcat(datasetDir, 'img/');

imgW = 400; % image width
imgH = 200; % image height

patchSize = 6;
poolDim = 15;
convolutionsStepSize = 50;


addpath ../libs/         % load libs

%% ========================
% loadinng matrixes
fprintf('\nLoading L2 features (sae1OptTheta, meanPatch) from %s  \n', strcat(datasetDir, 'SAE1_FEATURES.mat'));
load(strcat(datasetDir, 'SAE1_FEATURES.mat'));
fprintf('\nLoading L3 softmaxTheta from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
load(strcat(datasetDir, 'SOFTMAX_THETA.mat'));

visibleSize = patchSize * patchSize;
hiddenSizeL1 = 600;     % L1 hidden layer size

W = reshape(sae1OptTheta(1:visibleSize * hiddenSizeL1), hiddenSizeL1, visibleSize);
b = sae1OptTheta(2*hiddenSizeL1*visibleSize+1:2*hiddenSizeL1*visibleSize+hiddenSizeL1);

fprintf('sae1OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatch, 2), size(meanPatch, 1));
fprintf('softmaxTheta: %u x %u \n', size(softmaxTheta, 2), size(softmaxTheta, 1));


% prediction test on training dataset (make sure training set is not huge - create subset otherwise)
%testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, patchSize, poolDim, sae1OptTheta, meanPatch, hiddenSizeL1, softmaxTheta, convolutionsStepSize);

% prediction test on cross validation dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, patchSize, poolDim, sae1OptTheta, meanPatch, hiddenSizeL1, softmaxTheta, convolutionsStepSize);

% prediction test on test dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, patchSize, poolDim, sae1OptTheta, meanPatch, hiddenSizeL1, softmaxTheta, convolutionsStepSize);
