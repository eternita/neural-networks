% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

datasetDir = 'C:/Develop/_n4j-nn-data/dataset-30_400_200_x7/'; % dataset root dir
tempDir = 'temp/'; % for prediction export

imageDir = strcat(datasetDir, 'img_gau/');

maxTestSamples = 60; % if test set is large - create subset 
maxTopPredictions = 3;

imgW = 400; % image width
imgH = 200; % image height

addpath ../libs/         % load libs

%% ========================
% loadinng matrixes
theta1File = strcat(datasetDir, tempDir, 'THETA1.mat');
fprintf('\nLoading Thetta1 from %s  \n', theta1File);
load(theta1File);

theta2File = strcat(datasetDir, tempDir, 'THETA2.mat');
fprintf('\nLoading Thetta2 from %s  \n', theta2File);
load(theta2File);
fprintf('Theta1: %u x %u \n', size(Theta1, 1), size(Theta1, 2));
fprintf('Theta2: %u x %u \n', size(Theta2, 1), size(Theta2, 2));


% prediction test on training dataset (make sure training set is not huge - create subset otherwise)
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, Theta1, Theta2, maxTestSamples, 1);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tr_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on cross validation dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, Theta1, Theta2, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.cv_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on test dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, Theta1, Theta2, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'), prediction, 'precision',15); % export prediction
