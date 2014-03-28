% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

%datasetDir = 'E:/nn4coins/dataset-3_924_15_200_100_gau/';
datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-3_924_15_200_100_gau/';
imageDir = strcat(datasetDir, 'img/');

imgW = 200; % image width
imgH = 100; % image height

%% ========================
% loadinng matrixes
fprintf('\nLoading Thetta1 from %s  \n', strcat(datasetDir, 'THETTA1.mat'));
load(strcat(datasetDir, 'THETTA1.mat'));
fprintf('\nLoading Thetta2 from %s  \n', strcat(datasetDir, 'THETTA2.mat'));
load(strcat(datasetDir, 'THETTA2.mat'));


% prediction test on training dataset (make sure training set is not huge - create subset otherwise)
testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), imgW, imgH, Theta1, Theta2);

% prediction test on cross validation dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), imgW, imgH, Theta1, Theta2);

% prediction test on test dataset
testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), imgW, imgH, Theta1, Theta2);
