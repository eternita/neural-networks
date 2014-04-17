% Runs prediction on unlabeled data

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

%datasetDir = 'E:/nn4coins/dataset-3_924_14_200_100_gau/';
datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-247_297_400_200-mexico/'; % dataset root dir

imageDir = strcat(datasetDir, 'ci_images/'); % subdir with unlabeled images
tempDir = 'temp/'; % for prediction export
mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist

maxTopPredictions = 3;

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
%% ========================
% loading image files
imgDirFullPath = imageDir; % dir with unlabeled images
imgFiles = dir(fullfile(imgDirFullPath, '*.jpg')); % img files
m = length(imgFiles); % number of images

fprintf('Loading %u images for prediction ...\n', m);
unlabeledImagesX = zeros(imgW*imgH, m); % unlabeled images

sampleId = cell(length(imgFiles), 1);

% loop over files and load images into matrix
for idx = 1:m
    gImg = imread([imgDirFullPath imgFiles(idx).name]);
    imgV = reshape(gImg, 1, imgW*imgH); % unroll       
    unlabeledImagesX(:, idx) = imgV; 
    sampleId{idx, 1} = imgFiles(idx).name;
end

%% ========================
% run prediction
[pred] = netPredict(unlabeledImagesX, imgW, imgH, patchSize, poolSize, sae1OptTheta, meanPatch, hiddenSizeL2, softmaxTheta, convolutionsStepSize, maxTopPredictions);

%% ========================
% save prediction to file
fid = fopen(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'),'w');
for ii=1:m,  %% 1 -> 47165 rows
    fprintf(fid, '%s', imgFiles(ii).name);
    for i=1:size(pred, 2), %% 1 -> 26 columns
%        fprintf(fid, ',%u', data{ii, i}(1));
        fprintf(fid, ',%u', pred(ii, i));
    end
    fprintf(fid,'\n');
end
fclose(fid);



