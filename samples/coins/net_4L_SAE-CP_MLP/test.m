% Runs prediction test on TRAINING / CV / TEST datasets

clear ; close all; clc % cleanup 

%% =========== Initialization =============
% Check / setup parameters before run

datasetDir = 'C:/Develop/_n4j-nn-data/dataset-30_400_200_x7/'; % dataset root dir

imageDir = strcat(datasetDir, 'img_grayscale/');
tempDir = 'temp/'; % for prediction export

maxTestSamples = 60; % if test set is large - create subset 
maxTopPredictions = 3;

% configs are in separate file to easy share between train.m / test.m
config;

fprintf(' Parameters for L2  \n');
cnn{1}

% show matrix size transformation between layers
fprintf('\nL1 -> L2  (%u X %u X %u) -> (%u X %u X %u) / (%u -> %u) \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputChannels, cnn{1}.outputWidth, cnn{1}.outputHeight, cnn{1}.outputChannels, ...
                                        cnn{1}.inputWidth * cnn{1}.inputHeight * cnn{1}.inputChannels, cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels);
                                    
fprintf('\nL2 -> L3  %u -> %u \n', cnn{1}.outputWidth * cnn{1}.outputHeight * cnn{1}.outputChannels, inputSizeL4);

fprintf('\nL3 -> L4  %u -> %u \n', inputSizeL4, numOutputClasses);


%% ========================
% loadinng matrixes
fprintf('\nLoading L2 features (sae2OptTheta, meanPatchL2) from %s  \n', strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));
load(strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat'));

theta3File = strcat(datasetDir, tempDir, 'L3_THETA.mat');
fprintf('\nLoading L3 Theta3 from %s  \n', theta3File);
load(theta3File);

Theta3File = strcat(datasetDir, tempDir, 'L4_THETA.mat');
fprintf('\nLoading L4 Theta3 from %s  \n', Theta3File);
load(Theta3File);

W = reshape(sae2OptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
fprintf('sae2OptTheta: %u x %u meanPatch: %u x %u \n', size(W, 2), size(W, 1), size(meanPatchL2, 2), size(meanPatchL2, 1));

cnn{1}.theta = sae2OptTheta;
cnn{1}.meanPatch = meanPatchL2;

fprintf('Theta3: %u x %u \n', size(Theta3, 1), size(Theta3, 2));
fprintf('Theta4: %u x %u \n', size(Theta4, 1), size(Theta4, 2));

% prediction test on cross validation dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.cv.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.cv_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on test dataset
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tst.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples, maxTopPredictions);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tst_predict.csv'), prediction, 'precision',15); % export prediction

% prediction test on training dataset 
prediction = testPrediction(imageDir, strcat(datasetDir, 'coin.tr.csv'), cnn, Theta3, Theta4, convolutionsStepSize, maxTestSamples, 1);
dlmwrite(strcat(datasetDir, tempDir, 'coin.tr_predict.csv'), prediction, 'precision',15); % export prediction



