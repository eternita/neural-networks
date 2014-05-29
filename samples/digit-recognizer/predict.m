% Here is implemented prediction using following L3 neural network:
% L1 - input
% L2 - features trained with sparse auto-encoders
% L3 - softmax layer
%
%%======================================================================
%% Config & Init

clear ; close all; clc % cleanup

% file with data to predict
csvFile = strcat(datasetDir, 'data/test.csv');

% configs are in separate file to easy share between train.m / test.m / predict.m
config;

fprintf(' Parameters for L2  \n');
cnn{1}

fprintf('\nLoading test data from %s  \n', csvFile);
csvdata = csvread(csvFile, 1, 0);  % don't read header with labels  
fprintf('\nLoading data complete  \n');

m = size(csvdata, 1); % amount of training examples

images = reshape(csvdata(:, :), m, imgWidth, imgHeight); % M x width x width
images = permute(images,[1 3 2]);
images = reshape(images, m, imgWidth * imgHeight);
images = images'; % NxM

display_network(images(:, 1:100)); % display some first images for visual validation

% show matrix size transformation between layers
fprintf('\nL1 -> L2  (%u X %u) / %u -> %u \n', cnn{1}.inputWidth, cnn{1}.inputHeight, cnn{1}.inputWidth * cnn{1}.inputHeight, cnn{1}.features);
                                    
fprintf('\nL2 -> L3   %u -> %u \n', cnn{1}.features, numClasses);

saeL2ThetaFile = strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat');
fprintf('\nLoading L2 features (sae2OptTheta) from %s  \n', saeL2ThetaFile);
load(saeL2ThetaFile);
    
softmaxtThetaFile = strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat');
fprintf('\nLoading L3 features (softmaxTheta) from %s  \n', softmaxtThetaFile);
load(softmaxtThetaFile);

%%======================================================================
%% Run prediction

fprintf('\nRunning prediction  \n');

l3input = feedForwardAutoencoder(sae2OptTheta, cnn{1}.features, cnn{1}.inputVisibleSize, images);

[pred] = softmaxPredict(softmaxTheta, l3input, 1);

pred(pred == 10) = 0; % remap back 10 -> 0 (see train.m)


%% ========================
% save prediction to file
fid = fopen(strcat(datasetDir, tempDir, 'prediction4.csv'),'w');
    fprintf(fid,'ImageId,Label\n');
for ii=1:m,  %% 1 : M rows
    fprintf(fid, '%u,', ii);
    fprintf(fid, '%u', pred(ii, 1));
    fprintf(fid,'\n');
end
fclose(fid);

fprintf('\nDone  \n');
