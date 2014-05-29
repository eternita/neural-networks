% Here is implemented training for the following L3 neural network:
% L1 - input
% L2 - features trained with sparse auto-encoders
% L3 - softmax layer
%
%%======================================================================
%% Config & Init

clear ; close all; clc % cleanup

% file with data to train
csvFile = strcat(datasetDir, 'data/train.csv');

% configs are in separate file to easy share between train.m / test.m / predict.m
config;

fprintf(' Parameters for L2  \n');
cnn{1}

fprintf('\nLoading test data from %s  \n', csvFile);
csvdata = csvread(csvFile, 1, 0);  % don't read header with labels  
fprintf('\nLoading complete  \n');

%csvdata = csvdata(1:3000, :);

trainingLabels = csvdata(:, 1); % first column is labels
trainingLabels(trainingLabels == 0) = 10; % remap 0 -> 10 since our labels need to start from 1

m = size(csvdata, 1); % amount of training examples

% image transformation just for better visualization 
images = reshape(csvdata(:, 2:end), m, imgWidth, imgHeight); % M x width x width
images = permute(images,[1 3 2]);
images = reshape(images, m, imgWidth * imgHeight);
images = images'; % NxM (features x amount for tr. samples)

display_network(images(:, 1:100)); % display some images

%%======================================================================
%% L2 SAE training
fprintf('\nL2 SAE training ...\n');

saeL2ThetaFile = strcat(datasetDir, tempDir, 'L2_SAE_FEATURES.mat');
if exist(saeL2ThetaFile, 'file')
    % SAE1_FEATURES.mat file exists. 
    fprintf('Loading sparse auto-encoder features from %s  \n', saeL2ThetaFile);    
    load(saeL2ThetaFile);
else
    % SAE1_FEATURES.mat File does not exist. do generation
    fprintf('Cant load sparse auto-encoder features from %s  \n', saeL2ThetaFile);
    fprintf('  Do features extraction \n');
    
    %  Init theta with random parameters 
    theta = saeMatrixInit(cnn{1}.features, cnn{1}.inputVisibleSize);

    [sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   cnn{1}.inputVisibleSize, cnn{1}.features, ...
                                   saeLambda, saeSparsityParam, ...
                                   saeBeta, images), ...
                              theta, saeOptions);
                          
    save(saeL2ThetaFile, 'sae2OptTheta');
end

W = reshape(sae2OptTheta(1:cnn{1}.inputVisibleSize * cnn{1}.features), cnn{1}.features, cnn{1}.inputVisibleSize);
display_network(W');

%%======================================================================
%% L3 (Softmax) Training
fprintf('\nL3 training  ... \n')

softmaxtThetaFile = strcat(datasetDir, tempDir, 'L3_SOFTMAX_THETA.mat');
if exist(softmaxtThetaFile, 'file')
    % SOFTMAX_THETA.mat file exists. 
    fprintf('\nLoading softmax theta from %s  \n', softmaxtThetaFile);
    load(softmaxtThetaFile);
%    theta = softmaxTheta(:);  
else    
    % SOFTMAX_THETA.mat File does not exist. random initialization
    fprintf('\nCant load softmaxTheta from %s  \n', softmaxtThetaFile);
    fprintf('\n  Do random initialization for softmax theta \n');
    
    softmaxX = feedForwardAutoencoder(sae2OptTheta, cnn{1}.features, cnn{1}.inputVisibleSize, images);
    softmaxY = trainingLabels;
    numClassesL3 = numClasses;

    theta = 0.005 * randn(numClassesL3 * inputSizeL3, 1);

    [theta, cost] = minFunc( @(p) softmaxCost(p, ...
                                       numClassesL3, inputSizeL3, softmaxLambda, ...
                                       softmaxX, softmaxY), ...                                   
                                  theta, softmaxOptions);  

    softmaxTheta = reshape(theta, numClassesL3, inputSizeL3);
    save(softmaxtThetaFile, 'softmaxTheta');
end

%%======================================================================
%% Done

fprintf('\nTraining complete  ... \n')
