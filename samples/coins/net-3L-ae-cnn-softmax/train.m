clear ; close all; clc % cleanup
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
img_w = 400; % image width
img_h = 200; % image height

datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-100_3276_468_400_200_grayscale-cnn2/';
%datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-100_468_468_400_200_grayscale-cnn/';
unlabeledImgDir = 'img_unlabeled/'; % sub directory with images for auto-encoder training (unlabeled/for unsupervised feature extraction)
imgDir = 'img_labeled/'; % sub directory with images
tempDir = 'temp/'; % for pooled features used with mini batch

% !! when change batchSize - clean up temp
batchSize = 30; % batch size for L3 mini-batch algorithm
trainingIterationCount = 2000; % L3 amount of iterations over whole training set
numClassesL3 = 100; % amount of output lables, classes (e.g. coins)

hiddenSizeL2 = 600;     % L2 hidden layer size

patchSize = 6; % patch size/dimention for L2 feature extraction (using auto-encodes)
numPatches = 1000; % amount of patches for auto-encoder training

poolSize = 15; % used for pooling convolved features

visibleSizeL1 = patchSize * patchSize; % number of input units for the patch

%amountOfImagesForPatchGeneration = 500;


sparsityParam = 0.01;   % 0.01 desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.003;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       


addpath ../libs/         % load libs
addpath ../libs/minFunc/

convolutionsStepSize = 50;

softmaxLambda = 1e-4; % weight decay for L3


%  Use minFunc to minimize cost functions
options.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
options.maxIter = 800;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

softmaxOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.s
softmaxOptions.maxIter = 1; % update minFunc confugs for mini batch 
softmaxOptions.display = 'on';



%% Visualize some full size images
% visualize some full size images
csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    
visualAmount = 3^2;
fprintf('Visualize %u full size images ...\n', visualAmount);
[previewX] = loadImageSet(csvdata(1:visualAmount, 1), strcat(datasetDir, imgDir), img_w, img_h);
fullSizeImages = zeros(img_w^2, visualAmount);
for i = 1:visualAmount
    % visualization works for squared matrixes
    % before visualization convert img_h x img_w -> img_w * img_w
    fullSizeImages(:, i) = resizeImage2Square(previewX(:, i), img_w, img_h);
end;

display_network(fullSizeImages);

clear previewX fullSizeImages;

%pause;
%}
%%======================================================================

%% STEP : Patches for auto-encoders training

fprintf('Auto-encoders training ...\n')

if exist(strcat(datasetDir, 'PATCHES.mat'), 'file')
    % PATCHES.mat file exists. 
    fprintf('Loading patches for sparse auto-encoder training from %s  \n', strcat(datasetDir, 'PATCHES.mat'));
    load(strcat(datasetDir, 'PATCHES.mat'));
else
    % PATCHES.mat File does not exist. do generation
    fprintf('Cant load patches for sparse auto-encoder training from %s  \n', strcat(datasetDir, 'PATCHES.mat'));
    fprintf('  Do patch geenration \n');
    
    unlabeledImgDirFullPath = strcat(datasetDir, unlabeledImgDir); % dir with unlabeled images
    unlabeledImgFiles = dir(fullfile(unlabeledImgDirFullPath, '*.jpg')); % img files
    fprintf('Loading %u random images for patches ...\n', length(unlabeledImgFiles));
    unlabeledImagesX = zeros(img_w*img_h, length(unlabeledImgFiles)); % unlabeled images
    % loop over files and load images into matrix
    for idx = 1:length(unlabeledImgFiles)
        gImg = imread([unlabeledImgDirFullPath unlabeledImgFiles(idx).name]);
        imgV = reshape(gImg, 1, img_w*img_h); % unroll       
        unlabeledImagesX(:, idx) = imgV; 
    end
    
    fprintf('Generating %u patches (%u x %u) from images ...\n', numPatches, patchSize, patchSize);
    [patches, meanPatch] = getPatches(unlabeledImagesX, img_w, img_h, patchSize, numPatches);

    % remove (clean up some memory)
    clear shuffledX

    save(strcat(datasetDir, 'PATCHES.mat'), 'patches');
    display_network(patches(:,randi(size(patches,2),200,1)));
    fprintf('Patches generation complete ...\n');
%    pause;
end



%display_network(patches(:,randi(size(patches,2),200,1)));
%pause;

%%======================================================================
%% STEP : Learning L2 features with sparse autoencoder 


if exist(strcat(datasetDir, 'SAE1_FEATURES.mat'), 'file')
    % SAE1_FEATURES.mat file exists. 
    fprintf('Loading sparse auto-encoder features from %s  \n', strcat(datasetDir, 'SAE1_FEATURES.mat'));    
    load(strcat(datasetDir, 'SAE1_FEATURES.mat'));
else
    % SAE1_FEATURES.mat File does not exist. do generation
    fprintf('Cant load sparse auto-encoder features from %s  \n', strcat(datasetDir, 'SAE1_FEATURES.mat'));
    fprintf('  Do features extraction \n');
    
    %  Obtain random parameters theta
    theta = initializeParameters(hiddenSizeL2, visibleSizeL1);


    [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                       visibleSizeL1, hiddenSizeL2, ...
                                       lambda, sparsityParam, ...
                                       beta, patches), ...
                                  theta, options);

    save(strcat(datasetDir, 'SAE1_FEATURES.mat'), 'sae1OptTheta', 'meanPatch');
end

% Visualization Sparser Autoencoder Features to see that the features look good
W = reshape(sae1OptTheta(1:visibleSizeL1 * hiddenSizeL2), hiddenSizeL2, visibleSizeL1);
b = sae1OptTheta(2*hiddenSizeL2*visibleSizeL1+1:2*hiddenSizeL2*visibleSizeL1+hiddenSizeL2);
display_network(W'); % L2

%print -djpeg l2_sae_features.jpg   % save the visualization to a file 

%pause;

%%======================================================================
%% STEP : Implement convolution and pooling

fprintf('Loading training images for L3 training (convolution & pooling) ...\n')

csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    

sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
m = size(csvdata, 1); % amount of training examples

fprintf('Amount of training examples: %u \n', m);

%plain order - good for stop/restart training
shuffledOrder = 1:m;

% shuffling data for batch gradient descent
% DO NOT USE WITH CONV&POOL CACHING
%shuffledOrder = randperm(m);

shuffledSampleId = sampleId(shuffledOrder, :);
shuffledY = y(shuffledOrder, :);

        
%% =================== L3 (Softmax) layer Initialization ===================

%inputSize = 600 * 13 * 26;
inputSizeL3 = hiddenSizeL2 * ((img_h - patchSize + 1) / poolSize) * 2 * ((img_h - patchSize + 1) / poolSize);


if exist(strcat(datasetDir, 'SOFTMAX_THETA.mat'), 'file')
    % SOFTMAX_THETA.mat file exists. 
    fprintf('Loading softmax theta from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    load(strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    theta = softmaxTheta(:);  
else    
    % SOFTMAX_THETA.mat File does not exist. random initialization
    fprintf('Cant load Thetta1 from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    fprintf('  Do random initialization for softmax theta \n');
    theta = 0.005 * randn(numClassesL3 * inputSizeL3, 1);
end

%% =================== Training Layer3 with mini batch ===================
%
fprintf('\nTraining L3 with mini batch ... \n')

batchIterationCount = ceil(m / batchSize);

for trainingIter = 1 : trainingIterationCount % loop over training iterations
    fprintf('\nStarting training iteration %u from %u \n', trainingIter, trainingIterationCount);
    % loop over batches (training examples)
    
    for batchIter = 1 : batchIterationCount

        startPosition = (batchIter - 1) * batchSize + 1;
        endPosition = startPosition + batchSize - 1;
        if endPosition > m
            endPosition = m;
        end

        fprintf('\n training iteration (%u / %u): batch sub-iteration (%u / %u): start %u end %u from %u training samples \n', trainingIter, trainingIterationCount, batchIter, batchIterationCount, startPosition, endPosition, m);
        
%%======use caching for convolved and pooled features============        
        pooledFeaturesTempFile = strcat(datasetDir, tempDir, num2str(batchIter), '_pooledFeaturesTrain.mat');
        if exist(pooledFeaturesTempFile, 'file')
            % _pooledFeaturesTrain.mat file exists. 
            %fprintf('Found file with pooled features for iteration %u. Loading ... \n', batchIter);
            load(pooledFeaturesTempFile);
        else
            % File does not exist - do convolution and pooling
            fprintf('No file with pooled features for iteration %u. Do convolution and pooling ... \n', batchIter);
        
            [shuffledX] = loadImageSet(shuffledSampleId(startPosition:endPosition), strcat(datasetDir, imgDir), img_w, img_h);
            pooledFeaturesTrain = convolveAndPool(shuffledX, sae1OptTheta, hiddenSizeL2, img_h, img_w, patchSize, meanPatch, poolSize, convolutionsStepSize);
            save(pooledFeaturesTempFile, 'pooledFeaturesTrain');
        end
%%======================================================================        

%        fprintf('Pooled features (pooledFeaturesTrain) size ...\n')
%        size(pooledFeaturesTrain)

%        numTrainImages = size(shuffledX, 2)
        
        softmaxY = shuffledY(startPosition:endPosition, :);
        numTrainImages = size(pooledFeaturesTrain, 2);

        % Reshape the pooledFeatures to form an input vector for softmax
        softmaxX = permute(pooledFeaturesTrain, [1 3 4 2]);
        softmaxX = reshape(softmaxX, inputSizeL3, numTrainImages);

        %theta = softmaxTrain(theta, inputSize, numClasses, softmaxLambda, softmaxX, softmaxY, options);
        
        [theta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClassesL3, inputSizeL3, softmaxLambda, ...
                                   softmaxX, softmaxY), ...                                   
                              theta, softmaxOptions);        
        cost
    end; % for batchIter = 1 : batchIterationCount
    
    % Fold softmaxTheta into a nicer format
    softmaxTheta = reshape(theta, numClassesL3, inputSizeL3);
    % save softmaxTheta - can be used if training cycle interrupted 
    save(strcat(datasetDir, 'SOFTMAX_THETA.mat'), 'softmaxTheta');
    fprintf('Iteration %4i done - softmaxTheta saved', trainingIter);

end; % for trainingIter = 1 : trainingIterationCount % loop over training iterations


%}

fprintf('Training complete. \n');

%%======================================================================