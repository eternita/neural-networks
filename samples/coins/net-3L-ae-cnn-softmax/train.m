clear ; close all; clc % cleanup
%%======================================================================
%% Configuration
%  ! Setup and check all parameters before run

datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-5_25_400_200/'; % dataset root dir
trainSetCSVFile = 'coin.tr.shuffled.csv'; % this file will be generated from 'coin.tr.csv'

unlabeledImgDir = 'img_unlabeled/'; % sub directory with images for auto-encoder training (unlabeled/for unsupervised feature extraction)
imgDir = 'img_grayscale/'; % sub directory with images
tempDir = 'temp/'; % for pooled features used with mini batch

imgW = 400; % image width, ( width >= height )
imgH = 200; % image height

% !! WHEN CHANGE batchSizeL3 - CLEAN UP / DELETE TEMP DIRECTORY (tempDir)
batchSizeL3 = 100; % batch size for L3 mini-batch algorithm
numTrainIterL3 = 400; % L3 amount of iterations over whole training set
numClassesL3 = 5; % amount of output lables, classes (e.g. coins)

hiddenSizeL2 = 600;     % L2 hidden layer size

patchSize = 6; % patch size/dimention for L2 feature extraction (using auto-encodes)
saeNumPatches = 10000; % amount of patches for auto-encoder training

poolSize = 15; % used for pooling convolved features

visibleSizeL1 = patchSize * patchSize; % number of input units for the patch

saeSparsityParam = 0.01;   % desired average activation of the hidden units.
saeLambda = 0.003;     % weight decay for SAE (sparse auto-encoders)       
saeBeta = 3;            % weight of sparsity penalty term       

addpath ../libs/         % load libs
addpath ../libs/minFunc/

convolutionsStepSize = 50;

softmaxLambda = 1e-4; % weight decay for L3

%  Use minFunc to minimize cost functions
saeOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
saeOptions.maxIter = 800;	  % Maximum number of iterations of L-BFGS to run 
saeOptions.display = 'on';

softmaxOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
softmaxOptions.maxIter = 1; % update minFunc confugs for mini batch 
softmaxOptions.display = 'on';


%% Initializatoin

% create suffled training set - if doesn't created
if ~exist(strcat(datasetDir, 'coin.tr.shuffled.csv'), 'file')
    fprintf('Generating shuffled training set coin.tr.shuffled.csv from coin.tr.csv \n');
    shuffleTrainingSet(datasetDir, 'coin.tr.csv', 'coin.tr.shuffled.csv');
end

mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist

%% Visualize some full size images from training set
% make sure visualy we work on the right dataset

csvdata = csvread(strcat(datasetDir, trainSetCSVFile));    
visualAmount = 3^2;
fprintf('Visualize %u full size images ...\n', visualAmount);
[previewX] = loadImageSet(csvdata(1:visualAmount, 1), strcat(datasetDir, imgDir), imgW, imgH);
fullSizeImages = zeros(imgW^2, visualAmount);
for i = 1:visualAmount
    % visualization works for squared matrixes
    % before visualization convert img_h x img_w -> img_w * img_w
    fullSizeImages(:, i) = resizeImage2Square(previewX(:, i), imgW, imgH);
end;

display_network(fullSizeImages);

clear previewX fullSizeImages;

%pause;
%}
%%======================================================================

%% Patches for auto-encoders training

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
    unlabeledImagesX = zeros(imgW*imgH, length(unlabeledImgFiles)); % unlabeled images
    % loop over files and load images into matrix
    for idx = 1:length(unlabeledImgFiles)
        gImg = imread([unlabeledImgDirFullPath unlabeledImgFiles(idx).name]);
        imgV = reshape(gImg, 1, imgW*imgH); % unroll       
        unlabeledImagesX(:, idx) = imgV; 
    end
    
    fprintf('Generating %u patches (%u x %u) from images ...\n', saeNumPatches, patchSize, patchSize);
    [patches, meanPatch] = getPatches(unlabeledImagesX, imgW, imgH, patchSize, saeNumPatches);

    % remove (clean up some memory)
    clear shuffledX

    save(strcat(datasetDir, 'PATCHES.mat'), 'patches', 'meanPatch');
    display_network(patches(:,randi(size(patches,2),200,1)));
    fprintf('Patches generation complete ...\n');
%    pause;
end



%display_network(patches(:,randi(size(patches,2),200,1)));
%pause;

%%======================================================================
%% Learning L2 features with sparse autoencoders 


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
                                       saeLambda, saeSparsityParam, ...
                                       saeBeta, patches), ...
                                  theta, saeOptions);

    save(strcat(datasetDir, 'SAE1_FEATURES.mat'), 'sae1OptTheta', 'meanPatch');
end

% Visualization Sparser Autoencoder Features to see that the features look good
W = reshape(sae1OptTheta(1:visibleSizeL1 * hiddenSizeL2), hiddenSizeL2, visibleSizeL1);
b = sae1OptTheta(2*hiddenSizeL2*visibleSizeL1+1:2*hiddenSizeL2*visibleSizeL1+hiddenSizeL2);
display_network(W'); % L2

%print -djpeg l2_sae_features.jpg   % save the visualization to a file 

%pause;

%%======================================================================
%% Implement convolution and pooling

fprintf('Loading training images for L3 training (convolution & pooling) ...\n')

csvdata = csvread(strcat(datasetDir, trainSetCSVFile));    

sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
m = size(csvdata, 1); % amount of training examples

fprintf('Amount of training examples: %u \n', m);

        
%% L3 (Softmax) layer Initialization 

%inputSize = 600 * 13 * 26;
inputSizeL3 = hiddenSizeL2 * ((imgH - patchSize + 1) / poolSize) * 2 * ((imgH - patchSize + 1) / poolSize);

if exist(strcat(datasetDir, 'SOFTMAX_THETA.mat'), 'file')
    % SOFTMAX_THETA.mat file exists. 
    fprintf('Loading softmax theta from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    load(strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    theta = softmaxTheta(:);  
else    
    % SOFTMAX_THETA.mat File does not exist. random initialization
    fprintf('Cant load softmaxTheta from %s  \n', strcat(datasetDir, 'SOFTMAX_THETA.mat'));
    fprintf('  Do random initialization for softmax theta \n');
    theta = 0.005 * randn(numClassesL3 * inputSizeL3, 1);
end

%% Training Layer3 using mini batch gradient descent
%
fprintf('\nTraining L3 with mini batch ... \n')

batchIterationCount = ceil(m / batchSizeL3);

for trainingIter = 1 : numTrainIterL3 % loop over training iterations
    fprintf('\nStarting training iteration %u from %u \n', trainingIter, numTrainIterL3);
    % loop over batches (training examples)
    
    for batchIter = 1 : batchIterationCount

        startPosition = (batchIter - 1) * batchSizeL3 + 1;
        endPosition = startPosition + batchSizeL3 - 1;
        if endPosition > m
            endPosition = m;
        end

        fprintf('\n training iteration (%u / %u): batch sub-iteration (%u / %u): start %u end %u from %u training samples \n', trainingIter, numTrainIterL3, batchIter, batchIterationCount, startPosition, endPosition, m);
        
%%======use caching for convolved and pooled features============        
        pooledFeaturesTempFile = strcat(datasetDir, tempDir, num2str(batchIter), '_pooledFeaturesTrain.mat');
        if exist(pooledFeaturesTempFile, 'file')
            % _pooledFeaturesTrain.mat file exists. 
            %fprintf('Found file with pooled features for iteration %u. Loading ... \n', batchIter);
            load(pooledFeaturesTempFile);
        else
            % File does not exist - do convolution and pooling
            fprintf('No file with pooled features for iteration %u. Do convolution and pooling ... \n', batchIter);
        
            [shuffledX] = loadImageSet(sampleId(startPosition:endPosition), strcat(datasetDir, imgDir), imgW, imgH);
            pooledFeaturesTrain = convolveAndPool(shuffledX, sae1OptTheta, hiddenSizeL2, imgH, imgW, patchSize, meanPatch, poolSize, convolutionsStepSize);
            save(pooledFeaturesTempFile, 'pooledFeaturesTrain');
        end
%%======================================================================        

        softmaxY = y(startPosition:endPosition, :);
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

fprintf('Training complete. \n');

%%======================================================================