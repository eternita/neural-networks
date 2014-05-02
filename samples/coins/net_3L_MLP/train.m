% Runs training code using mini batch & L-BFGS for gradient descent 

clear ; close all; clc % cleanup

%% Setup the parameters - Check / set parameters before run

datasetDir = 'C:/Develop/_n4j-nn-data/dataset-30_400_200_x7/'; % dataset root dir
trainSetCSVFile = 'coin.tr.shuffled.csv'; % this file will be generated from 'coin.tr.csv'
imgDir = 'img_gau/'; % sub directory with images
tempDir = 'temp/'; % for pooled features used with mini batch

img_w = 400; % image width
img_h = 200; % image height

input_layer_size  = img_w * img_h;  % input layer size
hidden_layer_size = 1000;           % hidden layer size
num_output_labels = 30;             % output layer size, amount of coinIdx, from 1 to ...

lambda = 3; % weight decay term (regularization)
                          
trainingIterationCount = 1000; % amount of iterations over whole training set

batchSize=330; % training sets in a mini-batch


addpath ../libs/         % load libs
addpath ../libs/minFunc/

%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % use L-BFGS to optimize our cost function. 
options.maxIter = 1;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

cost_visual = zeros(trainingIterationCount, 1); % Cost Func over number of training iterations

mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist


%% =========== Loading and Initializing Pameters =============

% create suffled training set - if doesn't created
if ~exist(strcat(datasetDir, 'coin.tr.shuffled.csv'), 'file')
    fprintf('Generating shuffled training set coin.tr.shuffled.csv from coin.tr.csv \n');
    shuffleTrainingSet(datasetDir, 'coin.tr.csv', 'coin.tr.shuffled.csv');
end

% show current time
time = datestr(now,'yyyy-mm-dd HH:MM:SS FFF');
fprintf('Start Time: %s \n', time);

% Load Training Data
fprintf('Loading training data ...\n');

csvdata = csvread(strcat(datasetDir, trainSetCSVFile));    
sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
m = size(csvdata, 1); % amount of training examples

fprintf('Amount of training examples: %u \n', m);


fprintf('Initializing Neural Network Parameters ...\n')

%----- start load Thettas -------------------
% if Thettas were saved - load them from files. Otherwise - do random
% initialization for Thettas

theta1File = strcat(datasetDir, tempDir, 'THETA1.mat');
if exist(theta1File, 'file')
    % THETTA1.mat file exists. 
    fprintf('Loading Thetta1 from %s  \n', theta1File);
    load(theta1File);
    initial_Theta1 = Theta1;  
else
    % File does not exist. random initialization
    fprintf('Cant load Thetta1 from %s  \n  Do random initialization for Thetta1 \n', theta1File);
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
end

theta2File = strcat(datasetDir, tempDir, 'THETA2.mat');
if exist(theta2File, 'file')
    % THETTA2.mat file exists. 
    fprintf('Loading Thetta2 from %s  \n', theta2File);
    load(theta2File);
    initial_Theta2 = Theta2;  
else
    % File does not exist. random initialization
    fprintf('Cant load Thetta2 from %s  \n  Do random initialization for Thetta2 \n', theta2File);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_output_labels);
end

fprintf('Theta1: %u x %u \n', size(initial_Theta1, 2), size(initial_Theta1, 1));
fprintf('Theta2: %u x %u \n', size(initial_Theta2, 2), size(initial_Theta2, 1));
%----- end load Thettas -------------------

% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% remove unused variable (clean up some memory)
clear initial_Theta1 initial_Theta2 csvdata

%
%% =================== Training NN ===================
%
fprintf('\nTraining Neural Network... \n')

batchIterationCount = ceil(m / batchSize);

costs = zeros(trainingIterationCount, 1); % cost func over training iterations

for trainingIter = 1 : trainingIterationCount % loop over training iterations
    fprintf('\nStarting training iteration %u from %u \n', trainingIter, trainingIterationCount);
    % loop over batches (training examples)

    
    iterCost = 0;
    for batchIter = 1 : batchIterationCount

        startPosition = (batchIter - 1) * batchSize + 1;
        endPosition = startPosition + batchSize - 1;
        if endPosition > m
            endPosition = m;
        end

        fprintf('\n training iteration (%u / %u): batch sub-iteration (%u / %u): start %u end %u from %u \n', trainingIter, trainingIterationCount, batchIter, batchIterationCount, startPosition, endPosition, m);
        
        [shuffledX] = loadImageSet(sampleId(startPosition:endPosition), strcat(datasetDir, imgDir), img_w, img_h);
        
        shuffledX = bsxfun(@minus, shuffledX, mean(shuffledX, 1));
        
        %size(shuffledX)
        costFunction = @(p) mlpCost(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_output_labels, shuffledX, y(startPosition:endPosition), lambda);
        [nn_params, cost] = minFunc(costFunction, nn_params, options);
        
        iterCost = iterCost + cost;
    end;
    iterCost = iterCost/batchIterationCount;
    costs(trainingIter) = iterCost;
    
    
    fprintf('Iteration %4i \r', trainingIter);
    
    % save thetas - can be used if training cycle interrupted 
    saveThettas(nn_params, input_layer_size, hidden_layer_size, num_output_labels, strcat(datasetDir, tempDir));
    
    figure(2);
    xlabel('Training iterations');
    ylabel('Cost function');
    title('Cost function over training iterations');
    plot(costs);    

end;

fprintf('\nTraining Neural Network Complete ... \n')

             
% show current time
time = datestr(now,'yyyy-mm-dd HH:MM:SS FFF');
fprintf('Complited at: %s \n', time);
