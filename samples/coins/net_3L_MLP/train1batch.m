% Runs training code using mini batch & L-BFGS for gradient descent 

clear ; close all; clc % cleanup

%% Setup the parameters - Check / set parameters before run

graphDirName = 'logs'; % logs directory for output charts (debug only)

mkdir(graphDirName);

% dataset directory
%datasetDir = 'E:/nn4coins/dataset-247_19602_297_400_200_gau-mexico/';
datasetDir = 'C:/Develop/src/pavlikovkskiy/chn/data/dataset-5_1650_25_400_200_gau/';
img_w = 400; % image width
img_h = 200; % image height

input_layer_size  = img_w * img_h;  % input layer size
hidden_layer_size = 1000;           % hidden layer size
num_output_labels = 5;             % output layer size, amount of coinIdx, from 1 to ...

lambda = 3; % weight decay term (regularization)
                          
trainingIterationCount = 100; % amount of iterations over whole training set

batchSize=1000; % training sets in a mini-batch


addpath ../libs/         % load libs
addpath ../libs/minFunc/

%  Use minFunc to minimize the function
options.Method = 'lbfgs'; % use L-BFGS to optimize our cost function. 
options.maxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

cost_visual = zeros(trainingIterationCount, 1); % Cost Func over number of training iterations

%plot(cost_visual);
%pause;

%% =========== Loading and Initializing Pameters =============

% show current time
time = datestr(now,'yyyy-mm-dd HH:MM:SS FFF');
fprintf('Start Time: %s \n', time);

% Load Training Data
fprintf('Loading training data ...\n')
csvdata = csvread(strcat(datasetDir, 'coin.tr.csv'));    

sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
y = csvdata(:, 2); % second column is coinIdx
m = size(csvdata, 1); % amount of training examples

fprintf('Amount of training examples: %u \n', m);

% shuffling data for batch gradient descent
shuffledOrder = randperm(m);
shuffledSampleId = sampleId(shuffledOrder, :);
shuffledY = y(shuffledOrder, :);


fprintf('Initializing Neural Network Parameters ...\n')

%----- start load Thettas -------------------
% if Thettas were saved - load them from files. Otherwise - do random
% initialization for Thettas

if exist(strcat(datasetDir, 'THETTA1.mat'), 'file')
    % THETTA1.mat file exists. 
    fprintf('Loading Thetta1 from %s  \n', strcat(datasetDir, 'THETTA1.mat'));
    load(strcat(datasetDir, 'THETTA1.mat'));
    initial_Theta1 = Theta1;  
else
    % File does not exist. random initialization
    fprintf('Cant load Thetta1 from %s  \n', strcat(datasetDir, 'THETTA1.mat'));
    fprintf('  Do random initialization for Thetta1 \n');
    initial_Theta1 = mlpMatrixLayerInit(input_layer_size, hidden_layer_size);
end

if exist(strcat(datasetDir, 'THETTA2.mat'), 'file')
    % THETTA2.mat file exists. 
    fprintf('Loading Thetta2 from %s  \n', strcat(datasetDir, 'THETTA2.mat'));
    load(strcat(datasetDir, 'THETTA2.mat'));
    initial_Theta2 = Theta2;  
else
    % File does not exist. random initialization
    fprintf('Cant load Thetta2 from %s  \n', strcat(datasetDir, 'THETTA2.mat'));
    fprintf('  Do random initialization for Thetta2 \n');
    initial_Theta2 = mlpMatrixLayerInit(hidden_layer_size, num_output_labels);
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

[shuffledX] = loadImageSet(shuffledSampleId, strcat(datasetDir, 'img/'), img_w, img_h);

%size(shuffledX)
costFunction = @(p) nnCostFunction(p, ...
                               input_layer_size, ...
                               hidden_layer_size, ...
                               num_output_labels, shuffledX, shuffledY, lambda);
[nn_params, cost] = minFunc(costFunction, nn_params, options);

fprintf('\nTraining Neural Network Complete ... \n')

% save thetas
saveThettas(nn_params, input_layer_size, hidden_layer_size, num_output_labels, datasetDir);
             
% show current time
time = datestr(now,'yyyy-mm-dd HH:MM:SS FFF');
fprintf('Complited at: %s \n', time);
