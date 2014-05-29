%% Network configuration

numClasses = 10;

imgWidth = 28; % image width, ( width >= height )
imgHeight = 28; % image height

datasetDir = './'; % dataset root dir
tempDir = 'temp/'; % for pooled features used with mini batch
mkdir(strcat(datasetDir, tempDir)); % create temp dir - if doesn't exist

cnn = cell(1, 1); % for convolution layers L2

% L2
cnn{1}.inputWidth = imgWidth;
cnn{1}.inputHeight = imgHeight;
cnn{1}.inputChannels = 1;
cnn{1}.features = 400;
cnn{1}.inputVisibleSize = cnn{1}.inputWidth * cnn{1}.inputHeight;

cnn{1}.outputChannels = cnn{1}.features;
cnn{1}.outputSize = cnn{1}.outputChannels;

saeSparsityParam = 0.1;   % desired average activation of the hidden units.
saeLambda = 3e-3;     % weight decay for SAE (sparse auto-encoders)       
saeBeta = 3;            % weight of sparsity penalty term       


% L3
inputSizeL3 = cnn{1}.outputSize; 
softmaxLambda = 1e-4; % weight decay for L3


addpath ./libs/         % load libs
addpath ./libs/minFunc/

%  Use minFunc to minimize cost functions
saeOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
saeOptions.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
saeOptions.display = 'on';

softmaxOptions.Method = 'lbfgs'; % Use L-BFGS to optimize our cost function.
softmaxOptions.maxIter = 500; % update minFunc confugs for mini batch 
softmaxOptions.display = 'on';