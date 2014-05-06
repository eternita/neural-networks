function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
numCases = size(data, 2);

%groundTruth = full(sparse(labels, 1:numCases, 1)); % doesn't work if labels has no max case

%{
fprintf('softmaxCost theta size: %u X %u \n', size(theta, 1), size(theta, 2));
fprintf('softmaxCost data size: %u X %u \n', size(data, 1), size(data, 2));
fprintf('softmaxCost labels size: %u X %u \n', size(labels, 1), size(labels, 2));
fprintf('softmaxCost numClasses: %u \n', numClasses);
fprintf('softmaxCost inputSize: %u \n', inputSize);
%}

allLabels = repmat((1:numClasses)', [1, numCases]); % numClasses X numCases (labels x m)
yLabels = repmat(labels', [numClasses, 1]); % m x labels (repeated labels)
groundTruth = double(allLabels == yLabels);

M = theta * data;
p = bsxfun(@rdivide, exp(M), sum(exp(M)));

cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2); 

thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;

% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

