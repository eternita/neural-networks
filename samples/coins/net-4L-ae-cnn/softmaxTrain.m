function [softmaxOptTheta] = softmaxTrain(theta, inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end


%{
fprintf('softmaxTrain theta size: %u X %u \n', size(theta, 1), size(theta, 2));
fprintf('softmaxTrain inputSize: %u \n', inputSize);
fprintf('softmaxTrain numClasses: %u \n', numClasses);
fprintf('softmaxTrain lambda: %u \n', lambda);
fprintf('softmaxTrain inputData size: %u X %u \n', size(inputData, 1), size(inputData, 2));
fprintf('softmaxTrain labels size: %u X %u \n', size(labels, 1), size(labels, 2));
%}

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...                                   
                              theta, options);



                          
end                          
