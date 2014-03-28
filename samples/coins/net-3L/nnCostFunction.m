function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% ============================================

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)]; 

z3 = a2 * Theta2';
a3 = sigmoid(z3);

labelSequence = repmat(1:num_labels, [m, 1]); % m x num_labels
yLabels = repmat(y, [1, num_labels]); % m x num_labels

hk = a3; % hypothesis 
      
yk = double(labelSequence == yLabels); % labels
   
costK = yk .* log(hk) + (1 - yk) .* log(1 - hk);
   
J = -1/m * sum(costK(:));


% ============================================
% gradients
hk = a3';
yk = double(labelSequence == yLabels)'; % labels

delta3 = hk - yk;
temp = Theta2'*delta3;
temp = temp(2:end, :); % remove bias

delta2 = temp .* sigmoidGradient(z2');

Theta2_grad = (1 / m) * delta3 * a2;   
Theta1_grad = (1 / m) * delta2 * a1;
   

% ============================================
% regularization   
Theta1Reg = Theta1;
Theta1Reg(:, 1) = 0;
Theta2Reg = Theta2;
Theta2Reg(:, 1) = 0;

J = J + lambda / (2 * m) * (sum(Theta1Reg(:) .^ 2) + sum(Theta2Reg(:) .^ 2));


Theta2_grad = Theta2_grad + lambda / m * Theta2Reg;
Theta1_grad = Theta1_grad + lambda / m * Theta1Reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
