function [cost grad] = mlpCost(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%MLPCOST Implements the neural network cost function for a two layer
%neural network which performs classification


Theta1 = reshape(nn_params(1:(input_layer_size + 1) * hidden_layer_size), ...
                     (input_layer_size + 1), hidden_layer_size);

Theta2 = reshape(nn_params((1 + (input_layer_size + 1) * hidden_layer_size):end), ...
                     (hidden_layer_size + 1), num_labels);

             
%size(X)             
%size(Theta1)             
%size(Theta2)             

% Setup some useful variables
m = size(X, 2);
         
% ============================================

a1 = [ones(1, m); X];
z2 = Theta1' * a1;
a2 = [ones(1, m); sigmoid(z2)]; 

z3 = Theta2' * a2;
a3 = sigmoid(z3);

labelSequence = repmat(1:num_labels, [m, 1]); % m x num_labels
yLabels = repmat(y, [1, num_labels]); % m x num_labels

hk = a3'; % hypothesis 
      
yk = double(labelSequence == yLabels); % labels
   
costK = yk .* log(hk) + (1 - yk) .* log(1 - hk);
   
cost = -1/m * sum(costK(:));


% ============================================
% gradients
hk = a3';
yk = double(labelSequence == yLabels); % labels

delta3 = hk - yk;

temp = delta3 * Theta2';
temp = temp(:, 2:end); % remove bias

delta2 = temp .* sigmoidGradient(z2');

Theta2_grad = (1 / m) * a2 * delta3;   
Theta1_grad = (1 / m) * a1 * delta2;


% ============================================
% regularization   
Theta1Reg = Theta1;
Theta1Reg(:, 1) = 0;
Theta2Reg = Theta2;
Theta2Reg(:, 1) = 0;

cost = cost + lambda / (2 * m) * (sum(Theta1Reg(:) .^ 2) + sum(Theta2Reg(:) .^ 2));


Theta2_grad = Theta2_grad + lambda / m * Theta2Reg;
Theta1_grad = Theta1_grad + lambda / m * Theta1Reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
