function [pred] = softmaxPredict(theta, data)

% theta - trained theta
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%

[nop, pred] = max(theta * data);


end

