function [prediction] = softmaxPredict(theta, data, maxTopPredictions)

% theta - trained theta
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% maxTopPredictions - amount of top redictions
%
m = size(data, 2);

prediction = zeros(m, maxTopPredictions); % M x maxTopPredictions

M = theta * data; % K x M



[nop, thisPred] = max(M);

prediction(:, 1) = thisPred;

if maxTopPredictions > 1
    for j = 2:maxTopPredictions

        for i = 1:m
            M(thisPred(i), i) = NaN; % reset previous max
        end

        [nop, thisPred] = max(M);
        prediction(:, j) = thisPred;
    end
end



end

