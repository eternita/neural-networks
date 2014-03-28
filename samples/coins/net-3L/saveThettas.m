function saveThettas(nn_params, input_layer_size, hidden_layer_size, datasetDir)

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_output_labels, (hidden_layer_size + 1));

%    save(strcat(workingDir, num2str(trainingIter), '_THETTA1.mat'), 'Theta1');
    % Thetas is big -> override to the same file
    save(strcat(datasetDir, 'THETTA1.mat'), 'Theta1');
    save(strcat(datasetDir, 'THETTA2.mat'), 'Theta2');

end
