function saveThettas(nn_params, input_layer_size, hidden_layer_size, num_output_labels, datasetDir)

Theta1 = reshape(nn_params(1:(input_layer_size + 1) * hidden_layer_size), ...
                     (input_layer_size + 1), hidden_layer_size);

Theta2 = reshape(nn_params((1 + (input_layer_size + 1) * hidden_layer_size):end), ...
                     (hidden_layer_size + 1), num_output_labels);
                 
    % Thetas is big -> override to the same file
    save(strcat(datasetDir, 'THETA1.mat'), 'Theta1');
    save(strcat(datasetDir, 'THETA2.mat'), 'Theta2');

end
