function testPrediction(imageDir, datasetFile, imgW, imgH, Theta1, Theta2)

%TESTPREDICTION Implements the test on specific dataset
%
%   imageDir - dir with dataset images
%   datasetFile - CSV file with dataset
%   imgW - image width
%   imgH - image height
%   Theta1 - L1 matrix
%   Theta2 - L2 matrix
%

    csvdata = csvread(datasetFile);

    sampleId = csvdata(:, 1); % first column is sampleId (imageIdx)
    y = csvdata(:, 2); % second column is coinIdx
    m = size(csvdata, 1); % amount of training examples

    fprintf('\nRunning prediction test ...  ');
    fprintf('\n    dataset %s ', datasetFile);
    fprintf('\n    image dir %s ', imageDir);
    fprintf('\n    image size %u X %u ', imgW, imgH);
    fprintf('\n    %u items in dataset ', m);

    [X] = loadImageSet(sampleId, imageDir, imgW, imgH);

    pred = predict(Theta1, Theta2, X);

    fprintf('\n    Prediction accuracy: %f\n', mean(double(pred == y) * 100));

end
