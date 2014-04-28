function convolvedFeatures = convolve(patchDim, numFeatures, images, W, b, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  meanPatch - meanPatch matrices used for preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

patchSize = patchDim * patchDim;
numImages = size(images, 4);
imageDimH = size(images, 1);
imageDimW = size(images, 2);
imageChannels = size(images, 3);

assert(numFeatures == size(W, 1), 'W should have numFeatures rows');
assert(patchSize*imageChannels == size(W, 2), 'W should have patchDim^2*imageChannels columns');


% size(b)  % 400 x 1
% size(W)  % 400 x 192
% size(meanPatch)  % 192 x 1
% size(images)  % 64 x 64 x 3 x 8

% size(WT) = 400 x 192
% size(x) = 192 * m
% size(b_w) = 400 x 1

WT = W; % don't use ZCA whitening with sparse auto encoders
b_w = b - WT * meanPatch;

% size(W) == size(WT) == (numFeatures, patchDim * imageChannels)
% Compute per-channel features for 2-D convolution
features = zeros(imageChannels, numFeatures, patchSize);
for channel = 1:imageChannels
  offset = patchSize * (channel-1);
  features(channel, :, :) = WT(:, offset+1:offset+patchSize);
end

% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDimH - patchDim + 1, imageDimW - patchDim + 1);
for imageNum = 1:numImages
  for featureNum = 1:numFeatures

    % convolution of image with feature matrix for each channel
    convolvedImage = zeros(imageDimH - patchDim + 1, imageDimW - patchDim + 1);
    for channel = 1:imageChannels

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      offset = patchSize * (channel-1);
      feature = reshape(WT(featureNum, offset+1:offset+patchSize), patchDim, patchDim);

      % Flip the feature matrix because of the definition of convolution
      feature = flipud(fliplr(squeeze(feature)));

      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      convolved = conv2(im, feature, 'valid');
      convolvedImage = convolvedImage + convolved;

    end

    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    convolvedImage = sigmoid(convolvedImage + b_w(featureNum));

    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end

end

