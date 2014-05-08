function [patches, meanPatch] = getPatches2(images, patchSize, numPatches)
% images imgW X imgH X channels X imageSamples

% out imgW*imgH*channelDim X 1

imgW = size(images, 1);
imgH = size(images, 2);
channelDim = size(images, 3);
numImages = size(images, 4);


%images3 = reshape(images, imgH, imgW, size(images, 2));

% Initialize patches with zeros.  
patches = zeros(channelDim * patchSize * patchSize, numPatches);

for i = 1 : size(patches, 2)
    imgId = randi([1, numImages]);
    x = randi([1, imgW - patchSize]);
    y = randi([1, imgH - patchSize]);
%    patches(:, i) = reshape(images3(21:28,21:28, imgId), 1, patchsize * patchsize);
%    patches(:, i) = reshape(images3(x:x+patchSize-1,y:y+patchSize-1, imgId), 1, patchSize * patchSize);
    patches(:, i) = reshape(images(x:x+patchSize-1, y:y+patchSize-1, :, imgId), 1, channelDim * patchSize * patchSize);
end;

% Zero-mean the data
meanPatch = mean(patches, 2);  
patches = bsxfun(@minus, patches, meanPatch);

end
