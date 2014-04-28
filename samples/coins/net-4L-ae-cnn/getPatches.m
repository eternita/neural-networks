function [patches, meanPatch] = getPatches(images, imgW, imgH, patchSize, numPatches)
% images imgW*imgH X imageSamples

% out imgW*imgH X 1

images3 = reshape(images, imgH, imgW, size(images, 2));

% Initialize patches with zeros.  
patches = zeros(patchSize*patchSize, numPatches);

for i = 1 : size(patches, 2)
    imgId = randi([1, size(images3, 3)]);
    x = randi([1, size(images3, 1) - patchSize]);
    y = randi([1, size(images3, 1) - patchSize]);
%    patches(:, i) = reshape(images3(21:28,21:28, imgId), 1, patchsize * patchsize);
    patches(:, i) = reshape(images3(x:x+patchSize-1,y:y+patchSize-1, imgId), 1, patchSize * patchSize);
end;

% Zero-mean the data
meanPatch = mean(patches, 2);  
patches = bsxfun(@minus, patches, meanPatch);

end
