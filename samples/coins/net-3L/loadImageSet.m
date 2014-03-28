function [X] = loadImageSet(sampleId, imgDir, imgW, imgH)

    m = size(sampleId, 1); % amount of training examples
    X = zeros(m, imgW*imgH); % loaded images
    
    for i = 1 : m
        gImg = imread(strcat(imgDir, num2str(sampleId(i)), '.jpg')); % read image from the file
        imgV = reshape(gImg, 1, imgW*imgH); % unroll
        imgV = abs(double(imgV) - 255); % black-white switch
        X(i, :) = imgV; 
    end
        
end
