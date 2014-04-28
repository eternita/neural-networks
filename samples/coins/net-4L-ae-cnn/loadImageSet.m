function [X] = loadImageSet(sampleId, imgDir, imgW, imgH)

    m = size(sampleId, 1); % amount of training examples
    X = zeros(imgW*imgH, m); % loaded images
    
    for i = 1 : m
        gImg = imread(strcat(imgDir, num2str(sampleId(i)), '.jpg')); % read image from the file
        % gImg h x w
        
        imgV = reshape(gImg, 1, imgW*imgH); % unroll       
        X(:, i) = imgV; 
    end

end
