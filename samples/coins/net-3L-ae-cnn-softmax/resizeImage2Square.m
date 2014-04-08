function [out] = resizeImage2Square(image, imgW, imgH)
% image imgW*imgH X 1

% out imgW*imgH X 1

    img = reshape(image, imgH, imgW);

    
    out = zeros(imgW, imgW);
    out(1:imgH, :) = img;
    out(imgH+1: end, :) = 0;
    
    out = reshape(out, imgW ^ 2, 1);
    
end
