function [out] = resizeImage2Square(image, imgW, imgH)
% Coin images are not square. usualy width = 2*height
% function assume imgW >= imgH
% resizeImage2Square makes coin image square (so, it can be displayed with display_network)
%
% image imgW*imgH X 1
% out   imgW*imgW X 1

    img = reshape(image, imgH, imgW);

    
    out = zeros(imgW, imgW);
    out(1:imgH, :) = img;
    out(imgH+1: end, :) = 0;
    
    out = reshape(out, imgW ^ 2, 1);
    
end
