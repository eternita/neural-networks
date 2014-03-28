%% Draft/mock for detecting coin centers and sizes
%
% Use intensity distirbution to detect coin's locations (reverse & obverse)
%
% Code translated to java (used to remove extra space around a coin in the photo)


%% Initialization
clear ; close all; clc

% update file name here
imageFileName = 'img/1.jpg';


coinImage = imread(imageFileName);
%% Image visualization

% convert to 0 - white, 255 - black
whiteBlackReverseImg = abs(double(coinImage) - 255) ./ 255;

% plot intensity distribution over width
plot(sum(whiteBlackReverseImg, 1));

% plot intensity distribution over height
%plot(sum(whiteBlackReverseImg, 2));

% Put some labels 
hold on;
xlabel('Width (pixels)')
ylabel('Black intensity')
hold off;
%pause;


%% Image centers/sizes calculation
[leftCenterX leftCenterY leftWidth leftHeight   rightCenterX rightCenterY rightWidth rightHeight] = detectCoin(coinImage)



