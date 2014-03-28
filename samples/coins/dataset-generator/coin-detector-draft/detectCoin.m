%% returns center coords, widht, height for left and right images
function [leftCenterX leftCenterY leftWidth leftHeight   rightCenterX rightCenterY rightWidth rightHeight] = detectCoin(coinImage)

%% Initialization

% images loaded differently in Java vs Matlab
javaMatlabDiff = 1; 

width = size(coinImage, 2);
heigth = size(coinImage, 1);

% convert to 0 - white, 255 - black
coinImage = abs(double(coinImage) - 255) ./ 255;

widthSum = sum(coinImage, 1);

widthSum = javaMatlabDiff*widthSum;

%totalWidthSum = sum(widthSum)

%maxSpike = max(widthSum)
avg = mean(widthSum)
widthSum(widthSum > 3*avg) = avg;
%widthSum(widthSum > 3*avg) = 0;
middleMinThresould = 0.4 * avg


%% Width processing
% check threshold
uderThresholdWidth = widthSum - middleMinThresould;

%plot(uderThresholdWidth);
%pause;

uderThresholdWidthIndexes = find(uderThresholdWidth < 0);

% get start/end of the images using the X scale (width)
wStartLeftIdx = max(find(uderThresholdWidth(1:width/4) < 0));
wEndLeftIdx = width/4 - 1 + min(find(uderThresholdWidth(width/4:3*width/4) < 0));

wStartRightIdx = width/4 - 1 + max(find(uderThresholdWidth(width/4:3*width/4) < 0));
wEndRightIdx = 3*width/4 - 1 + min(find(uderThresholdWidth(3*width/4:width) < 0));

% split line betlween left and right images
wSplitIdx = 0.5*(wEndLeftIdx + wStartRightIdx);

%% Height processing

heightSumLeft = javaMatlabDiff * sum(coinImage(:, (1:wSplitIdx)), 2);

hAvgLeft = mean(heightSumLeft)
heightSumLeft(heightSumLeft > 3*hAvgLeft) = hAvgLeft;
hThresouldLeft = 0.4 * hAvgLeft

uderThresholdHeightLeft = heightSumLeft - hThresouldLeft;

%plot(uderThresholdHeightLeft)
%pause;

%hStartLeftIdx = max(find(uderThresholdHeightLeft(1:heigth/2) < 0))
%hEndLeftIdx = heigth/2 - 1 + min(find(uderThresholdHeightLeft(heigth/2:heigth) < 0))

hStartLeftIdx = min(find(uderThresholdHeightLeft(1:heigth/2) > 0))
hEndLeftIdx = heigth/2 - 1 + max(find(uderThresholdHeightLeft(heigth/2:heigth) > 0))

heightSumRight = javaMatlabDiff * sum(coinImage(:, (wSplitIdx + 1:end)), 2);

hAvgRight = mean(heightSumRight)
heightSumRight(heightSumRight > 3*hAvgRight) = hAvgRight;
hThresouldRight = 0.4 * hAvgRight

uderThresholdHeightRight = heightSumRight - hThresouldRight;

%plot(uderThresholdHeightRight)


%hStartRightIdx = max(find(uderThresholdHeightRight(1:heigth/2) < 0));
%hEndRightIdx = heigth/2 - 1 + min(find(uderThresholdHeightRight(heigth/2:heigth) < 0));

hStartRightIdx = min(find(uderThresholdHeightRight(1:heigth/2) > 0));
hEndRightIdx = heigth/2 - 1 + max(find(uderThresholdHeightRight(heigth/2:heigth) > 0));

leftCenterX = ceil(0.5*(wStartLeftIdx + wEndLeftIdx));
leftCenterY = ceil(0.5*(hStartLeftIdx + hEndLeftIdx));

rightCenterX = ceil(0.5*(wStartRightIdx + wEndRightIdx));
rightCenterY = ceil(0.5*(hStartRightIdx + hEndRightIdx));

leftWidth = ceil(wEndLeftIdx - wStartLeftIdx);
leftHeight = ceil(hEndLeftIdx - hStartLeftIdx);

rightWidth = ceil(wEndRightIdx - wStartRightIdx);
rightHeight = ceil(hEndRightIdx - hStartRightIdx);


%------------------------
end

