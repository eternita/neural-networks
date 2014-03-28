package net.coinshome.nn.input.img;

import java.awt.image.BufferedImage;

import net.coinshome.nn.utils.MatrixUtils;


public class CoinDetector {



	public static CoinImageInfo detect(BufferedImage img) 
	{
		CoinImageInfo cii = null;
		try
		{
//			File inImgFile = new File(IMG_INPUT_DIR, imgId + ".jpg");
			
//	        BufferedImage img = ImageIO.read(new FileInputStream(inImgFile));
	        
	        BufferedImage diffOfGau = ImageUtils.detectEdges(img);
			int[][] imgarrInt = ImageUtils.img2pixelArray(diffOfGau);
	        
			// looks like no reason to convert to grayscale
//	      BufferedImage grayImg = Img2GrayScaleConverter.convert(diffOfGau);
//			int[][] imgarrInt = ImageUtils.img2pixelArray(grayImg);
	        
//			System.out.println(" -- ");
//			MatrixUtils.print(imgarrInt);
			
	        ImageUtils.hex2int(imgarrInt); 
			double[][] imgarr = MatrixUtils.int2double(imgarrInt);
//			CoinImageInfo cii = detect(imgarr);
			
	/*		
	        // test only
			img = ImageIO.read(new FileInputStream(diffOfGauImgFile));
			imgarrInt = ImageUtils.img2pixelArray(img);
	//*/
			
			cii = detect(imgarr);
		} catch (Exception ex)
		{
        	System.out.println(ex.getMessage());
//			ex.printStackTrace();
			return null;
		}
		
//		System.out.println(cii);
		if (!isValid(cii))
		{
//			System.out.println("--- coin image info is not detected");
			return null;
		}
		
        return cii;

	}
	
	private static boolean isValid(CoinImageInfo cii)
	{
		if (0 >= cii.leftCenterX)
			return false;
		if (0 >= cii.leftCenterY)
			return false;
		if (0 >= cii.rightCenterX)
			return false;
		if (0 >= cii.rightCenterY)
			return false;
		
		if (0 >= cii.leftWidth)
			return false;
		if (0 >= cii.leftHeight)
			return false;
		if (0 >= cii.rightWidth)
			return false;
		if (0 >= cii.rightHeight)
			return false;
		
		int leftWidthMax = Math.max(cii.leftWidth, cii.rightWidth);
		int leftWidthMin = Math.min(cii.leftWidth, cii.rightWidth);
		
		if (leftWidthMax / leftWidthMin > 1.2) // widths ratio more 20%
			return false;
		
		int width = Math.max(cii.leftWidth, cii.rightWidth);
		int height = Math.max(cii.leftHeight, cii.rightHeight);
		
		if (Math.max(width, height) / Math.min(width, height) > 3) // width/height ratio > 300%
			return false;
		
		
		return true;
	}

	private static CoinImageInfo detect(double[][] imgarr) throws Exception
	{
	
//		% images loaded differently in Java vs Matlab
//		javaMatlabDiff = 3; 
		int javaMatlabDiff = 3;
		
//		width = size(coinImage, 2);
//		heigth = size(coinImage, 1);
		
		int width = imgarr[0].length;
		int heigth = imgarr.length;
		
		// convert to 0 - white, 255 - black
		// and normalize
		//
		// coinImage = abs(double(coinImage) - 255) ./ 255;

//		imgarr = MatrixUtils.divide(
//					MatrixUtils.abs(
//							MatrixUtils.minus(imgarr, 255)
//					), 
//					255); 
		
		// abs(double(coinImage) - 255)
        ImageUtils.mean(imgarr); 
        
		imgarr = MatrixUtils.divide(imgarr, 255); 
		

		
		// widthSum = sum(coinImage, 1);
		double[] widthSum = MatrixUtils.sum(imgarr, 1);
		
		// java-matlab img loaders compatibility
		widthSum = MatrixUtils.multiplex(widthSum, javaMatlabDiff);
		
//		System.out.println("size " + widthSum.length);
//		System.out.println(" -- ");
//		MatrixUtils.print(widthSum);

//		maxSpike = max(widthSum);
//		if maxSpike < 25
//		    maxSpike = 25;
//		end
		
		double maxSpike = MatrixUtils.max(widthSum);
		if (maxSpike < 25)
			maxSpike = 25;
		
//		middleMinThresould = 3 + (4/85) * (maxSpike - 25)
		double middleMinThresould = 3d + (4d/85d) * (maxSpike - 25d);
//		System.out.println("middleMinThresould: " + middleMinThresould);
		
//		% check threshold
//		uderThresholdWidth = widthSum - middleMinThresould;
		
		double[] uderThresholdWidth = MatrixUtils.minus(widthSum, middleMinThresould);
		
//		uderThresholdWidthIndexes = find(uderThresholdWidth < 0);
		
		double[] uderThresholdWidthIndexes = MatrixUtils.findLess(uderThresholdWidth, 0);
		
		
//		% get start/end of the images using the X scale (width)
//		wStartLeftIdx = max(find(uderThresholdWidth(1:width/4) < 0));
//		wEndLeftIdx = width/4 - 1 + min(find(uderThresholdWidth(width/4:3*width/4) < 0));

//		int wStartLeftIdx = 

//		System.out.println("MatrixUtils.part(uderThresholdWidth, 0, width/4 - 1)");
//		MatrixUtils.print(MatrixUtils.part(uderThresholdWidth, 0, width/4 - 1));
//		MatrixUtils.print(
//				MatrixUtils.findLess(
//						MatrixUtils.part(uderThresholdWidth, 0, width/4 - 1), 
//						0)
//				);
		
		int wStartLeftIdx = (int) MatrixUtils.max(
									MatrixUtils.findLess(
											MatrixUtils.part(uderThresholdWidth, 0, width/4 - 1), 
											0)
								);
//		System.out.println("wStartLeftIdx: " + wStartLeftIdx);
		int wEndLeftIdx = width/4 - 1 + (int) MatrixUtils.min(
				MatrixUtils.findLess(
						MatrixUtils.part(uderThresholdWidth, width/4, 3*width/4 - 1), 
						0)
			);
//		System.out.println("wEndLeftIdx: " + wEndLeftIdx);
		
		
//		wStartRightIdx = width/4 - 1 + max(find(uderThresholdWidth(width/4:3*width/4) < 0));
//		wEndRightIdx = 3*width/4 - 1 + min(find(uderThresholdWidth(3*width/4:width) < 0));
		
		int wStartRightIdx = width/4 - 1 + (int) MatrixUtils.max(
				MatrixUtils.findLess(
						MatrixUtils.part(uderThresholdWidth, width/4, 3*width/4 - 1), 
						0)
			);
		int wEndRightIdx = 3*width/4 - 1 + (int) MatrixUtils.min(
				MatrixUtils.findLess(
						MatrixUtils.part(uderThresholdWidth, 3*width/4, width - 1), 
						0)
			);
		
//		% split line betlween left and right images
//		wSplitIdx = 0.5*(wEndLeftIdx + wStartRightIdx);
//		
//		heightSumLeft = sum(coinImage(:, (1:wSplitIdx)), 2);
		
		int wSplitIdx = (int) (0.5d*(double)(wEndLeftIdx + wStartRightIdx));
		
		
		double[] heightSumLeft = MatrixUtils.sum(
						MatrixUtils.partCols(imgarr, 0, wSplitIdx), 
						2);
		
		// java-matlab img loaders compatibility
		heightSumLeft = MatrixUtils.multiplex(heightSumLeft, javaMatlabDiff);
//		MatrixUtils.print(heightSumLeft);
		
		
//		uderThresholdHeightLeft = heightSumLeft - middleMinThresould;

		double[] uderThresholdHeightLeft = MatrixUtils.minus(heightSumLeft, middleMinThresould);
		
//		MatrixUtils.print(uderThresholdHeightLeft);
//		MatrixUtils.print(MatrixUtils.part(uderThresholdHeightLeft, 0, heigth/2 - 1));
//		MatrixUtils.print(
//				MatrixUtils.findLess(
//						MatrixUtils.part(uderThresholdHeightLeft, 0, heigth/2 - 1), 
//						0)
//				);
		
//		hStartLeftIdx = max(find(uderThresholdHeightLeft(1:heigth/2) < 0));
//		hEndLeftIdx = heigth/2 - 1 + min(find(uderThresholdHeightLeft(heigth/2:heigth) < 0));
		
		int hStartLeftIdx = (int) MatrixUtils.min(
									MatrixUtils.findMore(
											MatrixUtils.part(uderThresholdHeightLeft, 0, heigth/2 - 1), 
											0)
								);
		int hEndLeftIdx = heigth/2 - 1 + (int) MatrixUtils.max(
				MatrixUtils.findMore(
						MatrixUtils.part(uderThresholdHeightLeft, heigth/2, heigth - 1), 
						0)
			);
		
//		System.out.println("hStartLeftIdx: " + hStartLeftIdx);
//		System.out.println("hEndLeftIdx: " + hEndLeftIdx);

		
//		heightSumRight = sum(coinImage(:, (wSplitIdx + 1:end)), 2);
//		uderThresholdHeightRight = heightSumRight - middleMinThresould;
		
		double[] heightSumRight = MatrixUtils.sum(
				MatrixUtils.partCols(imgarr, wSplitIdx, width), 
				2);

		// java-matlab img loaders compatibility
		heightSumRight = MatrixUtils.multiplex(heightSumRight, javaMatlabDiff);
		
		
		double[] uderThresholdHeightRight = MatrixUtils.minus(heightSumRight, middleMinThresould);

		
//		hStartRightIdx = max(find(uderThresholdHeightRight(1:heigth/2) < 0));
//		hEndRightIdx = heigth/2 - 1 + min(find(uderThresholdHeightRight(heigth/2:heigth) < 0));
		int hStartRightIdx = (int) MatrixUtils.min(
				MatrixUtils.findMore(
						MatrixUtils.part(uderThresholdHeightRight, 0, heigth/2 - 1), 
						0)
			);
		int hEndRightIdx = heigth/2 - 1 + (int) MatrixUtils.max(
				MatrixUtils.findMore(
						MatrixUtils.part(uderThresholdHeightRight, heigth/2, heigth - 1), 
						0)
			);

//		MatrixUtils.print(MatrixUtils.part(uderThresholdHeightRight, heigth/2, heigth - 1));
//		MatrixUtils.print(
//				MatrixUtils.findLess(
//						MatrixUtils.part(uderThresholdHeightRight, heigth/2, heigth - 1), 
//						0)
//				);
		
//		System.out.println("hStartRightIdx: " + hStartRightIdx);
//		System.out.println("hEndRightIdx: " + hEndRightIdx);
		
//		leftCenterX = ceil(0.5*(wStartLeftIdx + wEndLeftIdx));
//		leftCenterY = ceil(0.5*(hStartLeftIdx + hEndLeftIdx));
		
		int leftCenterX = (int) Math.ceil(0.5*(wStartLeftIdx + wEndLeftIdx));		
		int leftCenterY = (int) Math.ceil(0.5*(hStartLeftIdx + hEndLeftIdx));
		
//		rightCenterX = ceil(0.5*(wStartRightIdx + wEndRightIdx));
//		rightCenterY = ceil(0.5*(hStartRightIdx + hEndRightIdx));
		
		int rightCenterX = (int) Math.ceil(0.5*(wStartRightIdx + wEndRightIdx));
		int rightCenterY = (int) Math.ceil(0.5*(hStartRightIdx + hEndRightIdx));
		
		
//		leftWidth = ceil(wEndLeftIdx - wStartLeftIdx);
//		leftHeight = ceil(hEndLeftIdx - hStartLeftIdx);
//
//		rightWidth = ceil(wEndRightIdx - wStartRightIdx);
//		rightHeight = ceil(hEndRightIdx - hStartRightIdx);
		
		int leftWidth = (int) Math.ceil(wEndLeftIdx - wStartLeftIdx);
		int leftHeight = (int) Math.ceil(hEndLeftIdx - hStartLeftIdx);

		int rightWidth = (int) Math.ceil(wEndRightIdx - wStartRightIdx);
		int rightHeight = (int) Math.ceil(hEndRightIdx - hStartRightIdx);
		
		
		CoinImageInfo cii = new CoinImageInfo();
		cii.leftCenterX = leftCenterX;
		cii.leftCenterY = leftCenterY;
		cii.leftWidth = leftWidth;
		cii.leftHeight = leftHeight;
		
		cii.rightCenterX = rightCenterX;
		cii.rightCenterY = rightCenterY;
		cii.rightWidth = rightWidth;
		cii.rightHeight = rightHeight;
		
		
        return cii;
	}
	
}



