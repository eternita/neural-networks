package net.coinshome.nn.input.img;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;


public class ImageGenerator {

    private static int MAX_ANGLE_DEVIATION = 10;
    private static int ANGLE_DEVIATION_STEP = 3;
    
    // total generated images: 2*8*3 

	public static List<BufferedImage> generate(BufferedImage srcImg)
			throws Exception {
		
		List<BufferedImage> outImages = new ArrayList<BufferedImage>(); 
		outImages.add(srcImg);
		
		int partWidth = srcImg.getWidth()/2;
		int partHeight = srcImg.getHeight();

		// extract left
        BufferedImage leftImg = new BufferedImage(partWidth, partHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D leftGr = leftImg.createGraphics();
        leftGr.drawImage(srcImg, 0, 0, partWidth, partHeight, 0, 0, partWidth, partHeight, null);
        
        // extract right
        BufferedImage rightImg = new BufferedImage(partWidth, partHeight, BufferedImage.TYPE_INT_ARGB);
        Graphics2D rightGr = rightImg.createGraphics();
        rightGr.drawImage(srcImg, 0, 0, partWidth, partHeight, partWidth, 0, 2*partWidth, partHeight, null);

        // compose multiple images using various rotations
        rotate(srcImg, leftImg, rightImg, outImages);
/*        
        { // swap
            List<BufferedImage> swapImages = swap(leftImg, rightImg);
            leftImg = swapImages.get(0);
            rightImg = swapImages.get(1);
            
            // compose parts to one image
            BufferedImage result = compose(leftImg, rightImg, ImageUtils.deepCopy(srcImg));
            outImages.add(result);
        }
        
        // rotate swaped
        // compose multiple images using various rotations
        rotate(srcImg, leftImg, rightImg, outImages);
//*/        
        
		
		return outImages;

	}
	
	private static void rotate(BufferedImage srcImg, BufferedImage leftImg, BufferedImage rightImg, List<BufferedImage> outImages)
	{
        for (int i = ANGLE_DEVIATION_STEP; i < MAX_ANGLE_DEVIATION; i = i+ANGLE_DEVIATION_STEP)
        {
//            outImages.add(
//            		compose(
//            				rotate(leftImg, i*Math.PI/180), 
//            				rightImg, 
//            				ImageUtils.deepCopy(srcImg)));
//            outImages.add(
//            		compose(
//            				rotate(leftImg, -i*Math.PI/180), 
//            				rightImg, 
//            				ImageUtils.deepCopy(srcImg)));
//            
//
//            outImages.add(
//            		compose(
//            				leftImg, 
//            				rotate(rightImg, i*Math.PI/180), 
//            				ImageUtils.deepCopy(srcImg)));
//            outImages.add(
//            		compose(
//            				leftImg, 
//            				rotate(rightImg, -i*Math.PI/180), 
//            				ImageUtils.deepCopy(srcImg)));
            
            
            outImages.add(
            		compose(
            				rotate(leftImg, i*Math.PI/180), 
            				rotate(rightImg, i*Math.PI/180), 
            				ImageUtils.deepCopy(srcImg)));
            
            outImages.add(
            		compose(
            				rotate(leftImg, -i*Math.PI/180), 
            				rotate(rightImg, -i*Math.PI/180), 
            				ImageUtils.deepCopy(srcImg)));
            
            
//            outImages.add(
//            		compose(
//            				rotate(leftImg, -i*Math.PI/180), 
//            				rotate(rightImg, i*Math.PI/180), 
//            				ImageUtils.deepCopy(srcImg)));
//            outImages.add(
//            		compose(
//            				rotate(leftImg, i*Math.PI/180), 
//            				rotate(rightImg, -i*Math.PI/180), 
//            				ImageUtils.deepCopy(srcImg)));
            
            
        } // for (int i = 0; i < MAX_ANGLE_DEVIATION; i++)
        
        return;
	}
	
	private static BufferedImage compose(BufferedImage leftImg, BufferedImage rightImg, BufferedImage result)
	{
//        BufferedImage result = deepCopy(srcImg); 
		int partWidth = result.getWidth()/2;
		int partHeight = result.getHeight();
        
        Graphics2D gr = result.createGraphics();
		gr.drawImage(leftImg, 0, 0, partWidth, partHeight, null);
		gr.drawImage(rightImg, partWidth, 0, partWidth, partHeight, Color.WHITE, null);
        gr.dispose();
        
        return result;
	}
	
	/**
	 * swap left <-> right
	 * 
	 * @param leftImg
	 * @param rightImg
	 * @return
	 */
	private static List<BufferedImage> swap(BufferedImage leftImg, BufferedImage rightImg)
	{
        BufferedImage tmp = leftImg;
        
        leftImg = rightImg;
        rightImg = tmp;
        
        List<BufferedImage> imgs = new ArrayList<BufferedImage>(2);
        imgs.add(leftImg);
        imgs.add(rightImg);
        
        return imgs;
	}

	private static BufferedImage rotate(BufferedImage inImg, double angle)
	{
		
        BufferedImage outImg = ImageUtils.deepCopy(inImg);
        
        Graphics2D gr = (Graphics2D)outImg.getGraphics();
        // rotate in center
        gr.rotate(angle, inImg.getWidth()/2, inImg.getWidth()/2); 
		
        gr.drawImage(inImg, 0, 0, inImg.getWidth(), inImg.getWidth(), null);
        
        gr.dispose();
        return outImg;
	}
	

	

	
}
