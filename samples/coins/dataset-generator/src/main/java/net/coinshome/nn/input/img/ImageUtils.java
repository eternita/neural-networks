package net.coinshome.nn.input.img;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;

import com.jhlabs.image.ContrastFilter;
import com.jhlabs.image.DoGFilter;
import com.jhlabs.image.InvertFilter;

public class ImageUtils {

	
	
	/**
	 * result[row][col] = image.getRGB(col, row);
	 * 
	 * @param image
	 * @return
	 */
	public static int[][] img2pixelArray(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		int[][] result = new int[height][width];
		
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				result[row][col] = image.getRGB(col, row);
			}
		}

		return result;
	}

	/**
	 * 
	 * -1 -> 255 -> 0
	 *  
	 * @param arr
	 */
	public static void mean(double[][] arr) {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				int a = ((int)(((byte)arr[row][col]) & 0xFF));
				arr[row][col] = Math.abs(a - 255);
			}
		}
		return;
	}	

	public static void hex2int(int[][] arr) {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				arr[row][col] = ((int)(((byte)arr[row][col]) & 0xFF));
			}
		}
		return;
	}	
	
	public static BufferedImage detectEdges(BufferedImage img) {
        
		if (null == img)
			return null;
		
        img = makeCompatible(img);
//        img = deepCopy(img);
        
//		System.out.println(img);
        
		DoGFilter f = new DoGFilter();
		f.setInvert(true);
		
		BufferedImage outputImg = f.filter(img, img);
		
		return outputImg;
	}

	public static BufferedImage invert(BufferedImage img) {
        
		if (null == img)
			return null;
		
		InvertFilter f = new InvertFilter();
		
		BufferedImage outputImg = deepCopy(img);
		f.filter(img, outputImg);
		
		return outputImg;
	}
		
	public static BufferedImage contrast(BufferedImage img, float contrast, float brightness) {
        
		if (null == img)
			return null;
		
		ContrastFilter f = new ContrastFilter();
		f.setBrightness(brightness);
		f.setContrast(contrast);
		
		BufferedImage outputImg = deepCopy(img);
		f.filter(img, outputImg);
		
		return outputImg;
	}
	
	
	/**
	 * sometimes there are different color models - and DoGFilter does not work
	 * the method fix it
	 *  
	 * http://stackoverflow.com/questions/2597872/bufferedimage-colormodel-in-java
	 * 
	 * @param img
	 * @return
	 */
	public static BufferedImage makeCompatible(BufferedImage img)
	{
		
        BufferedImage result = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_ARGB);
        
        Graphics2D gr = result.createGraphics();
//        gr.drawRenderedImage(img, new AffineTransform()); //or some other drawImage function
//        gr.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), null);
        gr.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), Color.WHITE, null);
        gr.dispose();
        return result;
	}
	
	
	/**
	 * because
	 * BufferedImage result = new BufferedImage(srcImg.getWidth(), srcImg.getHeight(), BufferedImage.TYPE_INT_ARGB);
	 * 
	 * create pink image (java bug)
	 * 
	 * @param bi
	 * @return
	 */
	public static BufferedImage deepCopy(BufferedImage bi) {
		 ColorModel cm = bi.getColorModel();
		 boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
		 WritableRaster raster = bi.copyData(null);
		 return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
	}		
	
	
	
	/**
	 * @param args
	 * @throws Exception
	 */
	public static BufferedImage convert2grayScale(BufferedImage srcImage)
			throws Exception {

//		BufferedImage diffOfGauImg = PatternUtils.detectEdges(srcImage);

		// convert to grayscale
		BufferedImage image = new BufferedImage(srcImage.getWidth(),
				srcImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY); 
		Graphics g = image.getGraphics();

//		g.drawImage(diffOfGauImg, 0, 0, null);
		g.drawImage(srcImage, 0, 0, null);
		g.dispose();

		return image;

	}
	
	public static void writeFrames(BufferedImage img, CoinImageInfo cii) {
        
		if (null == img)
			return;
        
		Graphics2D gr = (Graphics2D) img.getGraphics();
		
		gr.setColor(Color.RED);
		
		gr.drawRect(cii.leftCenterX - cii.leftWidth/2, 
				    cii.leftCenterY - cii.leftHeight/2, 
				    cii.leftWidth, 
				    cii.leftHeight
				    );
		
		gr.drawRect(cii.rightCenterX - cii.rightWidth/2, 
			    cii.rightCenterY - cii.rightHeight/2, 
			    cii.rightWidth, 
			    cii.rightHeight
			    );
		
		return;
	}
	
	public static BufferedImage extractCoinImage(BufferedImage img, CoinImageInfo cii) {
        
		if (null == img)
			return null;
        int spaceBetweenImgs = 2; // empty space between images
        
        BufferedImage i1 = new BufferedImage(cii.leftWidth + cii.rightWidth + spaceBetweenImgs, cii.leftHeight, img.getType());

        i1.getGraphics().setColor(Color.white);
        
        i1.getGraphics().fillRect(0, 0, cii.leftWidth + cii.rightWidth + spaceBetweenImgs, cii.leftHeight);

        i1.getGraphics().drawImage(img, 0, 0, cii.leftWidth, cii.leftHeight, 
        		                   cii.leftCenterX - cii.leftWidth/2,
        		                   cii.leftCenterY - cii.leftHeight/2,
        		                   cii.leftCenterX + cii.leftWidth/2,
        		                   cii.leftCenterY + cii.leftHeight/2,
        		                   null);

        i1.getGraphics().drawImage(img, 
        		cii.leftWidth + spaceBetweenImgs, 
        		0, 
        		cii.leftWidth + spaceBetweenImgs + cii.rightWidth, 
        		cii.rightHeight, 
                cii.rightCenterX - cii.rightWidth/2,
                cii.rightCenterY - cii.rightHeight/2,
                cii.rightCenterX + cii.rightWidth/2,
                cii.rightCenterY + cii.rightHeight/2,
                null);
        
		
		return i1;
	}

	
	public static BufferedImage scaleImage(BufferedImage image, int newWidth, int newHeight) 
	{
		if (null == image)
			return null;
		
		int scaledWidth = newWidth;
		int scaledHeight = newHeight;

		// Make sure the aspect ratio is maintained, so the image is not
		// skewed
		double scaledRatio = (double) scaledWidth / (double) scaledHeight;
		int imageWidth = image.getWidth(null);
		int imageHeight = image.getHeight(null);
					
		double imageRatio = (double) imageWidth / (double) imageHeight;
		if (scaledRatio < imageRatio) {
			scaledHeight = (int) (scaledWidth / imageRatio);
		} else {
			scaledWidth = (int) (scaledHeight * imageRatio);
		}
		

		// Draw the scaled image
		BufferedImage thumbImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics2D = thumbImage.createGraphics();
		graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		
		// write white background (image can be transparent)
		graphics2D.setColor(Color.WHITE);
		graphics2D.fillRect(0, 0, newWidth, newHeight);

		graphics2D.drawImage(image, (newWidth - scaledWidth)/2, (newHeight - scaledHeight)/2, scaledWidth, scaledHeight, null);
//		graphics2D.drawImage(image, (newWidth - imageWidth)/2, (newHeight - imageHeight)/2, imageWidth, imageHeight, null);

		return thumbImage;
	}
	
/*	
    public static byte[] composeImg(byte[] avImg, byte[] revImg)
    {
        BufferedImage img1 = null;
        BufferedImage img2 = null;
        try {
            img1 = ImageIO.read(new ByteArrayInputStream(avImg));
            img2 = ImageIO.read(new ByteArrayInputStream(revImg));
            
            int height = img1.getHeight();
            int spaceBetweenImgs = 5; // empty space between images
            if (img1.getHeight() < img2.getHeight())
            {
                height = img2.getHeight();
            }
            
            BufferedImage i1 = new BufferedImage(img1.getWidth() + spaceBetweenImgs + img2.getWidth(), height, img1.getType());
            i1.getGraphics().setColor(Color.white);
            i1.getGraphics().fillRect(0, 0, img1.getWidth() + spaceBetweenImgs + img2.getWidth(), height);
            i1.getGraphics().drawImage(img1, 0, 0, img1.getWidth(), img1.getHeight(), null);

            i1.getGraphics().drawImage(img2, img1.getWidth() + spaceBetweenImgs, 0, img2.getWidth(), img2.getHeight(), null);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write( i1, "jpg", baos);
            return baos.toByteArray();
            
        } catch (Exception e) {
            Logger.error(ImageUtils.class, e);
        }

        return null;
    }	
		
*/
}
