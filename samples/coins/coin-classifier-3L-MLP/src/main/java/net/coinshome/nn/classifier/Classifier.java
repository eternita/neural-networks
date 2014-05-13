package net.coinshome.nn.classifier;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class Classifier {

	
	private Matrix thetta1 = null;
	private Matrix thetta2 = null;

	public Classifier(String baseDir) {
		thetta1 = loadMatlabMatrix(baseDir + "THETTA1.mat", "Theta1");
		thetta2 = loadMatlabMatrix(baseDir + "THETTA2.mat", "Theta2");
		
//		System.out.println("thetta1");
//		MatrixUtils.printMatrix(thetta1.getArray());
//
//		System.out.println("thetta2");
//		MatrixUtils.printMatrix(thetta2.getArray());
	}

	public Map<Integer, Double> predict(BufferedImage img, int topPredictionAmount) {
        

		double[][] inputImage = img2pixelArrayGrayScale(img); 
//		System.out.println("X");
//		MatrixUtils.printMatrix(inputImage);
		
        double[] inputImageUnrolled = MatrixUtils.unrol(inputImage);
        
        
        double[] xArr = inputImageUnrolled;

//        System.out.println("X unrolled");
//		MatrixUtils.printMatrix(xArr);
        
		// add bias
        double[] xArrWithBias = new double[1 + xArr.length];
        xArrWithBias[0] = 1;
        System.arraycopy(xArr, 0, xArrWithBias, 1, xArr.length);
        

//		double da[][] = new double[1][xArrWithBias.length];
//		da[0] = xArrWithBias;
//		Matrix X = new Matrix(da);
		
		Matrix xMatrix = new Matrix(new double[][]{xArrWithBias});
		
		Matrix h1m = xMatrix.times(thetta1.transpose());
		
//		double[][] h1sigmoid = h1m.getArray();  
//		printMatrix(h1m.getArray());
		
		double[][] dh1 = h1m.getArray();
		dh1[0] = MatrixUtils.sigmoid(h1m.getArray()[0]);
//		System.out.println("h1");		
//		MatrixUtils.printMatrix(dh1[0]);
		
        double[] dh1Bias = new double[1 + dh1[0].length];
        dh1Bias[0] = 1;
        System.arraycopy(dh1[0], 0, dh1Bias, 1, dh1[0].length);
		double dh1Bias2[][] = new double[1][dh1Bias.length];
		dh1Bias2[0] = dh1Bias; 
		
        Matrix h1Biasm = new Matrix(dh1Bias2);
        
        Matrix h2m = h1Biasm.times(thetta2.transpose());

		double[][] dh2 = h2m.getArray();
		dh2[0] = MatrixUtils.sigmoid(h2m.getArray()[0]);
		
//		MatrixUtils.print(dh2[0]);
		
		Map<Integer, Double> predictMap = getTopProbabilities(dh2[0], topPredictionAmount);
		
		
		return predictMap;
	}
	
	
	/**
	 * 
	 * @param in
	 * @param topAmount
	 * @return
	 */
	private static Map<Integer, Double> getTopProbabilities(double[] in, int topAmount)
	{
		Map<Integer, Double> topMap = new HashMap<Integer, Double>();
		
		for (int j = 1; j <= topAmount; j++)
		{
			double maxValue = 0;
			int maxIdx = -1;
			for (int i = 0; i < in.length; i++)
			{
				if (in[i] > maxValue)
				{
					maxValue = in[i];
					maxIdx = i;
				}
			}
			topMap.put(maxIdx + 1, maxValue);
			
			in[maxIdx] = -1; // reset this value
		}
		
		
		return topMap;
	}
	
	/**
	 * 
	 * -1 -> 255 -> 0
	 *  
	 * @param arr
	 */
	private static void mean(double[][] arr) {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				int a = ((int)(((byte)arr[row][col]) & 0xFF));
				arr[row][col] = Math.abs(a - 255);
			}
		}
		return;
	}
	




	private static double[][] img2pixelArrayGrayScale(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
		double[][] result = new double[height][width];
		
		Raster raster = image.getData();
		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				result[row][col] = raster.getSampleDouble(col, row, 0);
			}
		}

        mean(result); 
		return result;
	}
	
	private Matrix loadMatlabMatrix(String filePath, String variableName) {
		MatFileReader mfr = null;
		Matrix thetta1 = null;
		try {
			mfr = new MatFileReader(filePath);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		if (null == mfr) 
			return null;
		
		MLDouble mld = (MLDouble) mfr.getMLArray(variableName);
		if (null == mld)
			return null;

		double[][] dArr = mld.getArray();
			
//			System.out.println(thetta1.length +" " + thetta1[0].length + " " + thetta1[0][0]);
		thetta1 = new Matrix(dArr);

		return thetta1;
	}
	

	


}
