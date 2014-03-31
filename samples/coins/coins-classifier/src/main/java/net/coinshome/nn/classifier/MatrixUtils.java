package net.coinshome.nn.classifier;

import java.io.IOException;
import java.io.OutputStream;

public class MatrixUtils {

	public MatrixUtils() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * 2D -> 1D
	 * 
	 * @param arr
	 * @return
	 */
	public static double[] unrol(double[][] arr) {
		double[] outarr = new double[arr.length * arr[0].length];

		for (int col = 0; col < arr[0].length; col++) {
			for (int row = 0; row < arr.length; row++) {
//				outarr[col * arr[0].length + row] = arr[row][col];
				try
				{
//					outarr[col * arr[0].length + row] = arr[row][col];
					outarr[col * arr.length + row] = arr[row][col];
				} catch (Exception ex)
				{
					ex.printStackTrace();
				}
				
				// System.out.println(row*arr.length + col);
			}
		}

		return outarr;
	}
	
	public static double[] sigmoid(double[] in)
	{
//		g = 1.0 ./ (1.0 + exp(-z));
		
		double[] out = new double[in.length];
		for (int i = 0; i < in.length; i++)
		{
			double d = in[i];
			d = 1 / (1 + Math.exp(-d));
			out[i] = d;
		}

		return out;
	}
	
	public static void print(int[][] arr) {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				System.out.print(arr[row][col] + " ");
			}
			System.out.println(" ");
		}
	}
	
	public static void print(double[][] arr) {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				System.out.print(arr[row][col] + " ");
			}
			System.out.println(" ");
		}
	}
	
	public static void printCSV(double[][] arr, OutputStream out) throws IOException {
		for (int row = 0; row < arr.length; row++) {
			for (int col = 0; col < arr[0].length; col++) {
				if (0 < col)
					out.write(",".getBytes());
					
				out.write(("" + arr[row][col]).getBytes()); 
			}
			out.write("\n".getBytes());
		}
	}
	
	public static void print(double[] arr) {
		System.out.println("");
		for (int row = 0; row < arr.length; row++) {
				System.out.print(arr[row] + " ");
		}
		System.out.println("");
	}	
	

	public static double[][] int2double(int[][] input)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = input[i][j];
		
		return output;
	}
	
	public static double[][] minus(double[][] input, double value)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = input[i][j] - value;
		
		return output;
	}
	
	public static double[] minus(double[] input, double value)
	{
		double[] output = new double[input.length];
		
		for (int i = 0; i < input.length; i++)
			output[i] = input[i] - value;
		
		return output;
	}

	public static double[][] plus(double[][] input, double value)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = input[i][j] + value;
		
		return output;
	}
	
	public static double[][] divide(double[][] input, double value)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = input[i][j] / value;
		
		return output;
	}
	
	public static double[][] multiplex(double[][] input, double value)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = input[i][j] * value;
		
		return output;
	}
	
	public static double[] multiplex(double[] input, double value)
	{
		double[] output = new double[input.length];
		
		for (int i = 0; i < input.length; i++)
			output[i] = input[i] * value;
		
		return output;
	}
	
	public static double[][] abs(double[][] input)
	{
		double[][] output = new double[input.length][input[0].length];
		
		for (int i = 0; i < input.length; i++)
			for (int j = 0; j < input[0].length; j++)
				output[i][j] = Math.abs(input[i][j]);
		
		return output;
	}

	public static double max(double[] input)
	{
		double max = Double.MIN_VALUE;
		for (int i = 0; i < input.length; i++)
			if (input[i] > max)
				max = input[i];
		
		return max;
	}

	public static double min(double[] input)
	{
		double min = Double.MAX_VALUE;
		for (int i = 0; i < input.length; i++)
			if (input[i] < min)
				min = input[i];
		
		return min;
	}

	
	public static double[] sum(double[][] input, int dimension)
	{
		
		if (1 == dimension)
		{
			int length = input[0].length;
			double[] sum = new double[length];
			// sum(..., 1)
			for (int j = 0; j < input[0].length; j++)
			{
				double value = 0;
				for (int i = 0; i < input.length; i++)
				{
					value += input[i][j]; 
				}
				
				sum[j] = value;
			}
			
			return sum;
			
		} else if (2 == dimension) {
			int length = input.length;
			double[] sum = new double[length];
			// sum(..., 2)
			for (int i = 0; i < input.length; i++)
			{
				double value = 0;
				for (int j = 0; j < input[0].length; j++)
				{
					value += input[i][j]; 
				}
				
				sum[i] = value;
			}
			
			return sum;
		} else {
			return null;
		}
		
	}
	
	
	public static double[] findLess(double[] input, double value)
	{
		int size = 0;
		for (int i = 0; i < input.length; i++)
			if (input[i] < value)
				size++;

		double[] output = new double[size];
		
		int idx = 0;
		for (int i = 0; i < input.length; i++)
		{
			if (input[i] < value)
			{
				output[idx++] = i;
			}
		}
		
		return output;
	}
	
	
	public static double[] findMore(double[] input, double value)
	{
		int size = 0;
		for (int i = 0; i < input.length; i++)
			if (input[i] > value)
				size++;

		double[] output = new double[size];
		
		int idx = 0;
		for (int i = 0; i < input.length; i++)
		{
			if (input[i] > value)
			{
				output[idx++] = i;
			}
		}
		
		return output;
	}
	
	
	public static double[] part(double[] input, int startIdx, int endIdx)
	{
		int length = endIdx - startIdx;

		double[] output = new double[length];
		
		System.arraycopy(input, startIdx, output, 0, length);

		return output;
	}

	public static double[][] partCols(double[][] input, int startIdx, int endIdx)
	{
		int length = endIdx - startIdx;

		double[][] output = new double[input.length][length];
		for (int i = 0; i < input.length; i++)
			System.arraycopy(input[i], startIdx, output[i], 0, length);
		
		return output;
	}
	
}
