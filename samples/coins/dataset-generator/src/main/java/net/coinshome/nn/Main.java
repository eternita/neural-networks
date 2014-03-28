package net.coinshome.nn;

import net.coinshome.nn.input.CoinNNInputPreparer;


public class Main {


	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		prepareCoinNNInput();
		
	}

	private static void prepareCoinNNInput() throws Exception
	{
		CoinNNInputPreparer preperer = new CoinNNInputPreparer(
				"E:/nn4coins/dataset-all-400_200_gau",
//				"C:/Develop/src/pavlikovkskiy/chn/data/dataset-mexico-400_200",
//				"C:/Develop/src/pavlikovkskiy/chn/data/dataset-3_924_15_200_100_gau_ext_test",
				400,
				200,
				true // true - dif of gau, false - grayscale
				);
		preperer.prepare();
		
	}
		


}
