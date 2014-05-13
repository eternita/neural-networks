package net.coinshome.nn;

import net.coinshome.nn.input.CoinNNInputPreparer;


public class Main {

	// setup proxy if you need
	static
	{
//		System.setProperty("http.proxyHost", "your.proxy.com");
//		System.setProperty("http.proxyPort", "8080");		
	}

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
//				"C:/Develop/_n4j-nn-data/dataset-30_400_200_x7_br",
				"C:/Develop/_chn-data/dataset-454_400_200-mexico_x7",
				400,
				200,
				false // true - do aslo dif of gau, false - grayscale only
				);
		preperer.prepare();
		
	}
		


}
