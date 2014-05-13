package net.coinshome.nn;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import net.coinshome.nn.classifier.Classifier;

/**
 * 
 * Run coin classification using trained neural network
 * How to run:
 * Setup/check DATASET_DIR - point to dataset you work with 
 * Setup/check TEST_IMAGE_FILE_NAME - image you would like to classify
 *  
 *
 */
public class Main {

	// dataset dir - used for loading thettas, images, csv file (for decoding hypothesis to chn_id)
	static String DATASET_DIR = "C:/Develop/src/pavlikovkskiy/chn/data/dataset-3_924_15_200_100_gau/";
	// file with test image - should be under  <DATASET_DIR>/img/test
	static String TEST_IMAGE_FILE_NAME = "100001001.jpg";
	
	static final int AMOUNT_OF_TOP_PREDICTION = 3; // amount of top predictions (by probability) returned by classifier
	private static String CSV_SPLITER = ","; // splitter for CSV file
	
	// show or not predicted coin in browser
	static boolean SHOW_PREDICTION_IN_BROWSER = false;
	// path to browser
	static String URL_VIEWER_CMD = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe";
//	static String URL_VIEWER_CMD = "C:/Program Files (x86)/Internet Explorer/iexplore.exe";
	

	static Classifier c = null;
    
    static Map<Integer, String> coinIdxCHNIxMap = null;
    
    static {
    	// initialization
    	
		System.out.println("Loading classifier (thettas) from dataset " + DATASET_DIR);
    	c = new Classifier(DATASET_DIR);
		System.out.println("Loading classifier complete ");
    	
		System.out.println("Loading coinIdxCHNIxMap ... ");
    	coinIdxCHNIxMap = loadCoinIdComvertationMap();
		System.out.println("Loading coinIdxCHNIxMap complete. \n "); // + coinIdxCHNIxMap
    	
    }

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		runCoinClassifier(
				TEST_IMAGE_FILE_NAME, 
				SHOW_PREDICTION_IN_BROWSER
				);
		
		return;
	}

	private static void runCoinClassifier(String inputFileName, boolean showInBrowser) throws Exception
	{

		File inImgFile = new File(DATASET_DIR + "img/test/", inputFileName);
		
        BufferedImage img = ImageIO.read(new FileInputStream(inImgFile));
		
        // do classification
		Map<Integer, Double> predictionsMap = c.predict(img, AMOUNT_OF_TOP_PREDICTION);
		
		for (Integer idx : predictionsMap.keySet())
			System.out.println("hypothesis / probability  : " + idx + " / " + predictionsMap.get(idx));
		
		
		int topHypothesis = predictionsMap.keySet().iterator().next();
		
		// decode coinIdx (prediction) to chn id 
		String chnCoinId = coinIdxCHNIxMap.get(topHypothesis);
		String chnCoinURL = "http://www.coinshome.net/coin_details.htm?id=" + chnCoinId; 
		
		System.out.println("File name: " + inputFileName + " -> hypothesis " + topHypothesis + ", chnId " + chnCoinId);
		System.out.println("URL " + chnCoinURL);
		
		// open in browser
		if (showInBrowser)
			Runtime.getRuntime().exec(URL_VIEWER_CMD + " " + chnCoinURL);
		
	}
	

	/**
	 * Load map based on coin.all.csv file from dataset
	 * 
	 * @return
	 */
	private static Map<Integer, String> loadCoinIdComvertationMap() {
		Map<Integer, String> coinIdxCHNIxMap = new HashMap<Integer, String>();

		BufferedReader queryReader = null;
		String line = "";
		
		try {

			queryReader = new BufferedReader(new FileReader(DATASET_DIR + "/" + "coin.all.csv"));
			while ((line = queryReader.readLine()) != null) {

				// use comma as separator
				String[] lineArr = line.split(CSV_SPLITER);
				String chnCoinId = lineArr[0];
				String coinIdx = lineArr[5];
				coinIdxCHNIxMap.put(Integer.parseInt(coinIdx), chnCoinId);
				
			} // while ((line = queryReader.readLine()) != null)
			
			
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (queryReader != null) {
				try {
					queryReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		
		return coinIdxCHNIxMap;
	}


}
