package net.coinshome.nn.input;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.imageio.ImageIO;

import net.coinshome.nn.input.img.CoinDetector;
import net.coinshome.nn.input.img.CoinImageInfo;
import net.coinshome.nn.input.img.ImageGenerator;
import net.coinshome.nn.input.img.ImageUtils;
import net.coinshome.nn.utils.CrawlerUtils;

/**
 * 
 * 
 *
 */
public class CoinNNInputPreparer {

	private static String IMG_INPUT_DIR = "C:/Develop/prod/dump_all/coinshome.net/prod/share/fileStorage/600_300";
	
	// root directory for output data (csv files and images)
	private String OUT_WORKING_DIR = null;

	private static String QUERY_EXPORT_FILENAME = "coin_db_export.csv";
	
	private String QUERY_EXPORT_FILE = null;

	private String COIN_ALL_CSV_FILE = null;
	private String TR_SET_CSV_FILE = null;
	private String CV_SET_CSV_FILE = null;
	private String TST_SET_CSV_FILE = null;
	
	private String NOT_DETECTED_IMG_DIR = null;
	private String COLOR_IMG_OUTPUT_DIR = null;
	private String GRAY_IMG_OUTPUT_DIR = null;
	private String GAU_IMG_OUTPUT_DIR = null;
	
	private static int MIN_DETECTED_IMAGES_FOR_COIN = 1;
	private static String CSV_SPLITER = ",";

	private int IMG_OUTPUT_WIDTH = -1;
	private int IMG_OUTPUT_HEIGHT = -1;
	
	private boolean diffOfGaussianOutput = false;
	
	public CoinNNInputPreparer(String outWorkingDir, int imgOutputWidth, int imgOutputHeight, boolean diffOfGaussianOutput)
	{
		this.diffOfGaussianOutput = diffOfGaussianOutput;
		IMG_OUTPUT_WIDTH = imgOutputWidth;
		IMG_OUTPUT_HEIGHT = imgOutputHeight;
		
		OUT_WORKING_DIR = outWorkingDir;

		QUERY_EXPORT_FILE = OUT_WORKING_DIR + "/" + QUERY_EXPORT_FILENAME;
		COIN_ALL_CSV_FILE = OUT_WORKING_DIR + "/coin.all.csv";
		TR_SET_CSV_FILE = OUT_WORKING_DIR + "/coin.tr.csv";
		CV_SET_CSV_FILE = OUT_WORKING_DIR + "/coin.cv.csv";
		TST_SET_CSV_FILE = OUT_WORKING_DIR + "/coin.tst.csv";
		
		NOT_DETECTED_IMG_DIR = OUT_WORKING_DIR + "/not-detected-images";
		COLOR_IMG_OUTPUT_DIR = OUT_WORKING_DIR + "/img_color";
		GRAY_IMG_OUTPUT_DIR = OUT_WORKING_DIR + "/img_grayscale";
		GAU_IMG_OUTPUT_DIR = OUT_WORKING_DIR + "/img_gau";
		
		{
			  File theDir = new File(GRAY_IMG_OUTPUT_DIR);
			  if (!theDir.exists()) 
			    theDir.mkdir();  
			  
			  theDir = new File(GAU_IMG_OUTPUT_DIR);
			  if (!theDir.exists()) 
			    theDir.mkdir();  
			  
			  theDir = new File(COLOR_IMG_OUTPUT_DIR);
			  if (!theDir.exists()) 
			    theDir.mkdir();  
			  
			  theDir = new File(NOT_DETECTED_IMG_DIR);
			  if (!theDir.exists()) 
			    theDir.mkdir();  
		}
		
	}
	
	public void prepare()
	{
//		Map<String, Long> coinIdxMap = getCoinIndexes();
		Map<String, Long> coinIdxMap = new LinkedHashMap<String, Long>();

//		System.out.println(coinIdxMap);
		Long processedCoinInstances = 0l;
		Long processedImages = 0l;
		Map<String, Long> counters = new LinkedHashMap<String, Long>();
    	counters.put("processedCoinInstances", processedCoinInstances);
    	counters.put("processedImages", processedImages);


		BufferedReader queryReader = null;
		String line = "";
		List<String> coinRecords = new ArrayList<String>(); // records with the same chnCoinId
		Set<String> coinIdsWithLowDetectedInstances = new HashSet<String>(); // coin ids which have low detected instances
		String chnCoinId = null;
		
		int totalCoinInstances = 0;
		try {

			queryReader = new BufferedReader(new FileReader(QUERY_EXPORT_FILE));
			while ((line = queryReader.readLine()) != null) {

				// use comma as separator
				String[] lineArr = line.split(CSV_SPLITER);
				String coinId = lineArr[0];
				
				if (null == chnCoinId)
				{
					// very first record
					coinRecords.clear();
					coinRecords.add(line);
				
				} else if (chnCoinId.equals(coinId)) {
					
					coinRecords.add(line);
					
				} else {
					// !chnCoinId.equals(coinId)
					// chnCoinId has changed
					
					// flush previous data
					List<String> ciRecords = getCIRecords(coinRecords, coinIdxMap, coinIdsWithLowDetectedInstances, counters);
					splitAndSave2csv(ciRecords, COIN_ALL_CSV_FILE, TR_SET_CSV_FILE, CV_SET_CSV_FILE, TST_SET_CSV_FILE);
			
					totalCoinInstances += coinRecords.size();
					// add line with new chnCoinId
					coinRecords.clear();
					coinRecords.add(line);
				}
				
				chnCoinId = coinId;
			} // while ((line = queryReader.readLine()) != null)
			
			// flush the last coin
			List<String> ciRecords = getCIRecords(coinRecords, coinIdxMap, coinIdsWithLowDetectedInstances, counters);
			splitAndSave2csv(ciRecords, COIN_ALL_CSV_FILE, TR_SET_CSV_FILE, CV_SET_CSV_FILE, TST_SET_CSV_FILE);
			totalCoinInstances += coinRecords.size();
			
			System.out.println("Total Coin Instances (with undetected): " + totalCoinInstances);
			

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
		
/*		System.out.println("Coin ids with a lot of detection errors.");
		for (String coinId : coinIdsWithLowDetectedInstances)
			System.out.println(coinId);
*/		
				
		processedCoinInstances = counters.get("processedCoinInstances");
		processedImages = counters.get("processedImages");
		System.out.println("Saved Coins " + coinIdxMap.size());
		System.out.println("Saved CoinInstances " + processedCoinInstances);
		System.out.println("Generaged Images " + processedImages);
		return;
	}
	
	/**
	 * 
	 * 
	 * @param coinRecords contains records with all coin instances for the coin (the same chnCoinId)
	 * @param coinIdxMap
	 * @return list with strings for CSV
	 */
	private List<String> getCIRecords(
			List<String> coinRecords, 
			Map<String, Long> coinIdxMap, 
			Set<String> coinIdsWithLowDetectedInstances,
			Map<String, Long> counters
			) {
		

		String coinId = null;
		List<String> ciRecoids = new ArrayList<String>();
		int notDetectedCount = 0;
		for (int i = 0; i < coinRecords.size(); i++)
		{
			String ciRecord = coinRecords.get(i);
			String[] lineArr = ciRecord.split(CSV_SPLITER);
			coinId = lineArr[0]; 
			String ciId = lineArr[1]; 
			String imgId = lineArr[2]; 
//			String coinIdxStr = "" + coinIdx;
			Set<String> generatedImgIndexes = null;
			File inImgFile = null;
			{ // start process images
				try
				{
					inImgFile = new File(IMG_INPUT_DIR, imgId + ".jpg");
					
			        BufferedImage img = ImageIO.read(new FileInputStream(inImgFile));
					
			        CoinImageInfo cii = CoinDetector.detect(img);
			        if (null == cii)
			        {
			        	System.out.println("Coin is not detected for image " + inImgFile.getCanonicalPath());
			            ImageIO.write(img, "jpg", new FileOutputStream(new File(NOT_DETECTED_IMG_DIR, imgId + ".jpg")));
			            notDetectedCount++;
			        } else {
						long coinIdx = getCoinIndex(coinIdxMap, coinId);
						
						
			        	if ("tr".equals(getInputSetType(i + 1))) // do artificial image generation for training set only
			        	{
				        	generatedImgIndexes = generateImagesAndIndexes(img, cii, coinIdx, i + 1, Integer.MAX_VALUE);
			        	} else {
				        	generatedImgIndexes = generateImagesAndIndexes(img, cii, coinIdx, i + 1, 1);
			        	}

			        	
			        	{ // update counters
							Long processedCoinInstances = counters.get("processedCoinInstances");
							Long processedImages = counters.get("processedImages");

				        	processedCoinInstances++;
				        	processedImages += generatedImgIndexes.size();
				        	
				        	counters.put("processedCoinInstances", processedCoinInstances);
				        	counters.put("processedImages", processedImages);
			        	}
			        }
			        
				} catch (FileNotFoundException ex) {
		        	System.out.println(ex.getMessage());
		        	downloadFromSite(imgId, inImgFile);
//					ex.printStackTrace();
				} catch (Exception ex) {
					ex.printStackTrace();
				}
				
			} // end process images

	        
			if (null == generatedImgIndexes)
			{
				// coin was not detected or other error ...
				
			} else {
				long coinIdx = getCoinIndex(coinIdxMap, coinId);
				for (String imgIdx : generatedImgIndexes)
				{
					StringBuffer sb = new StringBuffer();
					sb.append(coinId)
						.append(CSV_SPLITER).append(ciId)
						.append(CSV_SPLITER).append(imgId)
						.append(CSV_SPLITER).append(imgIdx)
						.append(CSV_SPLITER).append(getInputSetType(i + 1))
						.append(CSV_SPLITER).append("" + coinIdx);
					
					ciRecoids.add(sb.toString());
					
				}
			}
	        
		} // for (String ciRecord : coinRecords)
		
		if (notDetectedCount > coinRecords.size() - MIN_DETECTED_IMAGES_FOR_COIN)
		{
			coinIdsWithLowDetectedInstances.add(coinId);
			System.out.println("Too many detection errors (" + notDetectedCount + ") for " + coinId + " which has " + coinRecords.size() + " images total");
		}
		
		if (notDetectedCount == coinRecords.size())
		{
			coinIdsWithLowDetectedInstances.add(coinId);
			System.out.println("No single image detected for " + coinId + " which has " + coinRecords.size() + " images total");
		}
		
		
		return ciRecoids;
	}
	
	private void downloadFromSite(String imgId, File inImgFile) {
		String imgUrl = "http://st.coinshome.net/fs/600_300/" + imgId + ".jpg";
		byte[] img = CrawlerUtils.downloadDataFromSite(imgUrl);
		FileOutputStream fos;
		try {
			fos = new FileOutputStream(inImgFile);
			fos.write(img);
			fos.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return;
	}

	private Set<String> generateImagesAndIndexes(BufferedImage img,
			CoinImageInfo cii, long coinIdx, int ciIdx, int maxAmount) {
		
		long maxCoinVariance = 100000; 
		long maxCIVariance = 1000;
		long completeImgIdx = maxCoinVariance*maxCIVariance*coinIdx + maxCIVariance*ciIdx + 0;
				
		Set<String> imgIndexes = new LinkedHashSet<String>();
		
		try {
			// dev only
//			File imgFile = new File(IMG_OUTPUT_DIR, "x_" + coinIdx + "_" + ciIdx + ".jpg");
//			ImageIO.write(img, "jpg", new FileOutputStream(imgFile));
			
			// extract coin image by coin image info
			BufferedImage extrCoinImg = ImageUtils.extractCoinImage(ImageUtils.deepCopy(img), cii.getFrame(2));
			
			// scale img - all output images should have the same dimension (width/height)
			extrCoinImg = ImageUtils.scaleImage(extrCoinImg, IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT);

//			// save original image (color) - just for convenience (not used in learning algorithm)
//			File origImgFile = new File(COLOR_IMG_OUTPUT_DIR, coinIdx + "_" + ciIdx + ".jpg");
//			ImageIO.write(extrCoinImg, "jpg", new FileOutputStream(origImgFile));
			
			List<BufferedImage> generatedImgs = ImageGenerator.generate(extrCoinImg, maxAmount);
			
		
			for (int imgIdx = 0; imgIdx < generatedImgs.size(); imgIdx++)
//			for (int imgIdx = 0; imgIdx < 1; imgIdx++)
			{
				BufferedImage genImg = generatedImgs.get(imgIdx);
				File genImgFile = null;

				completeImgIdx = maxCoinVariance*maxCIVariance*coinIdx + maxCIVariance*ciIdx + (imgIdx + 1);
				{
					// save original image (color)
					genImgFile = new File(COLOR_IMG_OUTPUT_DIR, completeImgIdx + ".jpg");
					ImageIO.write(genImg, "jpg", new FileOutputStream(genImgFile));
				}
				
				{// grayscale
					// scale img - all output images should have the same dimension (width/height)
//					genImg = ImageUtils.scaleImage(genImg, IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT);
					// convert to gray scale
					genImg = ImageUtils.convert2grayScale(genImg);
					
					genImgFile = new File(GRAY_IMG_OUTPUT_DIR, completeImgIdx + ".jpg");
					ImageIO.write(genImg, "jpg", new FileOutputStream(genImgFile));
				}
				

				// do the same for gau
				if (diffOfGaussianOutput)
				{
					genImg = generatedImgs.get(imgIdx);
					// convert to Dif of Gausian
					genImg = ImageUtils.detectEdges(genImg);
					// scale img - all output images should have the same dimension (width/height)
//					genImg = ImageUtils.scaleImage(genImg, IMG_OUTPUT_WIDTH, IMG_OUTPUT_HEIGHT);
					// convert to gray scale
					genImg = ImageUtils.convert2grayScale(genImg);
					
					genImgFile = new File(GAU_IMG_OUTPUT_DIR, completeImgIdx + ".jpg");
					ImageIO.write(genImg, "jpg", new FileOutputStream(genImgFile));
				}
				
				imgIndexes.add("" + completeImgIdx);
			}


		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		return imgIndexes;
	}

	/**
	 * tr / cv / tst
	 * 
	 * @param i - coin inst. index (counter)
	 * @return
	 */
	private String getInputSetType(int i)
	{
		return "tr";
/*				
		if (1 == i) return "tr";
		
		if (2 == i) return "cv";
		
		if (3 == i) return "tst";

		if (4 == i) return "cv";
		if (5 == i) return "tst";
		if (6 == i) return "tr";
		if (7 == i) return "cv";
		if (8 == i) return "tst";
		if (9 == i) return "cv";
		if (10 == i) return "tst";
		
//		if (4 == i) return "tr";
//		if (5 == i) return "tr";
//		if (6 == i) return "tr";
//		if (7 == i) return "cv";
//		if (8 == i) return "tst";
//		if (9 == i) return "tr";
//		if (10 == i) return "tr";
		
		if (i > 10)
			return getInputSetType(i - 10);
		else
			return "--"; // never occurs
//*/		
		
	}

	private long getCoinIndex(Map<String, Long> coinIdxMap, String coinId)
	{
		Long coinIdx = coinIdxMap.get(coinId);
				
		if (null == coinIdx)
		{
			coinIdx = (long) (coinIdxMap.size() + 1);
			coinIdxMap.put(coinId, coinIdx);
		}
		return coinIdx.longValue();
	}
	
	
	private void splitAndSave2csv(
			List<String> ciRecords, 
			String coinAllCSVFile, 
			String trainingSetFile, 
			String crossValidationSetFile,
			String testSetFile) {
		
		List<String> trRecords = new ArrayList<String>();
		List<String> cvRecords = new ArrayList<String>();
		List<String> tstRecords = new ArrayList<String>();
		
		for (String ciRecord : ciRecords)
		{
			String[] lineArr = ciRecord.split(CSV_SPLITER);
			String imgIdx = lineArr[3]; 
			String setType = lineArr[4]; 
			String coinIdx = lineArr[5];
			
			if ("tr".equals(setType))
			{
				trRecords.add(imgIdx + "," + coinIdx);
			} else if ("cv".equals(setType)) {
				cvRecords.add(imgIdx + "," + coinIdx);
			} else if ("tst".equals(setType)) {
				tstRecords.add(imgIdx + "," + coinIdx);
			} else {
				throw new RuntimeException("Wrong type: " + setType);
			}
			
		}
		
		save2csv(ciRecords, coinAllCSVFile);
		save2csv(trRecords, trainingSetFile);
		save2csv(cvRecords, crossValidationSetFile);
		save2csv(tstRecords, testSetFile);
		return;
	}
	
	/**
	 * Save list to a file
	 * 
	 * @param ciRecords
	 * @param coinAllCSVFile
	 */
	private void save2csv(List<String> ciRecords, String coinAllCSVFile) {

		BufferedWriter outWriter = null;
		try {
			outWriter = new BufferedWriter(new FileWriter(coinAllCSVFile, true));

			for (String s : ciRecords)
				outWriter.write(s + "\n");

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (outWriter != null) {
				try {
					outWriter.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return;
	}

	
}
