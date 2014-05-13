package net.coinshome.nn.utils;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;

public class CrawlerUtils {

	
	public static byte[] downloadDataFromSite(String url)
	{
		if (url.startsWith("//"))
		{
			url = "http:" + url;
		}
		
		ByteArrayOutputStream baos = new ByteArrayOutputStream(); 
		try {
			URL u = new URL(url);
			URLConnection uc = u.openConnection();
			InputStream is = uc.getInputStream();
			int bytesRead = 0;
            int bufferSize = 4000;
	         byte[] byteBuffer = new byte[bufferSize];				
	         while ((bytesRead = is.read(byteBuffer)) != -1) {
	             baos.write(byteBuffer, 0, bytesRead);
	         }
	         is.close();
//	         uc.c
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
//		byte[] r = null;
		
		
		return baos.toByteArray();
	}
	
	  
}
