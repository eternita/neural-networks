package net.coinshome.nn.input.img;

public class CoinImageInfo
{
	public int leftCenterX;
	public int leftCenterY;
	
	public int leftWidth;
	public int leftHeight;
	
	public int rightCenterX;
	public int rightCenterY;
	
	public int rightWidth;
	public int rightHeight;
	
	@Override
	public String toString() {
		return "CoinImageInfo [leftCenterX=" + leftCenterX + ", leftCenterY="
				+ leftCenterY + ", leftWidth=" + leftWidth + ", leftHeight="
				+ leftHeight + ", rightCenterX=" + rightCenterX
				+ ", rightCenterY=" + rightCenterY + ", rightWidth="
				+ rightWidth + ", rightHeight=" + rightHeight + "]";
	}	
	
	
	public CoinImageInfo getFrame(int border)
	{
		CoinImageInfo cii = new CoinImageInfo();
		cii.leftCenterX = leftCenterX;
		cii.leftCenterY = leftCenterY;
		cii.leftHeight = Math.max(leftHeight, rightHeight) + border;
		cii.leftWidth = Math.max(leftWidth, rightWidth) + border;
		
		cii.rightCenterX = rightCenterX;
		cii.rightCenterY = rightCenterY;
		cii.rightHeight = Math.max(leftHeight, rightHeight) + border;;
		cii.rightWidth = Math.max(leftWidth, rightWidth) + border; 
		
		return cii;
	}
	
	
}