Neural Network for Digit Recognizer competition on Kaggle.com

Details: https://www.kaggle.com/c/digit-recognizer

How to use:
1. download train ant test sets from kaggle.com and save to ./data folder
2. run train.m to train neural network (matrixes is stored in ./temp directory)
3. run predict.m to get prediction for test data. Output file prediction.csv is stored in ./temp directory

If you want to check prediction accuracy locally you can split train.csv on train and test sets (e.g. train.tr.csv & train.tst.csv)
and use labeled test set train.tst.csv for prediction testing.
