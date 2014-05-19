
To get started with these Neural Networks quickly do following:

1. Download dataset I used to develop these neural networks: http://st.coinshome.net/ml-dataset/dataset-30_400_200_x7.zip
   Extract the dataset archive to local hard drive.

2. Configure your Neural Network: 
	open train.m and set datasetDir = 'your-local-path/dataset-30_400_200_x7/'; 
	e.g. datasetDir = 'C:/temp/dataset-30_400_200_x7/'; 
	! keep forward slash at the end !

	Optionally, you may want to update NN configuration in config.m

3. Training.
	To train NN use train.m
	
4. Checking prediction accuracy on validation / test set.
   Run test.m to check accuracy on train / validation / test sets.
   Before running test.m setup / check configs inside test.m.
   
5. Apply it to your project: use different datasets, modify code, ...