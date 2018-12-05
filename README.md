
The demo for "Deep Forest Hashing for Image Retrieval"

1.Meng Zhou, Xianhua Zeng, Aozhu Chen
  contact: damengmy@foxmail.com

2.DFH is developed with Python 2.7, please make sure all the dependencies are installed
	joblib
	psutil
	scikit-learn>=0.18.1
	scipy
	simplejson
	xgboost
	cPickle
	
3.Setup (if needed )
	DFH needs pygco, the wrapper for the graph cuts package gco-v3.0
	Under normal circumstances, you can run DFH directly. 
	You'd better run it on Linux OS, we didn't test on Windows OS.
	If you encounter problems about pygco, please re-install it.
	We provide the packages in ./libs/gco-v3.0.zip ./libs/pygco.zip

	First,   unzip the files using unzip gco-v3.0.zip 
				     unzip pygco.zip
	Second,  move all the files in gco-v3.0 to pycgo/gco_source
	Third,   go to the folder pygco, run make all
	 

4.Description:
    demo.py:           the demo for DFH on MNIST dataset.


5.If you have any questions about this demo, please feel free to contact Meng Zhou (damengmy@foxmail.com)
    
