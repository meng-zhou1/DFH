---
The demo for "Deep Forest Hashing for Image Retrieval"
---

###1.Meng Zhou, Xianhua Zeng, Aozhu Chen
  contact: damengmy@foxmail.com

###2.DFH is developed with Python 2.7, please make sure all the dependencies are installed
	```
	joblib
	psutil
	scikit-learn>=0.18.1
	scipy
	simplejson
	xgboost
	cPickle
	```
###3.Setup (if needed )
	
DFH needs pygco, the wrapper for the graph cuts package gco-v3.0
	
Under normal circumstances, you can run DFH directly. 
	
You'd better run it on Linux OS, we didn't test on Windows OS.
	
If you encounter problems about pygco, please re-install it.
We provide the packages in ./libs/gco-v3.0.zip ./libs/pygco.zip

- First,   unzip the files using unzip gco-v3.0.zip 
				     unzip pygco.zip
- Second,  move all the files in gco-v3.0 to pycgo/gco_source
- Third,   go to the folder pygco, run make all
	 

###4.Description:
demo.py:           the demo for DFH on MNIST dataset.

###5.More datasets:
	
Besides MNIST, CIFAR-10 and NUS-WIDE, we include 2 other different datasets to contact experiments. We validate our model DFH on UCI-datasets, LETTER and YEAST. LETTER consists of 16000 train data points and 4000 test data points with 16-dimensional features. YEAST consists of 1038 train data points and 446 test data points with 8-dimensional features. The hyperparameters settings are shown in Table 10. The MAP results are shown in Table 11. The results show that DFH can be efficiently extended to other types of datasets for information retrieval. 
	
	
<table>
    <tr>
        <td td><td >12-bits</td><td >24-bits</td> <td >32-bits</td><td >48-bits</td>  
    </tr>
    <tr>
        <td >LETTER</td><td > 0.670 </td> <td > 0.908 </td><td >0.932 </td><td > 0.942</td>  
    </tr>
    <tr>
        <td >YEAST</td><td > 0.577 </td> <td > 0.617 </td><td > 0.640 </td> <td > 0.643 </td>
    </tr>
</table>

###6.If you have any questions about this demo, please feel free to contact Meng Zhou (damengmy@foxmail.com)
    
