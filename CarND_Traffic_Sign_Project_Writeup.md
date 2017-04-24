# **Traffic Sign Recognition** 

Developer:  Raymond Leung
Date: April 24 2017

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)  -- completed
* Explore, summarize and visualize the data set -- completed
* Design, train and test a model architecture -- completed
* Use the model to make predictions on new images -- completed
* Analyze the softmax probabilities of the new images -- completed
* (Optional) visualize the Neural Network's State -- NOT DONE
* Summarize the results with a written report -- completed


[//]: # (Image References)

[image1]: ./distribution_graph.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./signs/manwork.jpg "Traffic Sign 1"
[image5]: ./signs/wildanimal.jpg "Traffic Sign 2"
[image6]: ./signs/eightykm.jpg "Traffic Sign 3"
[image7]: ./signs/gostraightorright.jpg "Traffic Sign 4"
[image8]: ./signs/stop.jpg "Traffic Sign 5"

## Rubric Points
---

### Writeup / README

My Project Code is in [project code](https://github.com/rleung888/self_driving_car_class2.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Commented out train_test_split function to split 20% of train data to become validation data.  When using the train data for validation, the validation accuracy is 93% without any normalization.   I am using the validation set from the download.

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Normalize the image

There are two methods I used to normalize the images.   There are quite a lot of images in the training set and validation set are too dark.   I use the skimage.exposure.adjust_gamma to get the image brighter.   I used the gamma value 0.5 before but it is too bright and only help me to get to 92% accuracy.  I choose a slight darker parameter 0.75 and it works better and at certain point, it can get to 94.2% accuracy.

The second methods I used is to get the image arrange range from (0,255) to (-1, 1) or even (-0.5, 05) for relu to perform.   I tried couple methods, 
	(X/122.5) - 1
	(X/255) - 0.5
	rescale_intesity((1.0 * X), in_range=(0,255)) 
	X - mean(X)/std(X)
all the them give me pretty much the same percentage, not much result.   Around 89% without adjust gamma.   So I just use the mean and std and seems more custom to this trainging data set. 



#### 2. Architecture of model.

I did not change any thing on the LeNet-5 solution except for changing the color depth to 3 and output label to 43.    

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, output 28x28x6 	    |
| Activitation			| Relu, output 28x28x6							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, same padding, output 10x10x16     |
| Activitation			| Relu, output 10x10x16        					|
| Max pooling			| 2x2 stride, output 5x5x16        				|
| Flatten				| Output 400									|
| Fully connected		| Output 120									|
| Activitation          | Relu, output 120                              |
| Dropout               | keep_prob=0.5, output 120                     |
| Fully connected       | Output 84                                     |
| Activitation          | Relu, output 84                               |
| Fully connected       | Output 43 (label)                             |


#### 3. Train the model. 

I keep the learning rate to 0.001 as in the previous template.   Changed to 0.0005 or even lower, don't see any improvement at all.
For batch size, I change it to 200 from 128, it does not have much effect.
The Epoch plays a more important role in the setting, I increase from 10 to 30.   Although the accuracy get to 92% at Epoch 10, it does not stablize at 93%-94% level until Epoch 26.  I guess it may stablized at 94% if I add 10 more epoch but the Train accuracy is already reach 100%, not much for it to improve.  

#### 4. Trained Result 
My final model results were:
* training set accuracy: 0.999
* validation set accuracy:  0.935
* test set accuracy of 0.922

 

###Test a Model on New Images

#### 1. Test Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Prediction Result 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Man Work      		| Man Work   									| 
| Wild Animal    		| Wile Animal 									|
| Speed limit (80km/h)	| Speed limit (20km/h)					 		|
| Go Straight or Right	| Go Straight or Right      					|
| Stop					| Speed limit (60km/h)			 				|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.  

#### 3. Probability and Prediction 

| Probability         	|     Prediction	        					| Actual Sign	|
|:---------------------:|:---------------------------------------------:|:-------------:| 
| 0.999        			| Man Work  									| 0.999         |
| 1.000    				| Wild Animal 									| 1.000         |
| 0.516					| Speed limit (20km/h)							| 0.484         |
| 0.999	      			| Go Straight or Right			 				| 0.999         |
| 0.536				    | Speed limit (60km/h)     						| 0.440         |

For the images that failed the perdiction, they have similar percentage as the first image.   The Speed limit (80km/h) sign prediction is 0.484, second to max prediction 0.516.  The Stop sign prediction is 0.440, second to max prediction 0.536.   I thing the improvement will be the color contrast and brightness tuning on the data.   

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


