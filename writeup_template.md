#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output-images/dist.png "distribution"
[image2]: ./output-images/training_data_vis.png "training"
[image3]: ./output-images/training_data_prep.png "preprocessed"
[image4]: ./output-images/new_data_vis.png "newsigns"
[image5]: ./output-images/misclassified.png "misclassified"
[image6]: ./output-images/feature_map.png "featuremap"
[image7]: ./output-images/feature_map_notrain.png "featuremapnotrain"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yzhao380/trafficSignClassfier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is __34799__
* The size of the validation set is __4410__
* The size of test set is __12630__
* The shape of a traffic sign image is __(32, 32, 3)__
* The number of unique classes/labels in the data set is __43__

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing how the data distributed proportional to three categories of training, validation and testing data.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the data to a smaller range because it reserves all the channels however balances the data set to reduce biases towards highly represented values.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

![alt text][image3]

I did not add more data to the the data set for I would like to see what it could lead to without much augmentation.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU                  | Activation, ouputs 28x28x6                    |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU                  | Activation, ouputs 10x10x16                    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| outputs 120       							|
| RELU and dropout      | Activation 				|
| Fully connected		| outputs  84  									|
| RELU and dropout      | Activation 				|
| Fully connected		| outputs  43  									|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer to do a sweeping of parameters to achieve the best possible training and validation accuracy without a large discrepancy to avoid overfitting. Finally it comes down to parameters with __learning rate__ 0.001, __number of epochs__ 20, __batch size__ 128, and __drop out rate__ 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of __0.994__
* validation set accuracy of __0.941__
* test set accuracy of __0.912__

* What architecture was chosen? Why did you believe it would be relevant to the traffic sign application?

Lenet-5 was the baseline architecture I tried. It was chosen because of the proven successful classification of MNIST dataset, which consists of images of the same size.

* What were some problems with the architecture?

The Lenet-5 had the purpose of training grayscale dataset and the number of unique data classes of 10. And it didn't enforce dropout rate to check on overfitting.

* How was the architecture adjusted and why was it adjusted?

The input channels had been adjusted to reflect a change for colored channels. And after each fully connected layers, part of the data were dropped out to avoid overfitting.

* Which parameters were tuned? How were they adjusted and why?

The learning rate, number of epochs, batch size and drop out rate were tuned for a best combination. Learning rate was tuned by starting from a significantly large number to a fine-tuned small number. It was apparent that a larger value could result in overshooting and thus being adjusted. Batch size was selected from as much as memory could handle to a small number. Drop out rate was chosen based on whether the overfitting could be reduced. And number of epochs was decided by the least amount of time to achieve the goal of validation set accuracy.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The final model could be trained in a timely fashion (less than 2 mins for training on GPU) and gave out a good generalization for testing data with a promising accuracy above 91%.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]

The fifth image might be difficult to classify because it did not appear in the training or validation set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep left      		| Keep left  									|
| No entry     			| No entry										|
| Turn left ahead	    | Turn left ahead					            |
| Keep right	        | Keep right				 				|
| (not specified)		| Children crossing      		|

![alt text][image5]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 5. And the fifth image which was not in the data set was not given the correct label in the first place.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the __Output Top 5 Softmax Probabilities For Each Image Found on the Web__ section.

For the five images, the model is totally sure that this is sign (probability of 0.99 at least), and the image does contain its prediction. Except for the last image, where it will never get it correct because of a manual setting of correct label. The five soft max probabilities for the five images are

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Keep left  									|
| .99     				| No entry										|
| 1.00					| Turn left ahead 						        |
| 1.00	      			| Keep right					 				|
| .99				    | Children crossing      	     				|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The visual output of the feature map for trained network, after first conv layer for the first German sign image (keep left) found on the web is shown below.

![alt text][image6]

The edges from the image are clearly captured by the first convolution layer. It gives hints of what the original image could be looking like.

The yet-to-train network has a vague output for the same layer.

![alt text][image7]

It does sometimes capture the edges but makes it blurred in the next step. The conv layer seems to randomly preamble about finding the possible features.
