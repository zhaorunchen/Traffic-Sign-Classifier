# **Traffic Sign Recognition** 

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

[image1]: ./writeup_figures/dataset_dist.png "Visualization"
[image2]: ./writeup_figures/LeNet_V2.png "LeNet_V2"
[image3]: ./writeup_figures/augemented_dist.png "New dataset"
[image4]: ./writeup_figures/Top_k.png "Top k scores"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed. One can say that the distribution is not uniform for all traning, validating and testing data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color information may not be the necessary feature that the model needs to learn. (To do list: different color space such HLS, YUV should be tested.)

As a last step, I normalized (including de-mean) the image data because it helps in speed of training and performance.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried two different architectures.

The first model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 gray image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5		| 2x2 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 10x10x16 				|
| Fully connected		| input 400, output 120        					|
| RELU					|												|
| Fully connected		| input 120, output 84        					|
| RELU					|												|
| Dropout				| 50% keep										|
| Fully connected		| input 84, output 43        					|

Another architecture is shown in the picture below:

![alt text][image2]

1. 5x5 convolution (32x32x1 in, 28x28x6 out)

2. ReLU

3. 2x2 max pool (28x28x6 in, 14x14x6 out)

4. 5x5 convolution (14x14x6 in, 10x10x16 out)

5. ReLU

6. 2x2 max pool (10x10x16 in, 5x5x16 out)

7. 5x5 convolution (5x5x6 in, 1x1x400 out)

8. ReLu

9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)

10. Concatenate flattened layers to a single size-800 layer

11. Dropout layer

12. Fully connected layer (800 in, 43 out)
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model,

Optimizer: AdamOptimizer 

Batch size: 100

Number of epochs: 40

learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

For the first model, results were:
* validation set accuracy of 79%

For the second model, 
* validation set accuracy of 82%

The issue comes from the unbalanced dataset.

Therefore, we need to mitigate this by inserting more data to the original dataset. This approach is called data augementation.
The detials include randomly rotating, increasing or decreasing brightness, and so on.

Then the distribution of augemented dataset is shown below:

![alt text][image3]

For the first model, results become:
* validation set accuracy of 95%
* test set accuracy of 79%

For the second model, results become:
* validation set accuracy of 98%
* test set accuracy of 81%

When tuning the hyper-perameters, 
* smaller batch size tends to provide higher accuracy
* smaller learning rate tends to provide higher accuracy
* larger epoch tends to provide higher accuracy
* But as a trade-off, the training time is increasing.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The new German traffic signs can be seen in the code at cell 20.

The second image might be difficult to classify because it has relativly low brightness.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image4]

The ground truth is 1,22,35,15,37,18 for these 6 signs.

The first image has 58% probability to be 1, and the ground truth is also 1. However, even we correctly predict the result, the low probability may comes from smaller dataset of 1 in the augemented dataset.

By comparing the ground truth and the first prediction in top k predictions, we successfully classify all 6 signs, even the probability of correct calss is not 100%.



The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The disscussion is shown above. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


