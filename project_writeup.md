## Project Writeup
### Vehicle detection and tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:
1) Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier.
2) Also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. For those first two steps, normalize features and randomize a selection for training and testing.
3) Implement a sliding-window technique and use trained classifier to search for vehicles in images.
4) Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: output_images/sample_images.png
[image2]: output_images/HOG_example_car.png
[image3]: output_images/HOG_example_notcar.png
[image4]: output_images/sliding_windows/test1_sliding_windows.png
[image5]: output_images/HOG_subsampling/test1_HOG_subsampling.png
[image6]: output_images/HOG_subsampling/test1_heatmap_result.png 

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the project write up file. The code is in the jupyter notebook file cat_detection.ipynb under the same folder.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

This is in the section 1 & 2 of the code. First I defined a function to load all .png images from the training library, then displayed a pair of sample images from each library (in section 1). <br />

![alt text][image1]  <br />

After this, I have used the knowledge and code learned from class to define functions of getting hog features, computing binned color features, and computing color histogram features (section 2). A pair of 2x sample HOG feature images are shown. Also available within output_image/ folder.  <br />

![alt text][image2]  <br />
![alt text][image3]  <br />

#### 2. Explain how you settled on your final choice of HOG parameters.

This is at the section 2). Having thsoe functions set up, I started to try many combinations of HOG features. Initially HOG was using RGB color, and having read some references and examples, I have switched to use YCrCb color. First channel 0 was selected, and I have switched to 'ALL' and use all of them, since from the sample HOG example, although the biggest difference can be seen on channel 0, other two channels can also provide help distiguishing car and notcar images. I have put this into the training model, and improved the training results about 2%, and accuracy was around 98%. However, when I was doing the following task I found that still some mis-detections happened all the time, and I came back and increased more orients (to 11), and doubled pixels per cell (to 16). This provides > 99% of accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is at the end of section 2, along with the previous part (HOG feature parameter adjustment). I used a linear SVC as recommended. Having changed those HOG feature parameters, I combined all of the features, such as HOG features, binned color features, and histogram features. Changed spatial size to (16,16), left hist bins as 32, and used 11 orientations, 16 pixels per cell, and 2 cells per block on HOG features. For training data, since car images and notcar images have very close lengths, I used all of them by concatenating them and randomly split, using 20% test. The final test accuracy of SVC is 99.04%. I have then picked up 30 samples, and 29 of them have right prediction. One was predicted as a car images but it is actually not. This can be found at the end of In[104] of the code. A summary of parameters used are shown below.

| parameter        | value | 
|:-------------:|:-------------:|
| color_space    |  YCrCb    | 
| orient    | 11    | 
| pix_per_cell   |  16    | 
| cell_per_block    |  2    | 
| hog_channel    |  ALL  | 
| spatial_size   |  (16,16)   | 
| hist_bins    |  32  | 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This is in the section 3) of the code. First, based on the example codes learned in class, I have defined functions to process single image, which is very similar to feature extraction and using the same color_space as above. Then using the slide window approach to predict each box and stack them together and show. To reach better result, first I have narrowed down the searching range, limiting the range between 375 and 600 on Y axis. This prevents some mis-detections. 

The second thing is to find a good overlap ratio. First I have tried from (0.6, 0.6), and I have seen when testing image "test3.img", no car was detected within box, while there is a white car there. So this cannot satisfy the requirement, and I have raised the overlap ratio to get more chances to find a car. However, as the ratio increases, the computation time increase sharply, which will make the procedure very slow if we use it on real-time video processing. A short summary is listed below, based mainly on "test3.img", which normally has a smaller box amount difference between car and background.

| overlap        | time (sec)  | result description |
|:-------------:|:-------------:|:-------------| 
| (0.6, 0.6)      |  1.46    | car not detected with box |
| (0.65, 0.65)    |  1.88    | car detected, but not much overlapped boxes |
| (0.70, 0.70)    |  2.44    | car detected, good overlap | 
| (0.75, 0.75)    |  3.59    | car detected, good overlap, a little longer time |
| (0.80, 0.80)    |  5.74    | more overlap, longer time |
| (0.85, 0.85)    |  10.44   | more overlap, much longer time |
| (0.90, 0.90)    |  23.35   | extremely more overlap, too long |

So overall, I think a good range between 0.70 to 0.80 is reasonable, and I used overlap ratio as (0.70, 0.70). After running each of the test images, I have output all results in the output_images/sliding_windows/ folder. A sample test image is shown below. <br />

![alt text][image4] <br />

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

This is in the section 4) of the code. As suggested in the requirement, a HOG sub-sampling approach has been implemented instead of the slow HOG calculation on each window. This is done based on learned from class, applying same HOG and color parameters as above. After this process, normally several boxes can be found in a car region, and 1 or no box was found in no-car region. 

To get better performance, first I have added the heat map onto the image, and set a threshold among the picture to filter out those false detections. This works very well on most cases, however, in some minor cases, i.e., in test image 3, there maybe about 2-3 boxes detected on the car, and in test image 5, even some no-car area will have 2-3 boxes stacked and >5 boxes on the car. Therefore, it is hard to set a unique threshold that easily works for all of them. To solve this problem, I have added more boxes on different scales and ranges (in code In[149]), and since different scale may have different boxes and detections, and cars will have much more boxes than non-car area, stacking them together provides a relatively larger margin to set the threshold value among all images. 

Moreover, I have also used the concept of normalization and applied similar threshold method here, I used the max() of an arbitrary threshold value and 1/3 of the largest value on the heatmap. In this case, for example, I can set threshold value to 1 as very small value, and if there is no car in this image, most of the single mis-detection can be filtered out, and if there are cars in the image, the largest value on the heatmap should be very high, and even some other mis-detections have larger value than 1 (i.e, mis-detected twice), it can still be filtered out since it is smaller than 1/3 of the biggest value.

All of the 6 test images after HOG subsampling and after heatmap processing are saved in output_images/HOG_subsampling/ folder, and a sample result is shown below. <br />

![alt text][image5] <br />
![alt text][image6] <br />

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's the [link to my video result](outputVideo8.mp4). Also it is under the root path of this repo.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This is in the section 5) of the code, basically going through the process of each image. Some of my thinking and revision of threshold approaches are shown above. Other than that, to keep capturing all the boxes, and filtering out some single error, I have used a list of length 10 to save the recent 10 frames. I stacked the boxes together and set a bigger thresold value to process it. The threshold value is set to max() of 1/3 of the biggest value, or 1+3/8*(length of history). This is to give a smaller value at the beginning of the video, otherwise the car is detected too late. Also the division is done by multiplying values and compare, to prevent any float->int issue and give more resolution. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As shown in the video, the result is still not perfect and some mis-detection happened, but I believe overall it is satisfied and can be done better via more tuning or advanced approaches. After tried many many times of video checking, from my case, the easiest part to fail is at about middle of it, and there is one white car a bit far in the image, and if the threshold is too big, it will lost tracking of it, and if the threshold value is too small, there are more mis-detection happened, especially during the region of tree shadow. I think practial, with information from localizaiton and positioning, it should be able to better identify stable items since they never move, and filtering out those items will make car detection better.

