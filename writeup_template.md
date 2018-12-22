## Writeup Template
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. 
* Apply a distortion correction to raw images. 
* Apply a perspective transform to rectify binary image (“birds-eye view”). 
* Use color transforms and gradients to create a thresholded binary image. 
* Detect lane pixels and fit a polynomial expression to find the lane boundary. 
* Determine the curvature of the lane and vehicle position with respect to center. 
* Overlay the detected lane boundaries back onto the original image. 
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position in the video. 


[//]: # (Image References)


[detect_corners]: ./output_images/00_detect_corners.png "Detected corners"
[undistorted_chess]: ./output_images/01_undistorted_chess.png "Undistorted chess"
[undistorted_video_image]: ./output_images/02_undistorted_video_image.png "Undistorted image from video"
[staright_perpective_transform]: ./output_images/03_staright_perpective_transform.png "Straight line perpective transform"
[curved_perpective_transform]: ./output_images/04_curved_perpective_transform.png "curved line perspective transform"
[selected_layers]: ./output_images/05_selected_layers.png "the color chaneels"
[color_channels]: ./output_images/06_color_channels.png "Color channel transforms and sobel gradient"
[final_binary_image]: ./output_images/07_final_binary_image.png "color tranformed binary image"
[sliding_window]: ./output_images/08_sliding_window.png "sliding_window"
[adaptive_search]: ./output_images/09_adaptive_search.png "adaptive_search"
[r-curve]: ./output_images/10_formula1.png "r-curve"
[derivative]: ./output_images/11_formula2.png "1st and 2nd order derivative"
[final_rcurve]: ./output_images/12_formula3.png "final r curve"
[test_output_image]: ./output_images/13_test_output.png "final output"
[video1]: ./project_video_output.mp4 "output_Video"

### Project structure

* examples/project2.ipynb: Jupyter notebook with a step-by-step walkthrough of the different components of the pipeline
* camera_cal/: Folder containing a collection of chessboard images used for camera calibration and distortion correction
* test_images/: Folder containing a set of images for test purposes
* output_image/: readme_images: Directory to store images used within this README.md
* project_video.mp4: Video with dark road surfaces and non-uniform lighting conditions
* project_video_output.mp4: Resulting output on passing the project_video through the pipeline

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell number 2 to 6 of the IPython notebook located in "./examples/project2.ipynb" 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][detect_corners] <br/>
![alt text][undistorted_chess]


### Pipeline (single images)

#### 1. Distortion correction

To demonstrate this step, the distortion correction steps explained as above.
Below is one of the distortion corrected test image.
![alt text][undistorted_video_image]

#### 2. Perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in cell number 8 in the notebook `examples/project2.ipynb`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
# Points for the original image
src = np.float32([
    [210, 700],
    [570, 460], 
    [705, 460], 
    [1075, 700]
])
# Points for the new image
dst = np.float32([
    [400, 720],
    [400, 0], 
    [w-400, 0], 
    [w-400, 720]
])
```
Following the distortion correction, an undistorted image undergoes Perspective Transformation which warpes the image into a bird's eye view scene. This makes it easier to detect the lane lines (since they are relatively parallel) and measure their curvature.

* Firstly, we compute the transformation matrix by passing the src and dst points into cv2.getPerspectiveTransform. These points are determined empirically with the help of the suite of test images.
* Then, the undistorted image is warped by passing it into cv2.warpPerspective along with the transformation matrix

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The below image shows perpective transform for straight lines,

![alt text][staright_perpective_transform] <br/>

The below image shows perpective transform for curved lines,
![alt text][curved_perpective_transform]

#### 3. Color transforms, gradients or other methods to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cells #14 through #18 in `project2.ipynb`). 

Using OpenCV, we can pull out the individual channels from each color space, and see if it better isolates the laneline pixels
Clearly some of these channels perform better than others. The channels I selected are R channel of RGB, S channel of HLS, B-channel of LAB, L-channel of LUV. Then thresholding applied to each channel image using functions rgb_rthresh, hls_sthresh, lab_bthresh and luv_lthresh.

The below image shows all these channel layers of a warped image.
![alt text][selected_layers]<br/>

Sobel derivatives calulated using calc_sobel function.

Finally all the 4 channel layers and sobel layer combined to get the final thresholded binary image.

Below image shows each of the four thresholded channel layers and the sobel output layer.
![alt text][color_channels]<br/>

Combined thresholded binary image.
![alt text][final_binary_image]

#### 4. Lane Line detection: Sliding Window technique
We now have a warped, thresholded binary image where the pixels are either 0 or 1; 0 (black color) constitutes the unfiltered pixels and 1 (white color) represents the filtered pixels. The next step involves mapping out the lane lines and determining explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

The first technique employed to do so is: Peaks in Histogram & Sliding Windows

* We first take a histogram along all the columns in the lower half of the image. This involves adding up the pixel values along each column in the image. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. These are used as starting points for our search.

* From these starting points, we use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

The parameters used for the sliding window search are:

```python
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
```
The left and right lines have been identified and fit with a curved polynomial(2nd order) function.

The output of sliding window as below,

![alt text][sliding_window]

#### 5. Lane Line detection: Adaptive Search

Once we have successfully detected the two lane lines, for subsequent frames in a video, we search in a margin around the previous line position instead of performing a blind search.

Although the Peaks in Histogram and Sliding Windows technique does a reasonable job in detecting the lane line, it often fails when subject to non-uniform lighting conditions and discolouration. To combat this, a method that could perform adaptive thresholding over a smaller receptive field/window of the image was needed. The reasoning behind this approach was that performing adaptive thresholding over a smaller kernel would more effectively filter out our 'hot' pixels in varied conditions as opposed to trying to optimise a threshold value for the entire image.

Hence, a custom Adaptive Search technique was implemented to operate once a frame was successfully analysed and a pair of lane lines were polyfit through the Sliding Windows technique. This method:

* Follows along the trajectory of the previous polyfit lines and splits the image into a number of smaller windows.
* These windows are then iteratively passed into a function and their threshold values are computed as the mean of the pixel intensity values across the window. 
* Following this iterative thresholding process, the returned binary windows are stacked together to get a single large binary image with dimensions same as that of the input image
The code implemented in cells 25 and 26 of project2.ipynb

![alt text][adaptive_search]

#### 6. Radius of curvature of the lane and the position of the vehicle with respect to center.

Following this conversion, we can now compute the radius of curvature (see tutorial here) at any point x on the lane line represented by the function x = f(y) as follows:
![alt text][r-curve]

In the case of the second order polynomial above, the first and second derivatives are:
![alt text][derivative]

So, our equation for radius of curvature becomes:
![alt text][final_rcurve]

The code for steps implemented in cells 28 to 30 in project2.ipynb

#### 6. An example of result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #31 through #32 in my code in `project2.ipynb` in the function `final_pipeline()`.  Here is an example of my result on a test image:

![alt text][test_output_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [https://www.youtube.com/watch?v=7FStCCtXx9E](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a very tedious project which involved the tuning of several parameters by hand. With the traditional Computer Vision approach I was able to develop a strong intuition for what worked and why. I also learnt that the solutions developed through such approaches aren't very optimised and can be sensistive to the chosen parameters. As a result, I developed a strong appreciation for Deep Learning based approaches to Computer Vision. Although, they can appear as a black box at times Deep learning approaches avoid the need for fine-tuning these parameters, and are inherently more robust.

The challenges I encountered were almost exclusively due to non-uniform lighting conditions, shadows, discoloration and uneven road surfaces. Although, it wasn't difficult to select the thresholding parameters to successfully filter the lane pixels it was very time consuming. Furthermore, the two biggest problem with my pipeline that become evident in the harder challenge video are:

* Its inability to handle sharp turns and constantly changing slope of the road. This a direct consequence of the assumption made in Step 2 of the pipeline where the road in front of the vehicle is assumed to be relatively flat and straight(ish). This results in a 'static' perspective transformation matrix meaning that if the assumption doesn't hold true the lane lines will no longer be relatively parallel. As a result, it becomes a lot harder to assess the validity of the detected lane lines, because even if the lane lines in the warped image are not nearly parallel, they might still be valid lane lines.

* Poor lane lines detection in areas of glare/ severe shadows / combination of both. This results from the failure of the thresholding function to successfully filter out the lane pixels in these extreme lighting conditions

In the coming weeks, I aim to tackle these problems and improve the performance of the model on the challenge videos.