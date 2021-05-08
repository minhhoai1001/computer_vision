# OpenCV Text Detection with EAST

In this tutorial, you will learn how to use OpenCV to detect text in images using the EAST text detector.

The EAST text detector requires that we are running OpenCV 3.4.2 or OpenCV 4 on our systems — if you do not already have OpenCV 3.4.2 or better installed, please refer to my [OpenCV install guides](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/) and follow the one for your respective operating system.

In the first part of today’s tutorial, I’ll discuss why detecting text in natural scene images can be so challenging.

From there I’ll briefly discuss the EAST text detector, why we use it, and what makes the algorithm so novel — I’ll also include links to the original paper so you can read up on the details if you are so inclined.

Finally, I’ll provide my Python + OpenCV text detection implementation so you can start applying text detection in your own applications.

# Why is natural scene text detection so challenging?

![](../images/opencv_text_detection_challenges.jpg)
*Figure 1: Examples of natural scene images where text detection is challenging due to lighting conditions, image quality, and non-planar objects (Figure 1 of [Mancas-Thillou and Gosselin](https://www.tcts.fpms.ac.be/publications/regpapers/2007/VS_cmtbg2007.pdf)).*

