# Human Pose Estimation with Deep Learning
Human Pose estimation is an important problem that has enjoyed the attention of the Computer Vision community for the past few decades. It is a crucial step towards understanding people in images and videos. In this post, I write about the basics of Human Pose Estimation (2D) and review the literature on this topic. This post will also serve as a tutorial in Human Pose Estimation and can help you learn the basics.

# 1. What is Human Pose Estimation?
**Human Pose Estimation** is defined as the problem of localization of human joints (also known as keypoints - elbows, wrists, etc) in images or videos. It is also defined as the search for a specific pose in space of all articulated poses.

![](https://lh6.googleusercontent.com/mBbue8S4QV2QYnHHad_k8UNa-i3GdPtcerSpu2hYUcvKxgyMRIWsvVnT1qHFO0GWOzZO8bXYmuxjU8PSKB5WRzBJH6W0EPKcCZ0tTazpKcz6DGElojKrv3P9HAJRpnsetRflCP7I)

**2D Pose Estimation** - Estimate a 2D pose (x,y) coordinates for each joint from a RGB image.
**3D Pose Estimation** - Estimate a 3D pose (x,y,z) coordinates a RGB image.

![](https://nanonets.com/blog/content/images/2019/04/Screen-Shot-2019-04-11-at-5.17.56-PM.png)

Human Pose Estimation has some pretty cool applications and is heavily used in Action recognition, Animation, Gaming, etc. For example, a very popular Deep Learning app HomeCourt uses Pose Estimation to analyse Basketball player movements.

# 2 Why is it hard?
Strong articulations, small and barely visible joints, occlusions, clothing, and lighting changes make this a difficult problem.

![](https://nanonets.com/blog/content/images/2019/04/challenging.png)

# 3. Different approaches to 2D Human Pose Estimation

## 3.1 Classical approaches

- The classical approach to articulated pose estimation is using the pictorial structures framework. The basic idea here is to represent an object by a collection of "parts" arranged in a deformable configuration (not rigid). A "part" is an appearance template which is matched in an image. Springs show the spatial connections between parts. When parts are parameterized by pixel location and orientation, the resulting structure can model articulation which is very relevant in pose estimation. (A structured prediction task)

![](https://nanonets.com/blog/content/images/2019/04/deformable-1.png)

- The above method, however, comes with the limitation of having a pose model not depending on image data. As a result, research has focused on enriching the representational power of the models.

- **Deformable part models** - Yang and Ramanan use a mixture model of parts which expresses complex joint relationships. Deformable part models are a collection of templates arranged in a deformable configuration and each model has global template + part templates. These templates are matched for in an image to recognize/detect an object. The Part-based model can model articulations well. This is however achieved at the cost of limited expressiveness and does not take in global context into account.

## 3.2 Deep Learning based approaches
The classical pipeline has its limitations and Pose estimation has been greatly reshaped by CNNs. With the introduction of “DeepPose” by Toshev et al, research on human pose estimation began to shift from classic approaches to Deep Learning. Most of the recent pose estimation systems have universally adopted ConvNets as their main building block, largely replacing hand-crafted features and graphical models; this strategy has yielded drastic improvements on standard benchmarks.

In the next section, I’ll summarize a few papers in chronological order that represents the evolution of Human Pose Estimation starting with DeepPose from Google (This is not an exhaustive list, but a list of papers that I feel show the best progression/most significant ones per conference).

# 4 Papers covered
1. DeepPose

2. Efficient Object Localization Using Convolutional Networks

3. Convolutional Pose Machines

4. Human Pose Estimation with Iterative Error Feedback

5. Stacked Hourglass Networks for Human Pose Estimation

6. Simple Baselines for Human Pose Estimation and Tracking

7. Deep High-Resolution Representation Learning for Human Pose Estimation
## 4.1 DeepPose: Human Pose Estimation via Deep Neural Networks (CVPR’14) [arXiv](https://arxiv.org/pdf/1312.4659.pdf) 

DeepPose was the first major paper that applied Deep Learning to Human pose estimation. It achieved SOTA performance and beat existing models. In this approach, pose estimation is formulated as a CNN-based regression problem towards body joints. They also use a cascade of such regressors to refine the pose estimates and get better estimates. One important thing this approach does is the reason about pose in a holistic fashion, i.e even if certain joints are hidden, they can be estimated if the pose is reasoned about holistically. The paper argues that CNNs naturally provide this sort of reasoning and demonstrate strong results.

## Model
The model consisted of an AlexNet backend (7 layers) with an extra final layer that outputs 2k joint coordinates - $\left(x_{i}, y_{i}\right) * 2$  where %i \in\{1,2 \ldots k\}% is the number of joints). The model is trained using a L2  loss for regression.

![](https://nanonets.com/blog/content/images/2019/04/deeppose-1.png)

An interesting idea this model implements is refinement of the predictions using cascaded regressors. Initial coarse pose is refined and a better estimate is achieved. Images are cropped around the predicted joint and fed to the next stage, in this way the subsequent pose regressors see higher resolution images and thus learn features for finer scales which ultimately leads to higher precision.

![](https://nanonets.com/blog/content/images/2019/04/deeppose-2.png)

## Results
PCP is used on [LSP (Leeds sports dataset)](http://sam.johnson.io/research/lsp.html) and [FLIC (Frames Labeled In Cinema)](https://bensapp.github.io/flic-dataset.html). Have a look at the appendix to find the definitions of some of the popular evaluation metrics like PCP & PCK.

## Comments
- This paper applied Deep Learning (CNN) to Human Pose Estimation and pretty much kicked off research in this direction.

- Regressing to XY locations is difficult and adds learning complexity which weakens generalization and hence performs poorly in certain regions.

- Recent SOTA methods transform the problem to estimating $K$ heatmaps of size $W_0 × H_0 , \{H_1, H_2, . . . , H_k\}$, where each $H_k$ heatmap indicates the location confidence of the kth keypoint. (K keypoints in total). The next paper was fundamental in introducing this idea.

## 4.2 Efficient Object Localization Using Convolutional Networks (CVPR’15) [arXiv](https://arxiv.org/pdf/1411.4280.pdf)

This approach generates heatmaps by running an image through multiple resolution banks in parallel to simultaneously capture features at a variety of scales. The output is a discrete heatmap instead of continuous regression. A heatmap predicts the probability of the joint occurring at each pixel. This output model is very successful and a lot of the papers that followed predict heatmaps instead of direct regression.

![](https://lh6.googleusercontent.com/dv9cWpHBRofQdfXSMYyHt-37LUowHBvRz3IGJT1tziXha33txCOded4RGy3JIHyCBj9JDlFPexCIL60M2G8WpgGa5LiZCJWl8W8KFlSzsVToZLvngB503fBcpshYrfZ5BwvMul8g)

## Model
A multi-resolution CNN architecture (coarse heatmap model) is used to implement a sliding window detector to produce a coarse heatmap output.

The main motivation of this paper is to recover the spatial accuracy lost due to pooling in the initial model. They do this by using an additional ‘pose refinement’ ConvNet that refines the localization result of the coarse heat-map. However, unlike a standard cascade of models, they reuse existing convolution features. This not only reduces the number of trainable parameters in the cascade but also acts as a regulariser for the coarse heat-map model since the coarse and fine models are trained jointly.

In essence, the model consists of the heat-map-based parts model for coarse localization, a module to sample and crop the convolution features at a specified (x,y) location for each joint, as well as an additional convolutional model for fine-tuning.

![](https://lh4.googleusercontent.com/dJLCAw2hF2Cb4LdbmFuVnCMWAzrzXhfqvrJwnNwXFNlJySFHZio6a6WdYGiUM9w2CgdRXx9PWBtSSLjgtpvCugYBEySOhAd4VM_O8bQrAzsQC2R7yrUJD510LYmhScEoluYADsX_)

A critical feature of this method is the joint use of a ConvNet and a graphical model. The graphical model learns typical spatial relationships between joints.

## Training
The model is trained by minimizing the Mean Squared-Error (MSE) distance of our predicted heat-map to a target heat-map (The target is a 2D Gaussian of constant variance (σ ≈ 1.5 pixels) centered at the ground-truth (x,y) joint location)

## Results

![](https://lh4.googleusercontent.com/Q01Pza38hKkgs4K0wjDwJJlook4uC0GiXEDEe9BeTWW0vZaPauFSNj-EKqMm9K7yEyzaF0wK6_BwC7PaqIiSwO0k3vPDSr1cR-eVAFJYSm3yDxzMwsWFgvvlqvAOeXOlXFGdN5jv)

## Comments
- Heatmaps work better than direct joint regression
- Joint use of a CNN and Graphical Model
- However, these methods lack structure modelling. The space of 2D human poses is highly structured because of body part proportions, left-right symmetries, interpenetration constraints, joint limits (e.g. elbows do not bend back) and physical connectivity (e.g. wrists are rigidly related to elbows), among others. Modelling this structure should make it easier to pinpoint the visible keypoints and make it possible to estimate the occluded ones. The next few papers tackle this, in their own novel ways.

## 4.3 Convolutional Pose Machines (CVPR’16) [arXiv](https://arxiv.org/pdf/1602.00134.pdf) [code](https://github.com/shihenw/convolutional-pose-machines-release)

## Summary
- This is an interesting paper that uses something called a Pose machine. A pose machine consists of an image feature computation module followed by a prediction module. Convolutional Pose Machines are completely differentiable and their multi-stage architecture can be trained end to end. They provide a sequential prediction framework for learning rich implicit spatial models and work very well for Human pose.
- One of the main motivations of this paper is to learn long range spatial relationships and they show that this can be achieved by using larger receptive fields.

## Model

![](https://nanonets.com/blog/content/images/2019/04/CPM-1.png)

g1() and g2() predict heatmaps (belief maps in the paper). Above is a high level view. Stage 1 is the image feature computation module, and Stage 2 is the prediction module. Below is a detailed architecture. Notice how the receptive fields increase in size?

![](https://lh6.googleusercontent.com/Yw8gJBCuaBGHF8PzcmoSb2Q4CpTz58u0bQohxfY8eBTKnLulGR9dRg-dZDhVpjH4-Dt4mMqtBwQjDwFZt69af-ZBirXJOChEFrjTrxoPvf0xlR_X_Lr3lGqfWDoV9O0DLA4RyCA1)

A CPM can consist of > 2 Stages, and the number of stages is a hyperparameter. (Usually = 3). Stage 1 is fixed and stages > 2 are just repetitions of Stage 2. Stage 2 take heatmaps and image evidence as input. The input heatmaps add spatial context for the next stage. (Has been discussed in detail in the paper).

On a high level, the CPM refines the heatmaps through subsequent stages.

The paper used intermediate supervision after each stage to avoid the problem of vanishing gradients, which is a common problem for deep multi-stage networks.

![](https://lh3.googleusercontent.com/Nc154T90KPadBVSMINwTEjT7lq79sJfT0mYkhYiS0EnQDAVuaRlFop-tm_vnipiLC9A7T_xtKectIfO10XkR_MVWwZkd-0-AwaB83wh_KAc3c-AhxcgqNTlt96K_uXB9eoK-4g0n)

## Results
[**MPII**](http://human-pose.mpi-inf.mpg.de/):  PCKh-0.5 score achieves state of the art at 87.95%, which is 6.11% higher than the closest competitor, and it is noteworthy that on the ankle (the most challenging part), our PCKh@0.5 score is 78.28% which is 10.76% higher than the closest competitor.

**LSP**: Model achieves state of the art at 84.32% (90.5% when adding MPII training data).
## Comments
Introduced a novel CPM framework that showed SOTA performance of MPII, FLIC and LSP datasets.

## 4.4 Human Pose Estimation with Iterative Error Feedback (CVPR’16) [arXiv](https://arxiv.org/pdf/1507.06550) [code](https://github.com/pulkitag/ief)
## Summary
This is a pretty dense paper and I’ve tried to summarize it briefly without leaving out too much. The overall working is pretty straightforward: Predict what is wrong with the current estimates and correct them iteratively. To quote the authors, Instead of directly predicting the outputs in one go, they use a self-correcting model that progressively changes an initial solution by feeding back error predictions, and this process is called Iterative Error Feedback (IEF).

Let’s jump right to the model pipeline.

- The input consists of the image I and a representation of the previous output 
y_t−1. Keep in mind this is an iterative process and the same output is refined over steps.
- Input, x_t=I⊕g(y_t−1) where I is the image and yt−1 is the previous output 
    - f(x_t) outputs the correction εt and this added to the current output yt to generate yt+1, which takes the corrections into account.
    - g(yt+1) converts each keypoint in yt+1 into a heatmap channel so they can be stacked to the image I, so as to form the input for the next teration. This process is repeated T times till we get a refined yt+1 and is brought closer to the ground truth by the addition of ε_t.

![](https://lh4.googleusercontent.com/a6_rmwVDv5D29zTm7RbtGXiKmD8TEDBCoEGUOm_lvL5tH9E4ZnZiC0wWtlD7a2XcXJtlrjWzWt0AhfmDHWfYRwJ4j3IYzAu-s1Sp848-Ii34wj1-Htj86xMOZ8LrOsa3N8mgPoMz)

## Example

![](https://lh4.googleusercontent.com/_qqpyGbHwEDfhg6frIvyKp-KduhZQd1awOSLQ_gOpWDPUnWd-VlBdq5NDlTyEpyB70PylaEruoxYkFHbreXZDYW8yHsfMaAeAbdR0NZl15V0Cns_pX_0A_dGxFI_JjBTbSD6_HBX)

As you can see, the pose is refined over the correction steps.

## Results

![](https://lh5.googleusercontent.com/pSSSe9TruCcFa5oe42WXdfjeNPE2pRMJiVD9FVtlY3ahLo8Qi78j9GO_Mk6joU0Ft0dKNfKktewM9fzEq2xKGWX2LxYg69pyGgKxQakxEJEfMxRK3pqJUaC7Mu4jVyIvX50SdrTu)

## Comments
An elegant paper that introduces a good novelty and works very well.

## 4.5 Simple Baselines for Human Pose Estimation and Tracking (ECCV’18) [paper](https://arxiv.org/pdf/1804.06208) [code](https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/pose_estimation/train.py)
The previous approaches work very well but are complex. This work follows the question – how good could a simple method be? And achieved the state-of-the-art at mAP of 73.7% on COCO.

The network structure is quite simple and consists of a ResNet + few deconvolutional layers at the end. (Probably the simplest way to estimate heat maps)

While the hourglass network uses upsampling to increase the feature map resolution and puts convolutional parameters in other blocks, this method combines them as deconvolutional layers in a very simple way. It was quite surprising to see such a simple architecture perform better than one with skip connections that preserve the information for each resolution.

![](https://nanonets.com/blog/content/images/2019/04/baseline-1.png)

Mean Squared Error (MSE) is used as the loss between the predicted heatmaps and targeted heatmaps. The targeted heatmap H^k for joint k is generated by applying a 2D Gaussian centered on the kth joint’s ground truth location with std dev = 1 pixel.

## Results

![](https://lh3.googleusercontent.com/ctRO7DJnFs2M2s4yH1jgJEeSYYMmrsFfltMlcnlJge0CaGhpMCn2UA_vKSZcBjsuRu5OsmGOWUxhOYW9jpCRkWtAtW0wXnmAq4OwBoWeNy9riVQtM1a95VzLsl3k81_GzVJqG5av)

![](https://lh5.googleusercontent.com/uM7UAFXHfzDfVlgj5FXv3i1WYynLYpgEvDzVCkFDSLvQsF8i7oNAM58E1AZjG0bO7EcZS0COWu8d-f5Qy3wy-7Yx8662L0vOyboydv2O-frPb07J0TTahgVoo0dOUaRqUzf0wL9U)

## 4.6 Deep High-Resolution Representation Learning for Human Pose Estimation [HRNet] (CVPR’19) [arXiv](https://arxiv.org/pdf/1902.09212.pdf) [code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
The HRNet (High-Resolution Network) model has outperformed all existing methods on Keypoint Detection, Multi-Person Pose Estimation and Pose Estimation tasks in the COCO dataset and is the most recent. HRNet follows a very simple idea. Most of the previous papers went from a high → low → high-resolution representation. HRNet maintains a high-resolution representation throughout the whole process and this works very well.

![](https://nanonets.com/blog/content/images/2019/04/HRNet-1.png)

The architecture starts from a high-resolution subnetwork as the first stage, and gradually adds high-to-low resolution subnetworks one by one to form more stages and connect the multi-resolution subnetworks in parallel.

Repeated multi-scale fusions are conducted by exchanging information across parallel multi-resolution subnetworks over and over through the whole process.

Another pro is that this architecture does not use intermediate heatmap supervision, unlike the Stacked Hourglass.

Heatmaps are regressed using an MSE loss, similar to simple baselines. (add in article link)

## Results

![](https://lh5.googleusercontent.com/FD8ukgUUNgrgZOth9diqDUR2G-srg7AaaS-BBwvwOBowYSZFf0DueAEWSRQf0u5IcgTHc4A01Eff7vmh0g1VBNfnpIzVNHwPK1imhdExPFMILg7mQioXLa9ZBPhZjtY9XjzoFgHi)

# Appendix
## Common Evaluation Metrics

Evaluation metrics are needed to measure the performance of human pose estimation models.

**Percentage of Correct Parts - PCP**: A limb is considered detected  (a correct part) if the distance between the two predicted joint locations and the true limb joint locations is less than half of the limb length (Commonly denoted as PCP@0.5).

- It measures the detection rate of limbs. The con is that it penalizes shorter limbs more since shorter limbs have smaller thresholds.
- Higher the PCP, better the model.

**Percentage of Correct Key-points - PCK: A detected joint is considered correct if the distance between the predicted and the true joint is within a certain threshold. The threshold can either be:

- PCKh@0.5 is when the threshold = 50% of the head bone link
- PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter
- Sometimes 150 mm is taken as the threshold.
- Alleviates the shorter limb problem since shorter limbs have smaller torsos and head bone links.
- PCK is used for 2D and 3D (PCK3D). Again, the higher the better.

**Percentage of Detected Joints - PDJ**: A detected joint is considered correct if the distance between the predicted and the true joint is within a certain fraction of the torso diameter. PDJ@0.2 = distance between predicted and true joint < 0.2 * torso diameter.

**Object Keypoint Similarity (OKS) based mAP**: Commonly used in the COCO keypoints challenge.
