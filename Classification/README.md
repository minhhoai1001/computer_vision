# 1. Image Classification
Given below is a rough timeline of how the state-of-the-art models have improved over time. We have included only those models which are present in the Torchvision package.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Model_Timeline.png)

# 2. Model Comparison
So far we have discussed how we can use pre-trained models to perform image classification but one question that we have yet to answer is how do we decide which model to choose for a particular task. In this section we will compare the pre-trained models on the basis of the following criteria:

1. **Top-1 Error**: A top-1 error occurs if the class predicted by a model with highest confidence is not the same as the true class.
2. **Top-5 Error**: A top-5 error occurs when the true class is not among the top 5 classes predicted by a model (sorted in terms of confidence).
3. **Inference Time on CPU**: Inference time is the time taken for model inference step.
4. **Inference Time on GPU**
5. **Model size**: Here size stands for the physical space occupied by the .pth file of the pre-trained model supplied by PyTorch

A **good** model will have low **Top-1 error**, **low Top-5 error**, **low inference time** on CPU and GPU and **low model size**.

All the experiments were performed on the same input image and multiple times so that the average of all the results for a particular model can be taken for analysis. The experiments were performed on Google Colab. Now, let’s have a look at the results obtained.

## 2.1 Accuracy Comparison of Models
The first criterion we are going to discuss consists of Top-1 and Top-5 errors. Top-1 error refers to errors when the top predicted class is different from the ground truth. Since the problem is rather a difficult one, there is another error measure called Top-5 error. An prediction is classified as an error if none of the top-5 predicted classes are correct.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Accuracy-Comparison-of-Models.png)

Notice from the graph that both the errors follow a similar trend. AlexNet was the first attempt based on Deep Learning and there has been improvement in the error since then. Notable mentions are GoogLeNet, ResNet, VGGNet, ResNext.

## 2.2. Inference Time Comparison
Next, we will compare the models based on the time taken for model inference. One image was supplied to each model multiple times and the inference time for all the iterations was averaged. Similar process was performed for CPU and then for GPU on Google Colab. Even though there are some variations in the order, we can see that **SqueezeNet**, **ShuffleNet** and **ResNet-18** had a really low inference time, which is exactly what we want.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Model-Inference-Time-Comparison-on-CPU-ms-Lower-is-better-.png)

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Model-Inference-Time-Comparison-on-GPU-ms-Lower-is-better-.png)

## 2.3. Model Size Comparison
A lot of times when we are using a Deep Learning model on an android or iOS device, the model size becomes a deciding factor, sometimes even more important than accuracy. **SqueezeNet** has the minimum model size (5 MB), followed by **ShuffleNet V2** (6 MB) and **MobileNet V2** (14 MB). It’s obvious why these models are preferred in mobile apps utilizing deep learning.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Model-Size-Comparison.png)

## 2.4. Overall Comparison
We discussed about which model performed better on the basis of a particular criterion. We can squeeze all those important details in one bubble chart which we can then refer to for deciding which model to go for based on our requirements.

The x-coordinate we are using is Top-1 error (**lower is better**). The y-coordinate is the inference time on GPU in milliseconds (lower is better). The bubble size represents the model size (lower is better).

NOTE :
* Smaller Bubbles are better in terms of model size.
* Bubbles near the origin are better in terms of both Accuracy and Speed.

![](https://www.learnopencv.com/wp-content/uploads/2019/06/Pre-Trained-Model-Comparison.png)

# 3. Final Verdict
* It is clear from the above graph that **ResNet50** is the best model in terms of all three parameters ( small in size and closer to origin )
* **DenseNets** and **ResNext101** are expensive on inference time.
* **AlexNet** and **SqueezeNet** have pretty high error rate.