# Overview
Human pose estimation from video plays a critical role in various applications such as [quantifying physical exercises](https://google.github.io/mediapipe/solutions/pose_classification.html), sign language recognition, and full-body gesture control. For example, it can form the basis for yoga, dance, and fitness applications. It can also enable the overlay of digital content and information on top of the physical world in augmented reality.

MediaPipe Pose is a ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks on the whole body (or 25 upper-body landmarks) from RGB video frames utilizing our [BlazePose](https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html) research that also powers the [ML Kit Pose Detection API](https://developers.google.com/ml-kit/vision/pose-detection). Current state-of-the-art approaches rely primarily on powerful desktop environments for inference, whereas our method achieves real-time performance on most modern mobile phones, desktops/laptops, in python and even on the web.

# ML Pipeline
The solution utilizes a two-step detector-tracker ML pipeline, proven to be effective in our MediaPipe Hands and MediaPipe Face Mesh solutions. Using a detector, the pipeline first locates the person/pose region-of-interest (ROI) within the frame. The tracker subsequently predicts the pose landmarks within the ROI using the ROI-cropped frame as input. Note that for video use cases the detector is invoked only as needed, i.e., for the very first frame and when the tracker could no longer identify body pose presence in the previous frame. For other frames the pipeline simply derives the ROI from the previous frame’s pose landmarks.

The pipeline is implemented as a MediaPipe graph that uses a pose landmark subgraph from the pose landmark module and renders using a dedicated pose renderer subgraph. The pose landmark subgraph internally uses a pose detection subgraph from the pose detection module.

Note: To visualize a graph, copy the graph and paste it into MediaPipe Visualizer. For more information on how to visualize its associated subgraphs, please see visualizer documentation.

# Models
## Person/pose Detection Model (BlazePose Detector)
The detector is inspired by our own lightweight BlazeFace model, used in MediaPipe Face Detection, as a proxy for a person detector. It explicitly predicts two additional virtual keypoints that firmly describe the human body center, rotation and scale as a circle. Inspired by [Leonardo’s Vitruvian man](https://en.wikipedia.org/wiki/Vitruvian_Man), we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints.

![](https://google.github.io/mediapipe/images/mobile/pose_tracking_detector_vitruvian_man.png)

## Pose Landmark Model (BlazePose GHUM 3D)
The landmark model in MediaPipe Pose comes in two versions: a full-body model that predicts the location of 33 pose landmarks (see figure below), and an upper-body version that only predicts the first 25. The latter may be more accurate than the former in scenarios where the lower-body parts are mostly out of view.

Please find more detail in the BlazePose Google AI Blog, this [paper](https://arxiv.org/abs/2006.10204) and the model card, and the attributes in each landmark below.

![](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

# Solution APIs
## Cross-platform Configuration Options
Naming style and availability may differ slightly across platforms/languages.

**STATIC_IMAGE_MODE**

If set to `false`, the solution treats the input images as a video stream. It will try to detect the most prominent person in the very first images, and upon a successful detection further localizes the pose landmarks. In subsequent images, it then simply tracks those landmarks without invoking another detection until it loses track, on reducing computation and latency. If set to `true`, person detection runs every input image, ideal for processing a batch of static, possibly unrelated, images. Default to `false`.

**UPPER_BODY_ONLY**

If set to `true`, the solution outputs only the 25 upper-body pose landmarks. Otherwise, it outputs the full set of 33 pose landmarks. Note that upper-body-only prediction may be more accurate for use cases where the lower-body parts are mostly out of view. Default to `false`.

**SMOOTH_LANDMARKS**

If set to `true`, the solution filters pose landmarks across different input images to reduce jitter, but ignored if static_image_mode is also set to `true`. Default to `true`.

**MIN_DETECTION_CONFIDENCE**

Minimum confidence value (`[0.0, 1.0]`) from the person-detection model for the detection to be considered successful. Default to `0.5`.

**MIN_TRACKING_CONFIDENCE**

Minimum confidence value (`[0.0, 1.0]`) from the landmark-tracking model for the pose landmarks to be considered tracked successfully, or otherwise person detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. Ignored if static_image_mode is `true`, where person detection simply runs on every image. Default to `0.5`.

## Output
Naming style may differ slightly across platforms/languages.

**POSE_LANDMARKS**
- A list of pose landmarks. Each lanmark consists of the following:

- `x` and `y`: Landmark coordinates normalized to `[0.0, 1.0]` by the image width and height respectively.
- `z`: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of `z` uses roughly the same scale as `x`.

    **Note**: `z` is predicted only in full-body mode, and should be discarded when `upper_body_only` is true.

- `visibility`: A value in `[0.0, 1.0]` indicating the likelihood of the landmark being visible (present and not occluded) in the image.

## Python Solution API
Please first follow general [instructions](https://google.github.io/mediapipe/getting_started/python.html) to install MediaPipe Python package, then learn more in the companion Python Colab and the following usage example.

Supported configuration options:

- `static_image_mode`
- `upper_body_only`
- `smooth_landmarks`
- `min_detection_confidence`
- `min_tracking_confidence`

## Desktop
Please first see general instructions for desktop on how to build MediaPipe examples.

**MAIN EXAMPLE**
- Running on CPU
    - Graph: mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
    - Target: mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
- Running on GPU
    - Graph: mediapipe/graphs/pose_tracking/pose_tracking_gpu.pbtxt
    - Target: mediapipe/examples/desktop/pose_tracking:pose_tracking_gpu

**UPPER-BODY ONLY**

- Running on CPU
    - Graph: mediapipe/graphs/pose_tracking/upper_body_pose_tracking_cpu.pbtxt
    - Target: mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_cpu
- Running on GPU
    - Graph: mediapipe/graphs/pose_tracking/upper_body_pose_tracking_gpu.pbtxt
    - Target: mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_gpu