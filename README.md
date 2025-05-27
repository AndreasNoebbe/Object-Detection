# Python Video Object Detector

This project provides a Python script for performing object detection on video files using pre-trained models from TensorFlow Hub.

## Features

- Detects objects in video files.
- Displays processed video with bounding boxes and labels for detected objects.
- Allows saving the processed video to an output file.
- Customizable object detection model (via TensorFlow Hub URL).
- Adjustable confidence threshold for detections.
- Command-line interface for easy operation.

## Prerequisites

- Python 3.7+
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL_HERE> 
    cd <YOUR_REPOSITORY_DIRECTORY_HERE> 
    ```
    *(Replace `<YOUR_REPOSITORY_URL_HERE>` with the actual URL you cloned from and `<YOUR_REPOSITORY_DIRECTORY_HERE>` with the name of the directory created by the clone command, usually the repository name).*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install OpenCV, TensorFlow, NumPy, and TensorFlow Hub. Depending on your system, TensorFlow installation might take some time and might require specific CPU/GPU considerations (e.g., CUDA for GPU support). Refer to the official TensorFlow installation guide for more details if you encounter issues.

## Usage

Run the object detection script from the root directory of the project:

```bash
python -m object_detector.main -i path/to/your/video.mp4 -o path/to/your/output_video.avi
```

### Command-Line Arguments:

-   `-i VIDEO_PATH`, `--video_path VIDEO_PATH`: **(Required)** Path to the input video file (e.g., `my_video.mp4`).
-   `-o OUTPUT_PATH`, `--output_path OUTPUT_PATH`: (Optional) Path to save the processed video (e.g., `processed_output.avi`). If not provided, the video will be displayed in real-time but not saved. The output format depends on the codec used (currently MJPG, common for `.avi`).
-   `-m MODEL_URL`, `--model_url MODEL_URL`: (Optional) URL or path to the TensorFlow Hub model.
    Default: `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2`
-   `-t THRESHOLD`, `--threshold THRESHOLD`: (Optional) Confidence threshold for displaying detections (a value between 0.0 and 1.0).
    Default: `0.5`

### Example:

To process a video named `cool_video.mp4` and save the output as `detected_video.avi` with a threshold of 0.6:

```bash
python -m object_detector.main --video_path cool_video.mp4 --output_path detected_video.avi --threshold 0.6
```

To process a video and just display it without saving:
```bash
python -m object_detector.main --video_path cool_video.mp4
```

**Note on Sample Video:** This project does not include a sample video. You will need to provide your own video file for processing. Many royalty-free video resources are available online (e.g., Pexels, Pixabay). You can place your video in the `data/` directory for organization, but you can point the script to a video file anywhere on your system.

## Model

The default model is SSD MobileNet v2, pre-trained on the COCO dataset, accessed via TensorFlow Hub. This model is chosen for its balance of performance and efficiency. You can experiment with other models available on TensorFlow Hub by providing their URL via the `--model_url` argument. Ensure the chosen model has a compatible signature for inference (i.e., accepts a batch of images and outputs detections in a recognizable format).

## Future Improvements (Potential)

-   Add mapping for COCO class IDs to human-readable names.
-   Support for different output video codecs.
-   Option to specify webcam input.
-   More sophisticated post-processing of detections (e.g., Non-Max Suppression if not handled by the model).
-   Batch processing of multiple video files.

## Testing

To run the unit tests for this project, navigate to the root directory and ensure your virtual environment is activated and dependencies (`requirements.txt`) are installed.

Then, run the following command:

```bash
python -m unittest discover tests
```

This command will automatically discover and execute all test cases defined in files starting with `test_` within the `tests` directory.

Note: Some tests, like `test_load_default_model`, require an active internet connection to download model files from TensorFlow Hub.
