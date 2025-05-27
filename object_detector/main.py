import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import argparse # Added import

def load_model(model_url):
    print(f"Attempting to load model from: {model_url}")
    try:
        model = hub.load(model_url)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_video(video_path, model, output_path=None, threshold=0.5): # Added output_path and threshold
    print(f"Processing video: {video_path} with model: {model}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video writer setup if output_path is provided
    video_writer = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # Define the codec and create VideoWriter object
        # MJPG is a common codec for .avi or .mp4, but might need adjustment based on desired output format
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for path {output_path}")
            # Proceed with display only if writer fails
            video_writer = None 


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis,...] 
        detections = model(input_tensor)
        
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        
        frame_h, frame_w, _ = frame.shape # Renamed for clarity
        # detection_threshold = 0.5 # Now passed as argument 'threshold'

        for i in range(boxes.shape[0]):
            if scores[i] >= threshold: # Use argument 'threshold'
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (xmin * frame_w, xmax * frame_w, 
                                              ymin * frame_h, ymax * frame_h)
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"Class: {classes[i]} Score: {scores[i]:.2f}" # Keep COCO class IDs for now
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if video_writer:
            video_writer.write(frame)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection on a video using TensorFlow Hub model.")
    parser.add_argument('-i', '--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-o', '--output_path', type=str, default=None, help="Optional path to save the processed video. (e.g., output.mp4 or output.avi)")
    parser.add_argument('-m', '--model_url', type=str, 
                        default="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2", 
                        help="URL or path to the TensorFlow Hub model.")
    parser.add_argument('-t', '--threshold', type=float, default=0.5, 
                        help="Confidence threshold for displaying detections (0.0 to 1.0).")
    
    args = parser.parse_args()

    loaded_model = load_model(args.model_url)

    if loaded_model:
        process_video(args.video_path, loaded_model, args.output_path, args.threshold)
    else:
        print("Model not loaded. Exiting.")
    
    print("Object detection script finished.")
