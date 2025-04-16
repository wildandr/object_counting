import os
import matplotlib.pyplot as plt
from typing import Callable, Generator, Optional, Tuple
from tqdm import tqdm
import logging
import cv2  # Add cv2 import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

HOME = os.getcwd()
print(HOME)

# Fix video path construction
SOURCE_VIDEO_PATH = "/Users/owwl/Downloads/object_counting/video5.mp4"
TARGET_VIDEO_PATH = os.path.join(os.path.dirname(SOURCE_VIDEO_PATH), "video5_output.mp4")

# Add path checks
if not os.path.exists(SOURCE_VIDEO_PATH):
    raise FileNotFoundError(f"Source video not found: {SOURCE_VIDEO_PATH}")

# Ensure output directory exists
output_dir = os.path.dirname(TARGET_VIDEO_PATH)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logging.info(f"Created output directory: {output_dir}")

logging.info(f"Source video path: {SOURCE_VIDEO_PATH}")
logging.info(f"Target video path: {TARGET_VIDEO_PATH}")

import ultralytics
ultralytics.checks()

import supervision as sv
import numpy as np
print("supervision.__version__:", sv.__version__)


from ultralytics import YOLO
MODEL = "/Users/owwl/Downloads/object_counting/best (6).pt"
model = YOLO(MODEL)
model.fuse()


# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - person
selected_classes = [0]


LINE_START = sv.Point(0,  600)
LINE_END = sv.Point(1920, 600)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)

byte_tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=30, match_thresh=0.8, frame_rate=24)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=0, text_scale=.5)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# acquire first video frame
iterator = iter(generator)
frame = next(iterator)

# model prediction on single frame and conversion to supervision Detections
results = model(frame, verbose=False, device='cpu', conf=.015, iou=.02, imgsz=1280)[0]

# convert to Detections
detections = sv.Detections.from_ultralytics(results)
# only consider class id from selected_classes define above
detections = detections[np.isin(detections.class_id, selected_classes)]
# tracking detections
detections = byte_tracker.update_with_detections(detections)
line_zone.trigger(detections)

labels = [
    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
    for _,_,confidence,class_id,tracker_id,_
    in detections
]

# annotate and display frame
annotated_frame = trace_annotator.annotate(scene=frame.copy(),detections=detections)
annotated_frame=box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

byte_tracker = sv.ByteTrack(track_thresh=0.05, track_buffer=30, match_thresh=0.8, frame_rate=24)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=0, text_scale=.5)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False, device='cpu', conf=.015, iou=.02, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    # update line counter
    line_zone.trigger(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _,_,confidence,class_id,tracker_id,_
        in detections
    ]
    # annotate and display frame
    annotated_frame = trace_annotator.annotate(scene=frame.copy(),detections=detections)
    annotated_frame=box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
    return annotated_frame

def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
    debug: bool,
) -> None:
    try:
        source_video_info = sv.VideoInfo.from_video_path(video_path=source_path)
        # Get video properties
        fps = source_video_info.fps
        width = source_video_info.width
        height = source_video_info.height
        logging.info(f"Video properties - FPS: {fps}, Resolution: {width}x{height}")
        
        # Limit to 24 frames (1 second) for testing
        total_frames = min(24, source_video_info.total_frames)
        logging.info(f"Processing first {total_frames} frames for testing")
        
        # Configure video writer with basic settings
        video_info = sv.VideoInfo(
            fps=fps,
            width=width,
            height=height
        )
        
        with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
            frames_processed = 0
            for index, frame in enumerate(
                tqdm(sv.get_video_frames_generator(source_path=source_path), 
                     total=total_frames, 
                     desc="Processing frames")
            ):
                if index >= total_frames:
                    break
                    
                try:
                    result_frame = callback(frame, index)
                    if result_frame is None:
                        logging.error(f"Frame {index}: callback returned None")
                        continue
                    # Convert BGR to RGB if needed
                    if len(result_frame.shape) == 3 and result_frame.shape[2] == 3:
                        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    sink.write_frame(frame=result_frame)
                    frames_processed += 1
                except Exception as frame_error:
                    logging.error(f"Error processing frame {index}: {str(frame_error)}")
                    raise
                
                if index % 10 == 0:
                    logging.info(f"Processing frame {index}/{total_frames}")
        
        logging.info(f"Frames processed: {frames_processed}")
        
        # Verify output file was created
        if os.path.exists(target_path):
            file_size = os.path.getsize(target_path)
            logging.info(f"Video processing completed. Output saved to: {target_path} (Size: {file_size/1024/1024:.2f} MB)")
        else:
            logging.error(f"Failed to create output video at: {target_path}")
            
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise

process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback,
    debug=False
)