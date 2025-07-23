import cv2
import numpy as np
import time
import queue
import os
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from collections import defaultdict

from config import (
    DEVICE, REGIONS, MAX_QUEUE_SIZE, NUM_WORKERS,
    VEHICLE_CLASSES, RTSP_URLS, PROCESSED_FOLDER, IMAGE_SAVE_INTERVAL
)
import database

# Set constant frame skip to 2 as requested
FRAME_SKIP = 2

# SAHI configuration
SAHI_CONFIG = {
    'slice_height': 512,           # Optimized from 640
    'slice_width': 512,            # Optimized from 640
    'overlap_height_ratio': 0.5,   # Increased from 0.3
    'overlap_width_ratio': 0.5,    # Increased from 0.3
    'postprocess_type': "NMM",
    'postprocess_match_threshold': 0.4,  # Lowered from 0.5 for better matching
    'confidence_threshold': 0.15   # Lowered from 0.25 to catch distant objects
}

# Load models
def load_models():
    try:
        local_model_path = 'models/yolov8m.pt'

        # Initialize both direct YOLO model and SAHI wrapper
        yolo_model = YOLO(local_model_path)
        yolo_model.to(DEVICE)

        # Setup SAHI detection model with improved parameters for long-range detection
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=local_model_path,
            confidence_threshold=SAHI_CONFIG['confidence_threshold'],  # Lower threshold to detect distant vehicles
            device=DEVICE
        )

        return yolo_model, detection_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

yolo_model, detection_model = load_models()

class VehicleTracker:
    """Class to track vehicles and prevent duplicate detections"""
    def __init__(self, max_disappeared=15, min_distance=30):  # Adjusted for better tracking of distant objects
        self.next_object_id = 0
        self.objects = {}  # Store detected objects {ID: (x, y, vehicle_type, last_seen)}
        self.disappeared = {}  # Track how many frames an object has disappeared for
        self.max_disappeared = max_disappeared  # Max number of frames to keep an object (increased from 10)
        self.min_distance = min_distance  # Minimum distance to consider a new object (decreased from 50)

    def register(self, centroid, vehicle_type):
        """Register a new object"""
        self.objects[self.next_object_id] = (centroid[0], centroid[1], vehicle_type, time.time())
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids, vehicle_types):
        """Update tracked objects with new detections"""
        # No centroids? Just increase all disappeared counters
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        # First time? Register all objects
        if len(self.objects) == 0:
            registered_ids = []
            for i in range(len(centroids)):
                obj_id = self.register(centroids[i], vehicle_types[i])
                registered_ids.append(obj_id)
            return registered_ids

        # Get currently tracked object IDs and centroids
        object_ids = list(self.objects.keys())
        object_centroids = [(self.objects[id][0], self.objects[id][1]) for id in object_ids]

        # Compute distances between each pair of object centroids and new centroids
        new_detected_ids = []

        # For each new centroid
        for i, centroid in enumerate(centroids):
            # Find closest existing object
            min_dist = float('inf')
            min_idx = -1

            for j, obj_centroid in enumerate(object_centroids):
                # Calculate Euclidean distance
                dist = np.sqrt((centroid[0] - obj_centroid[0])**2 +
                               (centroid[1] - obj_centroid[1])**2)

                # Update minimum if this is closer
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j

            # If the closest object is close enough, update it
            if min_dist < self.min_distance and min_idx >= 0:
                # Update the object
                obj_id = object_ids[min_idx]
                self.objects[obj_id] = (centroid[0], centroid[1], vehicle_types[i], time.time())
                self.disappeared[obj_id] = 0
                # This is not a new detection, it's tracking an existing object
            else:
                # Register new object
                obj_id = self.register(centroid, vehicle_types[i])
                new_detected_ids.append(obj_id)

        # Update disappeared counts for any objects that didn't get matched
        for object_id in list(self.disappeared.keys()):
            if object_id not in new_detected_ids and object_centroids:
                self.disappeared[object_id] += 1

                # Deregister objects that have disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        return new_detected_ids

    def get_objects(self):
        """Return currently tracked objects"""
        return self.objects

def enhance_frame_for_detection(frame):
    """Enhance the frame to improve detection of distant objects"""
    try:
        # Apply contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        # Merge the CLAHE enhanced L-channel back with A and B channels
        merged = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced_frame
    except Exception as e:
        print(f"Error enhancing frame: {e}")
        return frame

class CameraProcessor:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self.last_save_time = time.time()
        self.use_sahi = True  # Flag to toggle between SAHI and direct YOLO
        self.tracker = VehicleTracker(max_disappeared=15, min_distance=30)  # Improved tracker settings
        self.last_processed_time = time.time()
        self.frame_count = 0
        self.detected_vehicles_count = defaultdict(int)  # Track vehicle counts by type
        self.use_multi_scale = True  # Enable multi-scale detection for better range coverage

    @staticmethod
    def is_point_in_polygon(x, y, vertices):
        return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

    @staticmethod
    def is_point_near_line(px, py, line_start, line_end, threshold=60):  # Increased from 30/40 to 60
        """Check if a point is near a line segment within a threshold distance"""
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate the distance from point to line segment
        dist = np.abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

        # Check if the point is within the bounding box of the line segment
        if min(x1, x2) - threshold <= px <= max(x1, x2) + threshold and \
           min(y1, y2) - threshold <= py <= max(y1, y2) + threshold:
            return dist <= threshold

        return False

    def process_frame(self, frame):
        try:
            self.frame_count += 1

            # Skip processing if less than 0.5 second has passed since the last processing
            current_time = time.time()
            if current_time - self.last_processed_time < 0.5:
                return None

            self.last_processed_time = current_time

            # Apply image enhancement for better detection of distant objects
            enhanced_frame = enhance_frame_for_detection(frame)

            if self.use_multi_scale and self.use_sahi:
                return self.process_multi_scale(enhanced_frame)
            else:
                return self.process_single_scale(enhanced_frame)

        except Exception as e:
            print(f"Error processing frame for camera {self.cam_id}: {e}")
            return None

    def process_single_scale(self, frame):
        """Process a single frame with SAHI"""
        try:
            if self.use_sahi:
                # Use SAHI for improved detection, especially for small/distant objects
                result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=SAHI_CONFIG['slice_height'],
                    slice_width=SAHI_CONFIG['slice_width'],
                    overlap_height_ratio=SAHI_CONFIG['overlap_height_ratio'],
                    overlap_width_ratio=SAHI_CONFIG['overlap_width_ratio'],
                    perform_standard_pred=True,
                    postprocess_type=SAHI_CONFIG['postprocess_type'],
                    postprocess_match_threshold=SAHI_CONFIG['postprocess_match_threshold'],
                    verbose=0
                )
                return result
            else:
                # Use direct YOLOv8 detection
                result = yolo_model(frame, verbose=False, conf=SAHI_CONFIG['confidence_threshold'])
                return result
        except Exception as e:
            print(f"Error in single-scale processing for camera {self.cam_id}: {e}")
            return None

    def process_multi_scale(self, frame):
        """Process frame at multiple scales for improved detection of distant objects"""
        try:
            # Process at original resolution
            result_original = self.process_single_scale(frame)
            if result_original is None:
                return None

            # Process at larger scale (upscaled) for better detection of distant objects
            height, width = frame.shape[:2]
            upscaled = cv2.resize(frame, (int(width * 1.5), int(height * 1.5)))
            result_upscaled = self.process_single_scale(upscaled)

            if result_upscaled is None:
                return result_original

            # For SAHI results, we need to combine the object prediction lists
            if hasattr(result_original, 'object_prediction_list') and hasattr(result_upscaled, 'object_prediction_list'):
                # Adjust coordinates for the upscaled detections
                for pred in result_upscaled.object_prediction_list:
                    # Scale coordinates back to original frame size
                    bbox = pred.bbox

                    # Create scaled coordinates
                    minx = bbox.minx / 1.5
                    miny = bbox.miny / 1.5
                    maxx = bbox.maxx / 1.5
                    maxy = bbox.maxy / 1.5

                    # Update the bbox coordinates
                    bbox.minx = minx
                    bbox.miny = miny
                    bbox.maxx = maxx
                    bbox.maxy = maxy

                # Add the adjusted upscaled detections to the original results
                result_original.object_prediction_list.extend(result_upscaled.object_prediction_list)

            return result_original

        except Exception as e:
            print(f"Error in multi-scale detection for camera {self.cam_id}: {e}")
            return self.process_single_scale(frame)  # Fallback to single-scale

    def process_zebra_crossing_vehicles(self, frame, result):
        try:
            zebra_data = REGIONS.get(f'cam{self.cam_id}', {}).get('Zebra')
            if not zebra_data:
                return frame, 0

            vertices = zebra_data['vertices']
            processed_frame = frame.copy()
            vehicle_count = 0
            vehicle_types = {}

            # For tracking new detections
            new_centroids = []
            new_vehicle_types = []

            # Draw the zebra crossing line
            if len(vertices) >= 2:
                # Create a thicker, more visible line
                cv2.line(processed_frame, tuple(vertices[0]), tuple(vertices[1]), zebra_data['color'], 4)

                # Add a semi-transparent overlay to highlight the crossing area
                mask = np.zeros_like(processed_frame)
                pts = np.array([[vertices[0][0]-20, vertices[0][1]-20],
                               [vertices[0][0]-20, vertices[0][1]+20],
                               [vertices[1][0]+20, vertices[1][1]+20],
                               [vertices[1][0]+20, vertices[1][1]-20]], np.int32)
                cv2.fillPoly(mask, [pts], (0, 0, 255, 50))
                alpha = 0.3  # Increased visibility
                processed_frame = cv2.addWeighted(processed_frame, 1, mask, alpha, 0)

            if result is None:
                # Draw existing tracked objects
                for obj_id, obj in self.tracker.objects.items():
                    x, y, vehicle_type, _ = obj
                    color = self._get_vehicle_color(vehicle_type)
                    cv2.circle(processed_frame, (int(x), int(y)), 5, color, -1)
                return processed_frame, 0

            # Handle both SAHI and direct YOLO results
            if self.use_sahi:
                # Process SAHI result
                for pred in result.object_prediction_list:
                    if pred.category.name.lower() not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())
                    score = float(pred.score.value)
                    category = pred.category.name.lower()

                    # Skip low confidence detections - lowered threshold for distant objects
                    if score < SAHI_CONFIG['confidence_threshold']:
                        continue

                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Check if vehicle is near the zebra line - increased threshold
                    if len(vertices) >= 2 and self.is_point_near_line(x_center, y_center,
                                                                     tuple(vertices[0]),
                                                                     tuple(vertices[1]),
                                                                     threshold=60):  # Increased from 40 to 60
                        # Add to tracking
                        new_centroids.append((x_center, y_center))
                        new_vehicle_types.append(category)

                        # Track vehicle types for UI
                        if category in vehicle_types:
                            vehicle_types[category] += 1
                        else:
                            vehicle_types[category] = 1

                        # Draw bounding box with different colors based on vehicle type
                        color = self._get_vehicle_color(category)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                        # Create nice looking label with background
                        label = f'{category}: {score:.2f}'
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(processed_frame, (x1, y1 - 25), (x1 + label_w + 10, y1), color, -1)
                        cv2.putText(processed_frame, label, (x1 + 5, y1 - 7),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Process direct YOLO result
                for detection in result:
                    boxes = detection.boxes

                    # Process each box individually
                    for i in range(len(boxes)):
                        try:
                            cls_id = int(boxes.cls[i].item())
                            conf = float(boxes.conf[i].item())
                            box_xyxy = boxes.xyxy[i].cpu().numpy()

                            category = detection.names[cls_id].lower()
                            if category not in VEHICLE_CLASSES:
                                continue

                            # Skip low confidence detections - lowered threshold
                            if conf < SAHI_CONFIG['confidence_threshold']:
                                continue

                            x1, y1, x2, y2 = map(int, box_xyxy)
                            x_center = int((x1 + x2) / 2.0)
                            y_center = int((y1 + y2) / 2.0)

                            # Check if vehicle is near the zebra line - increased threshold
                            if len(vertices) >= 2 and self.is_point_near_line(x_center, y_center,
                                                                            tuple(vertices[0]),
                                                                            tuple(vertices[1]),
                                                                            threshold=60):  # Increased from 40 to 60
                                # Add to tracking
                                new_centroids.append((x_center, y_center))
                                new_vehicle_types.append(category)

                                # Track vehicle types for UI
                                if category in vehicle_types:
                                    vehicle_types[category] += 1
                                else:
                                    vehicle_types[category] = 1

                                # Draw bounding box with different colors based on vehicle type
                                color = self._get_vehicle_color(category)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

                                # Create nice looking label with background
                                label = f'{category}: {conf:.2f}'
                                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                cv2.rectangle(processed_frame, (x1, y1 - 25), (x1 + label_w + 10, y1), color, -1)
                                cv2.putText(processed_frame, label, (x1 + 5, y1 - 7),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        except Exception as e:
                            print(f"Error processing individual detection: {e}")
                            continue

            # Update tracker with new detections
            new_ids = self.tracker.update(new_centroids, new_vehicle_types)

            # Save to database only newly detected vehicles
            for obj_id in new_ids:
                if obj_id in self.tracker.objects:
                    x, y, vehicle_type, _ = self.tracker.objects[obj_id]

                    # Count for this frame's detection
                    vehicle_count += 1

                    # Update overall vehicle count statistics
                    self.detected_vehicles_count[vehicle_type] += 1

                    # Save to database
                    database.save_zebra_crossing_data(
                        self.cam_id,
                        vehicle_type=vehicle_type,
                        x_position=int(x),
                        y_position=int(y),
                        road_side='main',
                        confidence=0.9  # Default high confidence for tracked objects
                    )

            # Add a nice looking vehicle count display with background
            count_text = f'Vehicles: {vehicle_count}'
            total_count_text = f'Total: {sum(self.detected_vehicles_count.values())}'

            # Create a semi-transparent overlay at the top
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (0, 0), (processed_frame.shape[1], 80), (0, 0, 0), -1)
            alpha = 0.7
            processed_frame = cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0)

            # Add count on the overlay
            cv2.putText(processed_frame, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(processed_frame, total_count_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add vehicle type counts
            x_pos = 200
            for i, (v_type, count) in enumerate(self.detected_vehicles_count.items()):
                color = self._get_vehicle_color(v_type)
                text = f"{v_type}: {count}"
                cv2.putText(processed_frame, text, (x_pos, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                x_pos += 120

                if i == 2 and x_pos > processed_frame.shape[1] - 120:
                    x_pos = 200

            # Add timestamp overlay at bottom right
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            (text_w, text_h), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(processed_frame,
                         (processed_frame.shape[1] - text_w - 20, processed_frame.shape[0] - text_h - 20),
                         (processed_frame.shape[1], processed_frame.shape[0]),
                         (0, 0, 0), -1)
            cv2.putText(processed_frame,
                      timestamp,
                      (processed_frame.shape[1] - text_w - 10, processed_frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.6,
                      (255, 255, 255),
                      2)

            return processed_frame, vehicle_count

        except Exception as e:
            print(f"Error processing zebra crossing vehicles for camera {self.cam_id}: {e}")
            return frame, 0

    def _get_vehicle_color(self, vehicle_type):
        """Return color based on vehicle type for visualization"""
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (0, 165, 255),  # Orange
            'bus': (0, 0, 255),      # Red
            'motorcycle': (255, 0, 0), # Blue
            'bicycle': (255, 255, 0)  # Cyan
        }
        return colors.get(vehicle_type, (255, 255, 255))  # Default to white

    def analyze_and_save(self, frame, result):
        # Process vehicles crossing zebra lines
        processed_frame, vehicle_count = self.process_zebra_crossing_vehicles(frame, result)

        # Save the processed frame regularly
        current_time = time.time()
        if current_time - self.last_save_time >= IMAGE_SAVE_INTERVAL:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(PROCESSED_FOLDER, f'cam{self.cam_id}', filename)
            cv2.imwrite(save_path, processed_frame)
            self.last_save_time = current_time

        return processed_frame, vehicle_count

def process_camera_stream(cam_id, duration=2):  # Process each stream for 2 seconds
    processor = CameraProcessor(cam_id)
    cap = cv2.VideoCapture(RTSP_URLS[f'cam{cam_id}'])

    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Using frame skip of 2
        if frame_count % FRAME_SKIP != 0:
            continue

        result = processor.process_frame(frame)
        processed_frame, vehicle_count = processor.analyze_and_save(frame, result)

    cap.release()
    processor.thread_pool.shutdown()

def get_single_frame(cam_id):
    """Get a single frame from a camera for preview"""
    try:
        cap = cv2.VideoCapture(RTSP_URLS.get(cam_id))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        return frame
    except Exception as e:
        print(f"Error capturing single frame from camera {cam_id}: {e}")
        return None

def clear_camera_images(cam_id):
    """Clear all processed images for a specific camera when boundaries change"""
    try:
        cam_folder = os.path.join(PROCESSED_FOLDER, f'cam{cam_id}')
        if os.path.exists(cam_folder):
            # Delete all files in the folder
            for file in os.listdir(cam_folder):
                file_path = os.path.join(cam_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all images for camera {cam_id}")
        return True
    except Exception as e:
        print(f"Error clearing images for camera {cam_id}: {e}")
        return False

def cyclic_processing():
    """Process each camera in a cycle"""
    while True:
        for cam_id in range(1, 5):
            process_camera_stream(cam_id)
            time.sleep(0.1)
