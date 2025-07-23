import os
import json
import numpy as np
import torch
from collections import deque

# Configuration
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_multi1.db'
FRAME_SKIP = 2  # Changed from 8 to 2
VEHICLE_THRESHOLD = 2
MAX_QUEUE_SIZE = 32
PROCESSING_TIMES = {f'cam{i+1}': deque(maxlen=50) for i in range(4)}
IMAGE_SAVE_INTERVAL = 1  # Save images every IMAGE_SAVE_INTERVAL seconds
PROCESS_DURATION = 2  # Process each stream for 2 seconds

# Authentication settings
DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin@123!'


# RTSP URLs for each camera
RTSP_URLS = {
    'cam1': "rtsp://192.168.100.201:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam2': "rtsp://192.168.100.202:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam3': "rtsp://192.168.100.203:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true",
    'cam4': "rtsp://192.168.100.204:554/rtsp/streaming?channel=1&subtype=1&onvif_metadata=true"
}


# Model configuration
NUM_BLOCKS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = os.cpu_count()  # Use os.cpu_count() instead of mp.cpu_count()
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# Color mappings
COLOR_MAPPING = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

REGIONS_FILE = 'regions_config.json'

# Hardcoded default regions - keep only the zebra crossing configurations
DEFAULT_REGIONS = {
    'cam1': {
        'Zebra': {
            'vertices': np.array([(300,461), (2170,489)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam2': {
        'Zebra': {
            'vertices': np.array([(546, 484), (997, 472)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam3': {
        'Zebra': {
            'vertices': np.array([(436, 523), (992, 535)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    },
    'cam4': {
        'Zebra': {
            'vertices': np.array([(271, 73), (287, 694)], dtype=np.int32),
            'weight': 1.0,
            'color': (0, 0, 255)
        }
    }
}

# Create necessary directories
def init_directories():
    os.makedirs('static/temp/', exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    for cam in RTSP_URLS.keys():
        os.makedirs(os.path.join(PROCESSED_FOLDER, cam), exist_ok=True)

# Load regions from config file
def load_regions():
    regions = DEFAULT_REGIONS.copy()

    if os.path.exists(REGIONS_FILE):
        with open(REGIONS_FILE, 'r') as file:
            saved_regions = json.load(file)

        for cam, cam_regions in regions.items():
            if cam in saved_regions:
                for r_name, r_data in saved_regions[cam].items():
                    if r_name == 'Zebra' and 'vertices' in r_data:
                        r_data['vertices'] = np.array(r_data['vertices'], dtype=np.int32)
                        regions[cam].update({r_name: r_data})

    return regions

# Configure CUDA settings if available
def configure_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

# Initialize environment
def initialize_environment():
    init_directories()
    configure_cuda()
    return load_regions()

REGIONS = initialize_environment()
