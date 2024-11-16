#!/usr/bin/env python3
"""
Advanced Room Mapper with 3D Scanning and Path Planning
Version: 1.0.0
"""
try:
    import cv2
    import numpy as np
    import open3d as o3d
    import tkinter as tk
    from tkinter import ttk, messagebox
    import customtkinter as ctk
    from PIL import Image, ImageTk
    import torch
    from ultralytics import YOLO
    import speech_recognition as sr
    import pygame
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    import json
    import logging
    import threading
    import queue
    from pathlib import Path
    from datetime import datetime
    import time
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import networkx as nx
    from scipy.spatial import KDTree
    from scipy.interpolate import splprep, splev
    import yaml
    import warnings
    import sys
    import os
    import serial
    import pyrealsense2 as rs
    from enum import Enum, auto
    import torch.nn as nn
    import torch.optim as optim
    from concurrent.futures import ThreadPoolExecutor
    import asyncio
    import mpl_toolkits.mplot3d.proj3d as proj3d

except ImportError as e:
    print(f"Missing required dependency: {e}")
    raise

# Suppress warnings but show critical ones
warnings.filterwarnings('ignore')
logging.captureWarnings(True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_mapper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ScanningState(Enum):
    """Enum for scanning states"""
    IDLE = auto()
    CALIBRATING = auto()
    SCANNING = auto()
    PROCESSING = auto()
    PATH_PLANNING = auto()
    ERROR = auto()

class ConfigManager:
    """Manages configuration settings for the application"""
    
    DEFAULT_CONFIG = {
        'cameras': {
            'left_camera_id': 0,
            'right_camera_id': 1,
            'thermal_camera_port': '/dev/ttyACM0',
            'thermal_camera_baudrate': 115200,
            'depth_camera_resolution': (640, 480),
            'depth_camera_fps': 30
        },
        'scanning': {
            'voxel_size': 0.05,
            'max_depth': 10.0,
            'min_depth': 0.5,
            'point_cloud_density': 'medium',
            'scan_quality': 'high'
        },
        'path_planning': {
            'safety_margin': 0.3,
            'smoothing_factor': 0.85,
            'max_iterations': 1000,
            'optimization_method': 'rrt_star'
        },
        'ml_models': {
            'object_detection': 'yolov8n.pt',
            'semantic_segmentation': 'segment-anything',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'visualization': {
            'point_size': 2,
            'path_color': [1, 0, 0],
            'background_color': [0.1, 0.1, 0.1],
            'show_coordinate_frame': True
        },
        'export': {
            'default_format': 'ply',
            'compression': True,
            'include_metadata': True
        }
    }
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info("Configuration loaded successfully")
                    return self.validate_config(config)
            else:
                self.save_config(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self.DEFAULT_CONFIG
            
    def save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def validate_config(self, config):
        """Validate and update configuration with any missing default values"""
        if not isinstance(config, dict):
            logger.error("Invalid configuration format")
            return self.DEFAULT_CONFIG
            
        validated = self.DEFAULT_CONFIG.copy()
        def update_dict(default, new):
            for key, value in default.items():
                if key in new:
                    if isinstance(value, dict) and isinstance(new[key], dict):
                        update_dict(value, new[key])
                    else:
                        if isinstance(value, type(new[key])):  # Type check
                            default[key] = new[key]
                        else:
                            logger.warning(f"Invalid type for config key {key}")
                            
        update_dict(validated, config)
        return validated
        
    def get(self, *keys):
        """Get configuration value using dot notation"""
        value = self.config
        for key in keys:
            value = value[key]
        return value
        
    def set(self, value, *keys):
        """Set configuration value using dot notation"""
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        self.save_config(self.config)

class EventSystem:
    """Event system for communication between components"""
    
    def __init__(self):
        self.subscribers = {}
        self._lock = threading.Lock()  # Add thread lock
        self.event_queue = queue.Queue()
        self._running = True
        self.event_thread = threading.Thread(target=self._process_events, daemon=True)
        self.event_thread.start()
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type, callback):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
            
    def publish(self, event_type, data=None):
        """Publish an event"""
        self.event_queue.put((event_type, data))
        
    def _process_events(self):
        """Process events in the queue"""
        while self._running:
            try:
                event_type, data = self.event_queue.get(timeout=0.1)
                if event_type in self.subscribers:
                    for callback in self.subscribers[event_type]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
            except queue.Empty:
                continue
                
    def shutdown(self):
        """Shutdown the event system"""
        self._running = False
        self.event_thread.join()

class CameraSystem:
    """Manages multiple camera inputs including stereo and thermal cameras"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.state = ScanningState.IDLE
        
        # Initialize cameras
        self.left_camera = None
        self.right_camera = None
        self.thermal_camera = None
        self.depth_camera = None
        
        # Threading
        self._running = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.thermal_queue = queue.Queue(maxsize=30)
        
        # Calibration data
        self.calibration_data = None
        
    def initialize_cameras(self):
        """Initialize all camera devices"""
        try:
            # Initialize stereo cameras
            self.left_camera = cv2.VideoCapture(self.config.get('cameras', 'left_camera_id'))
            self.right_camera = cv2.VideoCapture(self.config.get('cameras', 'right_camera_id'))
            
            # Set camera properties
            for camera in [self.left_camera, self.right_camera]:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                camera.set(cv2.CAP_PROP_FPS, 30)
                
            # Initialize thermal camera
            try:
                self.thermal_camera = serial.Serial(
                    port=self.config.get('cameras', 'thermal_camera_port'),
                    baudrate=self.config.get('cameras', 'thermal_camera_baudrate')
                )
            except serial.SerialException as e:
                logger.warning(f"Thermal camera not available: {e}")
                
            # Initialize RealSense depth camera if available
            try:
                self.depth_camera = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.depth_camera.start(config)
            except Exception as e:
                logger.warning(f"Depth camera not available: {e}")
                
            logger.info("Camera system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cameras: {e}")
            return False
            
    def start_capture(self):
        """Start capturing frames from all cameras"""
        self._running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        if self.thermal_camera:
            self.thermal_thread = threading.Thread(target=self._thermal_capture_loop, daemon=True)
            self.thermal_thread.start()
            
    def stop_capture(self):
        """Stop capturing frames"""
        self._running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'thermal_thread'):
            self.thermal_thread.join()
            
    def _capture_loop(self):
        """Main capture loop for stereo cameras"""
        while self._running:
            try:
                ret1, left_frame = self.left_camera.read()
                ret2, right_frame = self.right_camera.read()
                
                if ret1 and ret2:
                    # Apply calibration if available
                    if self.calibration_data is not None:
                        left_frame, right_frame = self.apply_calibration(left_frame, right_frame)
                        
                    self.frame_queue.put((left_frame, right_frame))
                    self.events.publish('new_frames', (left_frame, right_frame))
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
                
    def _thermal_capture_loop(self):
        """Capture loop for thermal camera"""
        while self._running and self.thermal_camera:
            try:
                if self.thermal_camera.in_waiting:
                    thermal_data = self.read_thermal_data()
                    self.thermal_queue.put(thermal_data)
                    self.events.publish('new_thermal_data', thermal_data)
            except Exception as e:
                logger.error(f"Error in thermal capture: {e}")
                time.sleep(0.1)
                
    def read_thermal_data(self):
        """Read and parse thermal camera data"""
        try:
            # Implementation specific to FLIR Lepton 3.5
            # This is a simplified version - actual implementation would depend on the camera protocol
            raw_data = self.thermal_camera.read(164)  # Lepton frame size
            thermal_frame = np.frombuffer(raw_data, dtype=np.uint16).reshape((60, 80))
            # Convert raw values to temperatures
            thermal_frame = (thermal_frame - 27315) / 100.0  # Convert to Celsius
            return thermal_frame
        except Exception as e:
            logger.error(f"Error reading thermal data: {e}")
            return None
            
    def calibrate_cameras(self):
        """Perform stereo camera calibration"""
        self.state = ScanningState.CALIBRATING
        
        try:
            # Prepare calibration parameters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            checkerboard = (9, 6)
            objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
            
            objpoints = []
            imgpoints_left = []
            imgpoints_right = []
            
            # Collect calibration frames
            for _ in range(20):  # Capture 20 different checkerboard positions
                ret1, left_frame = self.left_camera.read()
                ret2, right_frame = self.right_camera.read()
                
                if ret1 and ret2:
                    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                    
                    ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard, None)
                    ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard, None)
                    
                    if ret_left and ret_right:
                        objpoints.append(objp)
                        
                        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                        
                        imgpoints_left.append(corners_left)
                        imgpoints_right.append(corners_right)
                        
                        # Display the corners
                        cv2.drawChessboardCorners(left_frame, checkerboard, corners_left, ret_left)
                        cv2.drawChessboardCorners(right_frame, checkerboard, corners_right, ret_right)
                        
                        self.events.publish('calibration_frame', (left_frame, right_frame))
                        time.sleep(0.5)
                        
            # Perform stereo calibration
            ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                None, None, None, None,
                gray_left.shape[::-1],
                criteria=criteria,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            # Save calibration data
            self.calibration_data = {
                'mtx_left': mtx_left,
                'dist_left': dist_left,
                'mtx_right': mtx_right,
                'dist_right': dist_right,
                'R': R,
                'T': T,
                'E': E,
                'F': F
            }
            
            logger.info("Camera calibration completed successfully")
            self.state = ScanningState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.state = ScanningState.ERROR
            return False
            
    def apply_calibration(self, left_frame, right_frame):
        """Apply calibration to stereo frames"""
        if self.calibration_data is None:
            return left_frame, right_frame
            
        try:
            left_frame = cv2.undistort(
                left_frame,
                self.calibration_data['mtx_left'],
                self.calibration_data['dist_left']
            )
            right_frame = cv2.undistort(
                right_frame,
                self.calibration_data['mtx_right'],
                self.calibration_data['dist_right']
            )
            return left_frame, right_frame
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return left_frame, right_frame
            
    def cleanup(self):
        """Clean up camera resources"""
        try:
            self.stop_capture()
            
            if hasattr(self, 'left_camera') and self.left_camera:
                self.left_camera.release()
            if hasattr(self, 'right_camera') and self.right_camera:
                self.right_camera.release()
            if hasattr(self, 'thermal_camera') and self.thermal_camera:
                self.thermal_camera.close()
            if hasattr(self, 'depth_camera') and self.depth_camera:
                self.depth_camera.stop()
                
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")

class PointCloudProcessor:
    """Handles point cloud generation, processing, and optimization"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.voxel_size = self.config.get('scanning', 'voxel_size')
        self.current_point_cloud = o3d.geometry.PointCloud()
        self.processing_thread = None
        self._processing = False
        
    def process_stereo_frames(self, left_frame, right_frame, calibration_data):
        """Generate point cloud from stereo frames"""
        try:
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*16,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
            
            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
            
            # Generate point cloud
            h, w = left_gray.shape
            Q = calibration_data['Q']  # Perspective transformation matrix
            points = cv2.reprojectImageTo3D(disparity, Q)
            colors = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            
            # Filter valid points
            mask = disparity > disparity.min()
            points = points[mask]
            colors = colors[mask]
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            return pcd
            
        except Exception as e:
            logger.error(f"Error processing stereo frames: {e}")
            return None
            
    def merge_point_clouds(self, new_pcd):
        """Merge new point cloud with existing one"""
        if self.current_point_cloud.is_empty():
            self.current_point_cloud = new_pcd
            return
            
        try:
            # ICP registration
            result = o3d.pipelines.registration.registration_icp(
                new_pcd, self.current_point_cloud,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Transform and combine point clouds
            new_pcd.transform(result.transformation)
            self.current_point_cloud += new_pcd
            
            # Optimize combined point cloud
            self.optimize_point_cloud()
            
        except Exception as e:
            logger.error(f"Error merging point clouds: {e}")
            
    def optimize_point_cloud(self):
        """Optimize point cloud by removing noise and downsampling"""
        try:
            # Remove statistical outliers
            self.current_point_cloud, _ = self.current_point_cloud.remove_statistical_outliers(
                nb_neighbors=20,
                std_ratio=2.0
            )
            
            # Voxel downsampling
            self.current_point_cloud = self.current_point_cloud.voxel_down_sample(
                voxel_size=self.voxel_size
            )
            
            # Estimate normals
            self.current_point_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            
        except Exception as e:
            logger.error(f"Error optimizing point cloud: {e}")

class MLProcessor:
    """Handles machine learning tasks including object detection and semantic segmentation"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.device = self.config.get('ml_models', 'device')
        
        # Initialize models
        self.object_detector = None
        self.semantic_segmenter = None
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize YOLO for object detection
            model_path = self.config.get('ml_models', 'object_detection')
            self.object_detector = YOLO(model_path)
            
            # Initialize semantic segmentation model
            self.semantic_segmenter = self.load_semantic_segmentation_model()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            
    def load_semantic_segmentation_model(self):
        """Load semantic segmentation model"""
        try:
            # Implementation for loading semantic segmentation model
            # This is a placeholder - implement based on specific requirements
            return None
        except Exception as e:
            logger.error(f"Error loading semantic segmentation model: {e}")
            return None
            
    def detect_objects(self, frame):
        """Perform object detection on frame"""
        try:
            results = self.object_detector(frame)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': box.conf.cpu().numpy()[0],
                        'class': result.names[int(box.cls.cpu().numpy()[0])]
                    }
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
            
    def segment_scene(self, frame):
        """Perform semantic segmentation on frame"""
        try:
            # Implement semantic segmentation
            # This is a placeholder - implement based on specific requirements
            return None
        except Exception as e:
            logger.error(f"Error in semantic segmentation: {e}")
            return None
            
    def process_thermal_data(self, thermal_frame):
        """Process thermal data for anomaly detection"""
        try:
            # Convert thermal data to temperature values
            temp_min = np.min(thermal_frame)
            temp_max = np.max(thermal_frame)
            
            # Detect temperature anomalies
            threshold = np.mean(thermal_frame) + 2 * np.std(thermal_frame)
            anomalies = thermal_frame > threshold
            
            return {
                'temperature_range': (temp_min, temp_max),
                'anomalies': anomalies,
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"Error processing thermal data: {e}")
            return None
            
    def analyze_scene(self, frame, point_cloud):
        """Comprehensive scene analysis combining multiple ML tasks"""
        try:
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Perform semantic segmentation
            segmentation = self.segment_scene(frame)
            
            # Analyze spatial relationships
            spatial_analysis = self.analyze_spatial_relationships(point_cloud, detections)
            
            return {
                'detections': detections,
                'segmentation': segmentation,
                'spatial_analysis': spatial_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return None
            
    def analyze_spatial_relationships(self, point_cloud, detections):
        """Analyze spatial relationships between detected objects"""
        try:
            relationships = []
            
            if point_cloud and detections:
                # Convert point cloud to numpy array for processing
                points = np.asarray(point_cloud.points)
                
                # Build KD-tree for efficient nearest neighbor search
                tree = KDTree(points)
                
                # Analyze relationships between detected objects
                for i, det1 in enumerate(detections):
                    for j, det2 in enumerate(detections[i+1:], i+1):
                        # Calculate spatial relationship
                        relationship = self.calculate_spatial_relationship(
                            det1, det2, points, tree
                        )
                        relationships.append(relationship)
                        
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing spatial relationships: {e}")
            return []
            
    def calculate_spatial_relationship(self, det1, det2, points, tree):
        """Calculate spatial relationship between two detected objects"""
        try:
            # Calculate center points of detections
            center1 = (det1['bbox'][:2] + det1['bbox'][2:]) / 2
            center2 = (det2['bbox'][:2] + det2['bbox'][2:]) / 2
            
            # Find nearest 3D points to detection centers
            _, idx1 = tree.query(center1)
            _, idx2 = tree.query(center2)
            
            # Calculate 3D distance
            distance = np.linalg.norm(points[idx1] - points[idx2])
            
            return {
                'object1': det1['class'],
                'object2': det2['class'],
                'distance': distance,
                'relative_position': points[idx2] - points[idx1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating spatial relationship: {e}")
            return None

class PathPlanner:
    """Handles path planning and navigation in 3D space"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.safety_margin = self.config.get('path_planning', 'safety_margin')
        self.occupancy_grid = None
        self.graph = nx.Graph()
        self.path = None
        self.optimization_method = self.config.get('path_planning', 'optimization_method')
        
    def initialize_occupancy_grid(self, point_cloud):
        """Initialize 3D occupancy grid from point cloud"""
        try:
            points = np.asarray(point_cloud.points)
            voxel_size = self.config.get('scanning', 'voxel_size')
            
            # Create occupancy grid
            self.occupancy_grid = {}
            
            # Convert points to voxel coordinates
            voxel_coords = np.floor(points / voxel_size).astype(int)
            
            # Mark occupied voxels
            for coord in voxel_coords:
                self.occupancy_grid[tuple(coord)] = True
                
            # Add safety margin
            self.add_safety_margin()
            
            logger.info("Occupancy grid initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing occupancy grid: {e}")
            
    def add_safety_margin(self):
        """Add safety margin around obstacles"""
        try:
            margin_cells = int(np.ceil(self.safety_margin / self.config.get('scanning', 'voxel_size')))
            occupied_voxels = list(self.occupancy_grid.keys())
            
            for voxel in occupied_voxels:
                for dx in range(-margin_cells, margin_cells + 1):
                    for dy in range(-margin_cells, margin_cells + 1):
                        for dz in range(-margin_cells, margin_cells + 1):
                            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                            self.occupancy_grid[neighbor] = True
                            
        except Exception as e:
            logger.error(f"Error adding safety margin: {e}")
            
    def build_navigation_graph(self):
        """Build navigation graph for path planning"""
        try:
            self.graph.clear()
            voxel_size = self.config.get('scanning', 'voxel_size')
            
            # Find free voxels
            free_voxels = set()
            for voxel in self.occupancy_grid:
                if not self.occupancy_grid[voxel]:
                    free_voxels.add(voxel)
                    
            # Create graph nodes
            for voxel in free_voxels:
                self.graph.add_node(voxel)
                
            # Create graph edges
            for voxel in free_voxels:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                                
                            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                            if neighbor in free_voxels:
                                distance = np.sqrt(dx**2 + dy**2 + dz**2) * voxel_size
                                self.graph.add_edge(voxel, neighbor, weight=distance)
                                
            logger.info("Navigation graph built successfully")
            
        except Exception as e:
            logger.error(f"Error building navigation graph: {e}")
            
    def find_path(self, start_point, end_point):
        """Find optimal path between two points"""
        try:
            if start_point is None or end_point is None:
                raise ValueError("Start and end points must be specified")
                
            voxel_size = self.config.get('scanning', 'voxel_size')
            
            # Convert points to voxel coordinates
            start_voxel = tuple(np.floor(start_point / voxel_size).astype(int))
            end_voxel = tuple(np.floor(end_point / voxel_size).astype(int))
            
            if self.optimization_method == 'rrt_star':
                path = self.rrt_star_path(start_voxel, end_voxel)
            else:
                path = self.astar_path(start_voxel, end_voxel)
                
            if path:
                # Convert path back to real coordinates
                real_path = np.array(path) * voxel_size
                # Smooth the path
                smoothed_path = self.smooth_path(real_path)
                self.path = smoothed_path
                return smoothed_path
            else:
                logger.warning("No valid path found")
                return None
                
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None
            
    def astar_path(self, start_voxel, end_voxel):
        """Find path using A* algorithm"""
        try:
            path = nx.astar_path(
                self.graph,
                start_voxel,
                end_voxel,
                heuristic=lambda a, b: np.sqrt(sum((x-y)**2 for x, y in zip(a, b))),
                weight='weight'
            )
            return path
        except nx.NetworkXNoPath:
            return None
            
    def rrt_star_path(self, start_voxel, end_voxel):
        """Find path using RRT* algorithm"""
        try:
            max_iterations = self.config.get('path_planning', 'max_iterations')
            
            class Node:
                def __init__(self, coord):
                    self.coord = coord
                    self.parent = None
                    self.cost = 0
                    
            def distance(node1, node2):
                return np.sqrt(sum((a-b)**2 for a, b in zip(node1.coord, node2.coord)))
                
            def is_collision_free(coord1, coord2):
                # Check if path between coordinates is collision-free
                direction = np.array(coord2) - np.array(coord1)
                distance = np.linalg.norm(direction)
                steps = int(distance * 2)
                for i in range(steps):
                    point = coord1 + direction * i / steps
                    voxel = tuple(np.floor(point).astype(int))
                    if voxel in self.occupancy_grid and self.occupancy_grid[voxel]:
                        return False
                return True
                
            # Initialize RRT*
            nodes = [Node(start_voxel)]
            
            for _ in range(max_iterations):
                # Sample random point
                random_coord = tuple(np.random.randint(-10, 10, 3) + np.array(start_voxel))
                
                # Find nearest node
                nearest_node = min(nodes, key=lambda n: distance(n, Node(random_coord)))
                
                # Extend towards random point
                direction = np.array(random_coord) - np.array(nearest_node.coord)
                direction = direction / np.linalg.norm(direction)
                new_coord = tuple(np.array(nearest_node.coord) + direction)
                
                if is_collision_free(nearest_node.coord, new_coord):
                    new_node = Node(new_coord)
                    new_node.parent = nearest_node
                    new_node.cost = nearest_node.cost + distance(nearest_node, new_node)
                    
                    # Rewire nearby nodes
                    nearby_nodes = [n for n in nodes if distance(n, new_node) < 2.0]
                    for near_node in nearby_nodes:
                        if (is_collision_free(new_node.coord, near_node.coord) and
                            new_node.cost + distance(new_node, near_node) < near_node.cost):
                            near_node.parent = new_node
                            near_node.cost = new_node.cost + distance(new_node, near_node)
                            
                    nodes.append(new_node)
                    
                    # Check if we can connect to goal
                    if distance(new_node, Node(end_voxel)) < 1.0 and is_collision_free(new_node.coord, end_voxel):
                        # Reconstruct path
                        path = [end_voxel]
                        current_node = new_node
                        while current_node is not None:
                            path.append(current_node.coord)
                            current_node = current_node.parent
                        return path[::-1]
                        
            return None
            
        except Exception as e:
            logger.error(f"Error in RRT* path planning: {e}")
            return None
            
    def smooth_path(self, path):
        """Smooth the path using spline interpolation"""
        try:
            if len(path) < 3:
                return path
                
            # Fit spline to path points
            t = np.arange(len(path))
            x = path[:, 0]
            y = path[:, 1]
            z = path[:, 2]
            
            # Create spline functions
            tck, u = splprep([x, y, z], s=self.config.get('path_planning', 'smoothing_factor'))
            
            # Generate more points along the spline
            u_new = np.linspace(0, 1, num=len(path) * 5)
            smooth_path = np.column_stack(splev(u_new, tck))
            
            # Verify smoothed path is collision-free
            if self.verify_path(smooth_path):
                return smooth_path
            else:
                logger.warning("Smoothed path has collisions, returning original path")
                return path
                
        except Exception as e:
            logger.error(f"Error smoothing path: {e}")
            return path
            
    def verify_path(self, path):
        """Verify path is collision-free"""
        try:
            voxel_size = self.config.get('scanning', 'voxel_size')
            
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                direction = end - start
                distance = np.linalg.norm(direction)
                steps = int(distance / voxel_size * 2)
                
                for step in range(steps):
                    point = start + direction * step / steps
                    voxel = tuple(np.floor(point / voxel_size).astype(int))
                    if voxel in self.occupancy_grid and self.occupancy_grid[voxel]:
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error verifying path: {e}")
            return False

class Visualizer:
    """Handles 3D visualization and AR overlay"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.vis = None
        self.ar_renderer = None
        self.current_view = None
        self.coordinate_frame = None
        
        # Initialize visualization parameters
        self.point_size = self.config.get('visualization', 'point_size')
        self.path_color = self.config.get('visualization', 'path_color')
        self.background_color = self.config.get('visualization', 'background_color')
        
        # Initialize visualization window
        self.initialize_visualizer()
        
    def initialize_visualizer(self):
        """Initialize Open3D visualizer"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            
            # Set render options
            opt = self.vis.get_render_option()
            opt.background_color = np.asarray(self.background_color)
            opt.point_size = self.point_size
            
            # Add coordinate frame if configured
            if self.config.get('visualization', 'show_coordinate_frame'):
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=[0, 0, 0])
                self.vis.add_geometry(self.coordinate_frame)
                
            logger.info("Visualizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing visualizer: {e}")
            
    class ARRenderer:
        """Handles AR rendering and overlay"""
        
        def __init__(self):
            self.camera_matrix = None
            self.dist_coeffs = None
            self.ar_objects = {}
            
            # Initialize pygame for AR rendering
            pygame.init()
            
        def initialize_ar(self, camera_matrix, dist_coeffs):
            """Initialize AR parameters"""
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            
        def create_ar_overlay(self, frame, path=None, objects=None):
            """Create AR overlay on camera frame"""
            try:
                if self.camera_matrix is None:
                    return frame
                    
                # Create overlay surface
                overlay = frame.copy()
                
                if path is not None:
                    self.draw_path_ar(overlay, path)
                    
                if objects is not None:
                    self.draw_objects_ar(overlay, objects)
                    
                return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
            except Exception as e:
                logger.error(f"Error creating AR overlay: {e}")
                return frame
                
        def draw_path_ar(self, overlay, path):
            """Draw path in AR"""
            try:
                points_2d, _ = cv2.projectPoints(
                    path,
                    np.zeros(3),
                    np.zeros(3),
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                points_2d = points_2d.astype(np.int32)
                
                # Draw path
                for i in range(len(points_2d) - 1):
                    cv2.line(
                        overlay,
                        tuple(points_2d[i][0]),
                        tuple(points_2d[i + 1][0]),
                        (0, 0, 255),
                        2
                    )
                    
            except Exception as e:
                logger.error(f"Error drawing AR path: {e}")
                
        def draw_objects_ar(self, overlay, objects):
            """Draw detected objects in AR"""
            try:
                for obj in objects:
                    if 'bbox' in obj:
                        bbox = obj['bbox'].astype(np.int32)
                        cv2.rectangle(
                            overlay,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0),
                            2
                        )
                        
                        # Add label
                        cv2.putText(
                            overlay,
                            f"{obj['class']} ({obj['confidence']:.2f})",
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
            except Exception as e:
                logger.error(f"Error drawing AR objects: {e}")
                
    def update_visualization(self, point_cloud=None, path=None, objects=None):
        """Update 3D visualization"""
        try:
            if point_cloud is not None:
                if self.current_view is not None:
                    self.vis.remove_geometry(self.current_view)
                self.current_view = point_cloud
                self.vis.add_geometry(self.current_view)
                
            if path is not None:
                # Create line set for path
                lines = [[i, i + 1] for i in range(len(path) - 1)]
                colors = [self.path_color for _ in range(len(lines))]
                
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(path),
                    lines=o3d.utility.Vector2iVector(lines)
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                
                self.vis.add_geometry(line_set)
                
            # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            
    def create_interactive_visualization(self, point_cloud, path=None):
        """Create interactive visualization window"""
        try:
            # Create new window for interactive visualization
            vis_interactive = o3d.visualization.VisualizerWithKeyCallback()
            vis_interactive.create_window()
            
            # Add geometries
            vis_interactive.add_geometry(point_cloud)
            
            if path is not None:
                # Create line set for path
                lines = [[i, i + 1] for i in range(len(path) - 1)]
                colors = [self.path_color for _ in range(len(lines))]
                
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(path),
                    lines=o3d.utility.Vector2iVector(lines)
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis_interactive.add_geometry(line_set)
                
            # Add coordinate frame
            if self.config.get('visualization', 'show_coordinate_frame'):
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.5, origin=[0, 0, 0])
                vis_interactive.add_geometry(coordinate_frame)
                
            # Set view control
            ctr = vis_interactive.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0, 0, -1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, -1, 0])
            
            # Run visualization
            vis_interactive.run()
            vis_interactive.destroy_window()
            
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {e}")
            
    def capture_screenshot(self, filename):
        """Capture screenshot of current visualization"""
        try:
            self.vis.capture_screen_image(filename)
            logger.info(f"Screenshot saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            
    def cleanup(self):
        """Clean up visualization resources"""
        try:
            if self.vis is not None:
                self.vis.destroy_window()
            pygame.quit()
            
        except Exception as e:
            logger.error(f"Error cleaning up visualizer: {e}")
        
    def setup_point_selection(self):
        """Setup point selection interface"""
        self.selected_points = []
        self.selecting_active = False
        self.point_cloud_widget = None
        
        # Create point selection frame
        self.selection_frame = ctk.CTkFrame(self.control_frame)
        self.selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Point selection buttons
        self.start_selection_btn = ctk.CTkButton(
            self.selection_frame,
            text="Start Point Selection",
            command=self.start_point_selection
        )
        self.start_selection_btn.pack(pady=5)
        
        self.clear_points_btn = ctk.CTkButton(
            self.selection_frame,
            text="Clear Points",
            command=self.clear_selected_points
        )
        self.clear_points_btn.pack(pady=5)
        
        # Point coordinates display
        self.points_display = ctk.CTkTextbox(
            self.selection_frame,
            height=100,
            width=200
        )
        self.points_display.pack(pady=5)
        
    def start_point_selection(self):
        """Enable point selection mode"""
        self.selecting_active = True
        self.start_selection_btn.configure(
            text="Selection Active",
            state="disabled"
        )
        
        # Update point cloud widget for selection
        if self.point_cloud_widget:
            def on_click(event):
                if not self.selecting_active:
                    return
                    
                if event.inaxes:
                    # Convert 2D click to 3D coordinates
                    clicked_point = self._get_3d_point(event)
                    if clicked_point is not None:
                        self.selected_points.append(clicked_point)
                        self._update_points_display()
                        self._highlight_selected_point(clicked_point)
                        
                        # If we have enough points (2 for path planning)
                        if len(self.selected_points) >= 2:
                            self.selecting_active = False
                            self.start_selection_btn.configure(
                                text="Start Point Selection",
                                state="normal"
                            )
                            # Trigger path planning
                            self.events.publish('points_selected', self.selected_points)
            
            self.point_cloud_widget.mpl_connect('button_press_event', on_click)
            
    def _get_3d_point(self, event):
        """Convert 2D click to 3D point"""
        try:
            # Get the axes
            ax = event.inaxes
            
            # Get view projection matrix
            proj3d = proj3d
            
            # Get mouse click coordinates
            mouse_x, mouse_y = event.xdata, event.ydata
            
            # Convert 2D coordinates to 3D
            if self.current_view is not None:
                points = np.asarray(self.current_view.points)
                
                # Find nearest point to click
                x2, y2, _ = proj3d.proj_transform(points[:, 0], points[:, 1], 
                                                points[:, 2], ax.get_proj())
                
                # Convert to screen coordinates
                screen_coords = np.column_stack([x2, y2])
                
                # Find nearest point
                distances = np.sqrt(np.sum((screen_coords - 
                                          np.array([mouse_x, mouse_y]))**2, axis=1))
                nearest_point_idx = np.argmin(distances)
                
                return points[nearest_point_idx]
                
        except Exception as e:
            logger.error(f"Error getting 3D point: {e}")
            return None
            
    def _update_points_display(self):
        """Update the display of selected points"""
        self.points_display.delete('1.0', tk.END)
        for i, point in enumerate(self.selected_points):
            point_type = "Start" if i == 0 else "End" if i == 1 else f"Point {i+1}"
            self.points_display.insert(tk.END, 
                f"{point_type}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})\n")
            
    def _highlight_selected_point(self, point):
        """Highlight selected point in visualization"""
        # Create sphere at selected point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        
        # Add to visualization
        self.vis.add_geometry(sphere)
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def clear_selected_points(self):
        """Clear all selected points"""
        self.selected_points.clear()
        self.points_display.delete('1.0', tk.END)
        self.start_selection_btn.configure(
            text="Start Point Selection",
            state="normal"
        )
        self.selecting_active = False
        
        # Refresh visualization
        self.update_visualization(self.current_view)

class VoiceController:
    """Handles voice commands and speech recognition"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.commands = {
            "start scanning": self.start_scanning,
            "stop scanning": self.stop_scanning,
            "calibrate": self.calibrate,
            "find path": self.find_path,
            "save data": self.save_data,
            "load data": self.load_data,
            "clear": self.clear_data,
            "help": self.show_help
        }
        self.is_listening = False
        self.listening_thread = None
        
        # Calibrate microphone for ambient noise
        self.calibrate_microphone()
        
    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info("Microphone calibration complete")
        except Exception as e:
            logger.error(f"Error calibrating microphone: {e}")
            
    def start_listening(self):
        """Start listening for voice commands"""
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        logger.info("Voice control activated")
        
    def stop_listening(self):
        """Stop listening for voice commands"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join()
        logger.info("Voice control deactivated")
        
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    self.process_command(command)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    logger.error(f"Could not request results from speech recognition service: {e}")
                    
            except Exception as e:
                logger.error(f"Error in listening loop: {e}")
                time.sleep(1)
                
    def process_command(self, command):
        """Process recognized voice command"""
        try:
            logger.info(f"Received voice command: {command}")
            
            for cmd, func in self.commands.items():
                if cmd in command:
                    func()
                    return
                    
            logger.info("Command not recognized")
            self.events.publish('voice_feedback', "Command not recognized")
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            
    # Command implementations
    def start_scanning(self):
        self.events.publish('command', 'start_scanning')
        self.events.publish('voice_feedback', "Starting scan")
        
    def stop_scanning(self):
        self.events.publish('command', 'stop_scanning')
        self.events.publish('voice_feedback', "Stopping scan")
        
    def calibrate(self):
        self.events.publish('command', 'calibrate')
        self.events.publish('voice_feedback', "Starting calibration")
        
    def find_path(self):
        self.events.publish('command', 'find_path')
        self.events.publish('voice_feedback', "Finding path")
        
    def save_data(self):
        self.events.publish('command', 'save_data')
        self.events.publish('voice_feedback', "Saving data")
        
    def load_data(self):
        self.events.publish('command', 'load_data')
        self.events.publish('voice_feedback', "Loading data")
        
    def clear_data(self):
        self.events.publish('command', 'clear_data')
        self.events.publish('voice_feedback', "Clearing data")
        
    def show_help(self):
        help_text = "Available commands: " + ", ".join(self.commands.keys())
        self.events.publish('voice_feedback', help_text)

class DataManager:
    """Handles data export and import operations"""
    
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.supported_formats = {
            'ply': self._handle_ply,
            'obj': self._handle_obj,
            'stl': self._handle_stl,
            'fbx': self._handle_fbx,
            'json': self._handle_json
        }
        self.output_dir = Path('output') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_data(self, data, format_type='ply', filename=None):
        """Export data to specified format"""
        try:
            if format_type not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format_type}")
                
            if filename is None:
                filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
                
            filepath = self.output_dir / filename
            
            # Handle export based on format
            self.supported_formats[format_type](data, filepath, mode='export')
            
            logger.info(f"Data exported successfully to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
            
    def import_data(self, filepath):
        """Import data from file"""
        try:
            format_type = filepath.suffix[1:]  # Remove the dot
            if format_type not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Handle import based on format
            data = self.supported_formats[format_type](filepath, None, mode='import')
            
            logger.info(f"Data imported successfully from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error importing data: {e}")
            return None
            
    def _handle_ply(self, data, filepath, mode='export'):
        """Handle PLY format export/import"""
        try:
            if mode == 'export':
                if isinstance(data, o3d.geometry.PointCloud):
                    o3d.io.write_point_cloud(str(filepath), data)
                else:
                    raise ValueError("Data must be a point cloud for PLY format")
            else:  # import
                return o3d.io.read_point_cloud(str(filepath))
        except Exception as e:
            logger.error(f"Error handling PLY format: {e}")
            return None
            
    def _handle_obj(self, data, filepath, mode='export'):
        """Handle OBJ format export/import"""
        try:
            if mode == 'export':
                if isinstance(data, o3d.geometry.TriangleMesh):
                    o3d.io.write_triangle_mesh(str(filepath), data)
                else:
                    # Convert point cloud to mesh if necessary
                    mesh = self._convert_to_mesh(data)
                    o3d.io.write_triangle_mesh(str(filepath), mesh)
            else:  # import
                return o3d.io.read_triangle_mesh(str(filepath))
        except Exception as e:
            logger.error(f"Error handling OBJ format: {e}")
            return None
            
    def _handle_stl(self, data, filepath, mode='export'):
        """Handle STL format export/import"""
        try:
            if mode == 'export':
                if isinstance(data, o3d.geometry.TriangleMesh):
                    o3d.io.write_triangle_mesh(str(filepath), data)
                else:
                    mesh = self._convert_to_mesh(data)
                    o3d.io.write_triangle_mesh(str(filepath), mesh)
            else:  # import
                return o3d.io.read_triangle_mesh(str(filepath))
        except Exception as e:
            logger.error(f"Error handling STL format: {e}")
            return None
            
    def _handle_fbx(self, data, filepath, mode='export'):
        """Handle FBX format export/import"""
        try:
            # Note: FBX handling might require additional libraries
            logger.warning("FBX format support is limited")
            return None
        except Exception as e:
            logger.error(f"Error handling FBX format: {e}")
            return None
            
    def _handle_json(self, data, filepath, mode='export'):
        """Handle JSON format export/import"""
        try:
            if mode == 'export':
                # Convert point cloud to serializable format
                if isinstance(data, o3d.geometry.PointCloud):
                    serialized_data = {
                        'points': np.asarray(data.points).tolist(),
                        'colors': np.asarray(data.colors).tolist() if data.has_colors() else None,
                        'normals': np.asarray(data.normals).tolist() if data.has_normals() else None
                    }
                    with open(filepath, 'w') as f:
                        json.dump(serialized_data, f)
            else:  # import
                with open(filepath, 'r') as f:
                    data = json.load(f)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(data['points']))
                if data['colors']:
                    pcd.colors = o3d.utility.Vector3dVector(np.array(data['colors']))
                if data['normals']:
                    pcd.normals = o3d.utility.Vector3dVector(np.array(data['normals']))
                return pcd
        except Exception as e:
            logger.error(f"Error handling JSON format: {e}")
            return None
            
    def _convert_to_mesh(self, point_cloud):
        """Convert point cloud to mesh"""
        try:
            # Estimate normals if they don't exist
            if not point_cloud.has_normals():
                point_cloud.estimate_normals()
                
            # Create mesh using Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=8)
                
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            return mesh
            
        except Exception as e:
            logger.error(f"Error converting to mesh: {e}")
            return None
            
    def save_metadata(self, metadata):
        """Save scanning metadata"""
        try:
            metadata_file = self.output_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

class RoomMapper:
    """Main application class integrating all components"""
    
    def __init__(self):
        # Initialize configuration and event system
        self.config_manager = ConfigManager()
        self.event_system = EventSystem()
        
        # Initialize all components
        self.initialize_components()
        
        # Set up event handlers
        self.setup_event_handlers()
        
        # Application state
        self.running = False
        self.current_state = ScanningState.IDLE
        self.scan_metadata = {
            'start_time': None,
            'end_time': None,
            'points_collected': 0,
            'room_dimensions': None,
            'calibration_data': None
        }
        
        # Check hardware connections
        self.hardware_status = self.check_hardware_connections()
        
        # Only proceed if minimum required hardware is available
        if not (self.hardware_status['stereo_left'] or self.hardware_status['depth']):
            logger.error("No primary camera detected. At least one camera is required.")
            raise RuntimeError("No camera detected")
        
    def initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize camera system
            self.camera_system = CameraSystem(self.config_manager, self.event_system)
            
            # Initialize point cloud processor
            self.point_cloud_processor = PointCloudProcessor(self.config_manager, self.event_system)
            
            # Initialize ML processor
            self.ml_processor = MLProcessor(self.config_manager, self.event_system)
            
            # Initialize path planner
            self.path_planner = PathPlanner(self.config_manager, self.event_system)
            
            # Initialize visualizer
            self.visualizer = Visualizer(self.config_manager, self.event_system)
            
            # Initialize voice controller
            self.voice_controller = VoiceController(self.config_manager, self.event_system)
            
            # Initialize data manager
            self.data_manager = DataManager(self.config_manager, self.event_system)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
    def setup_event_handlers(self):
        """Set up event handlers for system events"""
        self.event_system.subscribe('new_frames', self.handle_new_frames)
        self.event_system.subscribe('new_thermal_data', self.handle_thermal_data)
        self.event_system.subscribe('command', self.handle_command)
        self.event_system.subscribe('error', self.handle_error)
        self.event_system.subscribe('points_selected', self.handle_points_selected)
        
    def start(self):
        """Start the room mapping system"""
        try:
            logger.info("Starting Room Mapper system")
            self.running = True
            
            # Initialize cameras
            if not self.camera_system.initialize_cameras():
                raise RuntimeError("Failed to initialize cameras")
                
            # Start voice control
            self.voice_controller.start_listening()
            
            # Start camera capture
            self.camera_system.start_capture()
            
            # Start main application loop
            self.main_loop()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.cleanup()
            raise
            
    def main_loop(self):
        """Main application loop"""
        try:
            while self.running:
                if self.current_state == ScanningState.SCANNING:
                    self.process_scan()
                elif self.current_state == ScanningState.PATH_PLANNING:
                    self.process_path_planning()
                    
                # Process any pending events
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.cleanup()
            raise
            
    def process_scan(self):
        """Process scanning operations"""
        try:
            # Get latest frames from queue
            frames = self.camera_system.frame_queue.get_nowait()
            if frames:
                left_frame, right_frame = frames
                
                # Process frames
                point_cloud = self.point_cloud_processor.process_stereo_frames(
                    left_frame, right_frame,
                    self.camera_system.calibration_data
                )
                
                if point_cloud is not None:
                    # Update point cloud
                    self.point_cloud_processor.merge_point_clouds(point_cloud)
                    
                    # Update visualization
                    self.visualizer.update_visualization(
                        point_cloud=self.point_cloud_processor.current_point_cloud
                    )
                    
                    # Update metadata
                    self.scan_metadata['points_collected'] = len(
                        self.point_cloud_processor.current_point_cloud.points
                    )
                    
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing scan: {e}")
            
    def process_path_planning(self):
        """Process path planning operations"""
        try:
            if self.path_planner.path is None:
                # Initialize occupancy grid
                self.path_planner.initialize_occupancy_grid(
                    self.point_cloud_processor.current_point_cloud
                )
                
                # Build navigation graph
                self.path_planner.build_navigation_graph()
                
                # Find path between selected points
                path = self.path_planner.find_path(
                    self.start_point,
                    self.end_point
                )
                
                if path is not None:
                    # Update visualization with path
                    self.visualizer.update_visualization(
                        path=path
                    )
                    
                    # Update state
                    self.current_state = ScanningState.IDLE
                    
        except Exception as e:
            logger.error(f"Error in path planning: {e}")
            self.current_state = ScanningState.ERROR
            
    def handle_new_frames(self, frames):
        """Handle new frames event"""
        try:
            if self.current_state == ScanningState.SCANNING:
                # Perform object detection
                detections = self.ml_processor.detect_objects(frames[0])
                
                # Update AR visualization
                ar_frame = self.visualizer.ar_renderer.create_ar_overlay(
                    frames[0],
                    path=self.path_planner.path,
                    objects=detections
                )
                
                # Update GUI
                self.visualizer.update_camera_feed(ar_frame)
                
        except Exception as e:
            logger.error(f"Error handling new frames: {e}")
            
    def handle_thermal_data(self, thermal_data):
        """Handle thermal data event"""
        try:
            if thermal_data is not None:
                # Process thermal data
                thermal_analysis = self.ml_processor.process_thermal_data(thermal_data)
                
                # Update visualization if needed
                if thermal_analysis and 'anomalies' in thermal_analysis:
                    self.visualizer.update_thermal_overlay(thermal_analysis)
                    
        except Exception as e:
            logger.error(f"Error handling thermal data: {e}")
            
    def handle_command(self, command):
        """Handle voice commands"""
        try:
            if command == 'start_scanning':
                self.start_scanning()
            elif command == 'stop_scanning':
                self.stop_scanning()
            elif command == 'calibrate':
                self.calibrate_system()
            elif command == 'find_path':
                self.start_path_planning()
            elif command == 'save_data':
                self.save_current_data()
            elif command == 'load_data':
                self.load_saved_data()
            elif command == 'clear_data':
                self.clear_current_data()
                
        except Exception as e:
            logger.error(f"Error handling command: {e}")
            
    def handle_error(self, error):
        """Handle error events"""
        logger.error(f"System error: {error}")
        self.current_state = ScanningState.ERROR
        
    def handle_points_selected(self, points):
        """Handle selected points event"""
        try:
            if len(points) >= 2:
                self.start_point = points[0]
                self.end_point = points[1]
                
                # Switch to path planning state
                self.current_state = ScanningState.PATH_PLANNING
                
                # Start path planning process
                self.process_path_planning()
                
        except Exception as e:
            logger.error(f"Error handling selected points: {e}")
            
    def cleanup(self):
        """Clean up system resources"""
        try:
            logger.info("Cleaning up system resources")
            
            # Stop all ongoing operations
            self.running = False
            self.current_state = ScanningState.IDLE
            
            # Clean up components
            self.camera_system.cleanup()
            self.visualizer.cleanup()
            self.voice_controller.stop_listening()
            self.event_system.shutdown()
            
            # Save final metadata
            self.scan_metadata['end_time'] = datetime.now().isoformat()
            self.data_manager.save_metadata(self.scan_metadata)
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def check_hardware_connections(self):
        """Check hardware connections and return status of each component"""
        status = {
            'stereo_left': False,
            'stereo_right': False,
            'thermal': False,
            'depth': False,
            'microphone': False
        }
        
        try:
            # Check stereo cameras
            left_cam = cv2.VideoCapture(self.config.get('cameras', 'left_camera_id'))
            status['stereo_left'] = left_cam.isOpened()
            left_cam.release()
            
            right_cam = cv2.VideoCapture(self.config.get('cameras', 'right_camera_id'))
            status['stereo_right'] = right_cam.isOpened()
            right_cam.release()
            
            # Check thermal camera
            try:
                thermal = serial.Serial(
                    port=self.config.get('cameras', 'thermal_camera_port'),
                    baudrate=self.config.get('cameras', 'thermal_camera_baudrate')
                )
                status['thermal'] = True
                thermal.close()
            except serial.SerialException:
                pass
            
            # Check RealSense depth camera
            try:
                pipeline = rs.pipeline()
                pipeline.start()
                status['depth'] = True
                pipeline.stop()
            except Exception:
                pass
            
            # Check microphone
            try:
                sr.Microphone()
                status['microphone'] = True
            except OSError:
                pass
            
            # Print status report
            print("\nHardware Connection Status:")
            print("-------------------------")
            for device, connected in status.items():
                status_str = "✓ Connected" if connected else "✗ Not Connected"
                print(f"{device.title():12}: {status_str}")
            print("-------------------------\n")
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking hardware connections: {e}")
            return status

def main():
    """Main entry point"""
    try:
        # Create and start room mapper
        mapper = RoomMapper()
        mapper.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
