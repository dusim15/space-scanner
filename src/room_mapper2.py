import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('room_mapper.log'),
        logging.StreamHandler()
    ]
)

class CameraCalibration:
    def __init__(self, checkerboard_size=(9, 6)):
        self.checkerboard_size = checkerboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints_left = []  # 2D points in left image plane
        self.imgpoints_right = []  # 2D points in right image plane
        
    def capture_calibration_frames(self, left_camera, right_camera, num_frames=20):
        """Capture frames for calibration"""
        frames_captured = 0
        
        while frames_captured < num_frames:
            ret1, left_frame = left_camera.read()
            ret2, right_frame = right_camera.read()
            
            if not ret1 or not ret2:
                logging.error("Failed to capture calibration frames")
                return False
                
            gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.checkerboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.checkerboard_size, None)
            
            if ret_left and ret_right:
                # Refine corners
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints_left.append(corners_left)
                self.imgpoints_right.append(corners_right)
                
                frames_captured += 1
                logging.info(f"Captured calibration frame {frames_captured}/{num_frames}")
                
                # Draw and display corners
                cv2.drawChessboardCorners(left_frame, self.checkerboard_size, corners_left, ret_left)
                cv2.drawChessboardCorners(right_frame, self.checkerboard_size, corners_right, ret_right)
                cv2.imshow('Left Calibration', left_frame)
                cv2.imshow('Right Calibration', right_frame)
                
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        return True
        
    def calibrate_cameras(self, left_frame, right_frame):
        """Perform stereo calibration"""
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Calibrate each camera individually
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_left, gray_left.shape[::-1], None, None)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_right, gray_right.shape[::-1], None, None)
            
        # Stereo calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_left, self.imgpoints_right,
            mtx_left, dist_left, mtx_right, dist_right,
            gray_left.shape[::-1], criteria_stereo, flags)
            
        # Stereo rectification
        rect_left, rect_right, proj_mat_left, proj_mat_right, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            gray_left.shape[::-1], R, T)
            
        return {
            'left_matrix': mtx_left,
            'right_matrix': mtx_right,
            'left_distortion': dist_left,
            'right_distortion': dist_right,
            'R': R,
            'T': T,
            'Q': Q
        }

class RoomMapper:
    def __init__(self):
        self.config = self.load_config()
        
        # Initialize cameras
        try:
            self.left_camera = cv2.VideoCapture(self.config['left_camera_id'])
            self.right_camera = cv2.VideoCapture(self.config['right_camera_id'])
        except Exception as e:
            logging.error(f"Failed to initialize cameras: {str(e)}")
            raise
            
        # Camera parameters
        self.calibration = None
        self.focal_length = self.config['focal_length']
        self.baseline = self.config['baseline']
        self.max_depth = self.config['max_depth']
        
        # 3D mapping parameters
        self.voxel_size = self.config['voxel_size']
        self.points = []
        self.colors = []
        self.occupancy_grid = {}
        
        # Path planning parameters
        self.safety_margin = self.config['safety_margin']
        
        # Create output directory
        self.output_dir = Path('output') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def load_config():
        """Load configuration from JSON file"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            config = {
                'left_camera_id': 0,
                'right_camera_id': 1,
                'focal_length': 1000.0,
                'baseline': 0.1,
                'max_depth': 10.0,
                'voxel_size': 0.1,
                'safety_margin': 0.3
            }
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            return config

    def perform_calibration(self):
        """Perform camera calibration"""
        logging.info("Starting camera calibration...")
        calibrator = CameraCalibration()
        
        if calibrator.capture_calibration_frames(self.left_camera, self.right_camera):
            ret1, left_frame = self.left_camera.read()
            ret2, right_frame = self.right_camera.read()
            
            if ret1 and ret2:
                self.calibration = calibrator.calibrate_cameras(left_frame, right_frame)
                
                # Save calibration data
                calibration_file = self.output_dir / 'calibration.json'
                with open(calibration_file, 'w') as f:
                    json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in self.calibration.items()}, f)
                logging.info(f"Calibration data saved to {calibration_file}")
            else:
                logging.error("Failed to capture frames for calibration")
        else:
            logging.error("Calibration frame capture failed")

    def preprocess_frames(self, left_frame, right_frame):
        """Preprocess frames with calibration data"""
        if self.calibration is None:
            return left_frame, right_frame
            
        # Undistort and rectify images
        left_rect = cv2.undistort(left_frame, self.calibration['left_matrix'], 
                                 self.calibration['left_distortion'])
        right_rect = cv2.undistort(right_frame, self.calibration['right_matrix'], 
                                  self.calibration['right_distortion'])
        return left_rect, right_rect

    def compute_disparity(self, left_img, right_img):
        """Enhanced disparity computation with preprocessing"""
        try:
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            left_gray = cv2.equalizeHist(left_gray)
            right_gray = cv2.equalizeHist(right_gray)
            
            # Create stereo matcher with WLS filter
            left_matcher = cv2.StereoBM_create(numDisparities=16*16, blockSize=15)
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(8000)
            wls_filter.setSigmaColor(1.5)
            
            # Compute disparity maps
            left_disp = left_matcher.compute(left_gray, right_gray)
            right_disp = right_matcher.compute(right_gray, left_gray)
            
            # Apply WLS filter
            filtered_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)
            
            return filtered_disp
            
        except Exception as e:
            logging.error(f"Error in disparity computation: {str(e)}")
            return None

    def save_point_cloud(self, points, colors):
        """Save point cloud to file"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        filename = self.output_dir / f'point_cloud_{time.strftime("%Y%m%d_%H%M%S")}.ply'
        o3d.io.write_point_cloud(str(filename), pcd)
        logging.info(f"Point cloud saved to {filename}")

    def interactive_point_selection(self):
        """Enhanced interactive point selection with GUI"""
        if len(self.points) == 0:
            logging.error("No point cloud data available")
            return None, None
            
        def pick_points(pcd):
            print("\nPoint Selection Instructions:")
            print("1. Hold 'Shift' and left click to select points")
            print("2. Select exactly two points: start and end")
            print("3. Press 'Q' to finish selection")
            
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window("Point Selection")
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            
            return vis.get_picked_points()
            
        # Create point cloud for visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors / 255.0)
        
        picked_points = pick_points(pcd)
        
        if len(picked_points) != 2:
            logging.error("Please select exactly two points")
            return None, None
            
        start_point = self.points[picked_points[0]]
        end_point = self.points[picked_points[1]]
        
        return start_point, end_point

    def find_path(self, start_point, end_point):
        """Enhanced path finding with safety checks and optimization"""
        try:
            # Convert points to voxel coordinates
            start_voxel = tuple(np.floor(np.array(start_point) / self.voxel_size).astype(int))
            end_voxel = tuple(np.floor(np.array(end_point) / self.voxel_size).astype(int))
            
            # Create graph with safety margins
            G = nx.Graph()
            safe_voxels = set()
            
            # Mark safe voxels
            for voxel in self.occupancy_grid:
                if not self.occupancy_grid[voxel]:
                    is_safe = True
                    # Check surrounding voxels for safety margin
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                neighbor = (voxel[0]+dx, voxel[1]+dy, voxel[2]+dz)
                                if neighbor in self.occupancy_grid and self.occupancy_grid[neighbor]:
                                    distance = np.sqrt(dx**2 + dy**2 + dz**2) * self.voxel_size
                                    if distance < self.safety_margin:
                                        is_safe = False
                                        break
                    if is_safe:
                        safe_voxels.add(voxel)
            
            # Create edges between safe voxels
            for voxel in safe_voxels:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            neighbor = (voxel[0]+dx, voxel[1]+dy, voxel[2]+dz)
                            if neighbor in safe_voxels:
                                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                                G.add_edge(voxel, neighbor, weight=distance)
            
            # Find path using A* algorithm
            try:
                path = nx.astar_path(G, start_voxel, end_voxel, heuristic=lambda a, b: np.sqrt(
                    sum((x-y)**2 for x, y in zip(a, b))))
                # Convert back to real coordinates
                real_path = [np.array(p) * self.voxel_size for p in path]
                
                # Smooth the path
                smoothed_path = self.smooth_path(real_path)
                return smoothed_path
                
            except nx.NetworkXNoPath:
                logging.error("No valid path found between points")
                return None
                
        except Exception as e:
            logging.error(f"Error in path finding: {str(e)}")
            return None

    def smooth_path(self, path):
        """Smooth the path using spline interpolation"""
        if len(path) < 3:
            return path
            
        try:
            from scipy.interpolate import splprep, splev
            
            # Convert path to numpy array
            path_array = np.array(path)
            
            # Fit spline
            tck, u = splprep([path_array[:, 0], path_array[:, 1], path_array[:, 2]], s=0.0)
            
            # Create more points
            u_new = np.linspace(0, 1, num=len(path) * 5)
            smoothed = np.column_stack(splev(u_new, tck))
            
            return smoothed
            
        except Exception as e:
            logging.warning(f"Path smoothing failed: {str(e)}")
            return path

    def run(self):
        """Main running function with enhanced error handling and visualization"""
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Capture frames
                ret1, left_frame = self.left_camera.read()
                ret2, right_frame = self.right_camera.read()
                
                if not ret1 or not ret2:
                    logging.error("Failed to capture frames")
                    break
                    
                # Preprocess frames
                left_rect, right_rect = self.preprocess_frames(left_frame, right_frame)
                
                # Compute disparity
                disparity = self.compute_disparity(left_rect, right_rect)
                if disparity is None:
                    continue
                    
                # Convert to point cloud
                points, colors = self.disparity_to_pointcloud(disparity, left_rect)
                
                # Update point cloud and occupancy grid
                if len(points) > 0:
                    self.points = np.vstack(self.points, points) if len(self.points) > 0 else points
                    self.colors = np.vstack(self.colors, colors) if len(self.colors) > 0 else colors
                    self.create_occupancy_grid(points)
                
                # Visualize
                vis_frame = left_rect.copy()
                cv2.putText(vis_frame, f"Points: {len(self.points)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Room Mapping', vis_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    fps = frame_count / (time.time() - start_time)
                    logging.info(f"FPS: {fps:.2f}, Total points: {len(self.points)}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_point_cloud(self.points, self.colors)
                
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.left_camera.release()
        self.right_camera.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup completed")

def main():
    try:
        mapper = RoomMapper()
        
        # Perform initial calibration
        mapper.perform_calibration()
        
        # Start scanning
        logging.info("Starting room scanning...")
        print("\nControls:")
        print("'Q': Quit scanning")
        print("'S': Save current point cloud")
        mapper.run()
        
        # Path planning
        if len(mapper.points) > 0:
            logging.info("Starting point selection...")
            start_point, end_point = mapper.interactive_point_selection()
            
            if start_point is not None and end_point is not None:
                path = mapper.find_path(start_point, end_point)
                if path is not None:
                    mapper.visualize(mapper.points, mapper.colors, path)
                    
        logging.info("Program completed successfully")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()