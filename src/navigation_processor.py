from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial import KDTree
from queue import PriorityQueue
import cv2
import time

class NavigationState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY = "recovery"

class TerrainType(Enum):
    FLAT = "flat"
    ROUGH = "rough"
    STAIRS = "stairs"
    SLOPE = "slope"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Pose:
    position: Point3D
    orientation: Point3D  # roll, pitch, yaw

@dataclass
class Path:
    points: List[Point3D]
    costs: List[float]
    terrain_types: List[TerrainType]
    safety_scores: List[float]
    smoothness: float
    total_distance: float
    estimated_time: float

class NavigationProcessor:
    def __init__(self,
                 map_resolution: float = 0.05,  # meters per pixel
                 planning_horizon: float = 10.0,  # meters
                 safety_margin: float = 0.5,  # meters
                 max_planning_time: float = 1.0):  # seconds
        """
        Initialize NavigationProcessor with configuration parameters.
        
        Args:
            map_resolution: Resolution of the navigation map
            planning_horizon: How far ahead to plan
            safety_margin: Minimum distance from obstacles
            max_planning_time: Maximum time allowed for planning
        """
        self.map_resolution = map_resolution
        self.planning_horizon = planning_horizon
        self.safety_margin = safety_margin
        self.max_planning_time = max_planning_time
        
        # Navigation state
        self.state = NavigationState.IDLE
        self.current_pose = None
        self.goal_pose = None
        self.current_path = None
        
        # Initialize maps
        self.occupancy_map = None
        self.cost_map = None
        self.terrain_map = None
        self.elevation_map = None
        
        # Path planning parameters
        self.planning_algorithms = {
            'global': self._astar_planner,
            'local': self._dwa_planner,
            'recovery': self._recovery_planner
        }
        
        # Initialize helper structures
        self._init_helper_structures()
        
    def process(self, navigation_data: Dict) -> Dict:
        """
        Process navigation data and generate path plans
        
        Args:
            navigation_data: Dictionary containing:
                - current_pose: Current robot pose
                - goal_pose: Target pose
                - obstacles: List of detected obstacles
                - terrain_data: Terrain classification data
                - sensor_data: Raw sensor readings
                
        Returns:
            Dictionary containing navigation commands and path data
        """
        try:
            # Update internal state
            self._update_state(navigation_data)
            
            # Update maps
            self._update_maps(navigation_data)
            
            # Check if replanning is needed
            if self._needs_replanning():
                self.state = NavigationState.REPLANNING
            
            # Process based on current state
            if self.state == NavigationState.IDLE:
                return self._process_idle()
            elif self.state == NavigationState.PLANNING:
                return self._process_planning()
            elif self.state == NavigationState.EXECUTING:
                return self._process_executing()
            elif self.state == NavigationState.REPLANNING:
                return self._process_replanning()
            elif self.state == NavigationState.EMERGENCY_STOP:
                return self._process_emergency()
            elif self.state == NavigationState.RECOVERY:
                return self._process_recovery()
            
        except Exception as e:
            print(f"Navigation processing error: {e}")
            return self._generate_emergency_response()
    
    def _init_helper_structures(self):
        """Initialize helper data structures for navigation"""
        # Movement primitives for local planning
        self.movement_primitives = self._generate_movement_primitives()
        
        # Collision checking lookup tables
        self.collision_lookup = self._init_collision_lookup()
        
        # Cost function weights
        self.weights = {
            'distance': 0.3,
            'smoothness': 0.2,
            'safety': 0.3,
            'terrain': 0.2
        }
        
        # Recovery behaviors
        self.recovery_behaviors = [
            self._simple_backup,
            self._rotate_in_place,
            self._spiral_search
        ]
    
    def _update_state(self, navigation_data: Dict):
        """Update internal state with new navigation data"""
        self.current_pose = navigation_data['current_pose']
        
        if 'goal_pose' in navigation_data:
            self.goal_pose = navigation_data['goal_pose']
            if self.state == NavigationState.IDLE:
                self.state = NavigationState.PLANNING
    
    def _update_maps(self, navigation_data: Dict):
        """Update all navigation maps with new sensor data"""
        # Update occupancy grid
        self._update_occupancy_map(navigation_data['obstacles'])
        
        # Update elevation map
        self._update_elevation_map(navigation_data['sensor_data'])
        
        # Update terrain classification
        self._update_terrain_map(navigation_data['terrain_data'])
        
        # Update cost map
        self._update_cost_map()
    
    def _astar_planner(self, start: Point3D, goal: Point3D) -> Optional[Path]:
        """A* path planning implementation"""
        def heuristic(a: Point3D, b: Point3D) -> float:
            return np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2)
        
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {self._point_to_key(start): None}
        cost_so_far = {self._point_to_key(start): 0}
        
        while not frontier.empty():
            current = frontier.get()[1]
            
            if self._points_close(current, goal):
                return self._reconstruct_path(came_from, start, current)
            
            for next_point in self._get_neighbors(current):
                new_cost = cost_so_far[self._point_to_key(current)] + self._movement_cost(current, next_point)
                
                if (self._point_to_key(next_point) not in cost_so_far or 
                    new_cost < cost_so_far[self._point_to_key(next_point)]):
                    cost_so_far[self._point_to_key(next_point)] = new_cost
                    priority = new_cost + heuristic(goal, next_point)
                    frontier.put((priority, next_point))
                    came_from[self._point_to_key(next_point)] = current
        
        return None
    
    def _dwa_planner(self, current_pose: Pose, goal: Point3D) -> Optional[Path]:
        """Dynamic Window Approach for local planning"""
        best_path = None
        best_score = float('-inf')
        
        for velocity in self._generate_velocity_samples():
            for angular_velocity in self._generate_angular_samples():
                # Simulate trajectory
                path = self._simulate_trajectory(current_pose, velocity, angular_velocity)
                
                # Score trajectory
                score = self._score_trajectory(path, goal)
                
                if score > best_score:
                    best_score = score
                    best_path = path
        
        return best_path
    
    def _recovery_planner(self) -> Optional[Path]:
        """Plan recovery behavior when stuck"""
        for behavior in self.recovery_behaviors:
            path = behavior()
            if path and self._is_path_safe(path):
                return path
        return None
    
    def _update_occupancy_map(self, obstacles: List[Dict]):
        """Update occupancy grid with new obstacle data"""
        if self.occupancy_map is None:
            self._initialize_maps()
            
        for obstacle in obstacles:
            self._add_obstacle_to_map(obstacle)
            
        # Apply morphological operations for better coverage
        self.occupancy_map = cv2.dilate(
            self.occupancy_map,
            kernel=np.ones((3, 3), np.uint8)
        )
    
    def _update_elevation_map(self, sensor_data: Dict):
        """Update elevation map with new sensor readings"""
        if 'point_cloud' in sensor_data:
            points = sensor_data['point_cloud']
            for point in points:
                self._update_elevation_cell(point)
    
    def _update_terrain_map(self, terrain_data: Dict):
        """Update terrain classification map"""
        if 'classifications' in terrain_data:
            for classification in terrain_data['classifications']:
                self._update_terrain_cell(classification)
    
    def _update_cost_map(self):
        """Update cost map based on all other maps"""
        if self.cost_map is None:
            self._initialize_maps()
            
        # Combine different cost components
        obstacle_cost = self._compute_obstacle_cost()
        terrain_cost = self._compute_terrain_cost()
        elevation_cost = self._compute_elevation_cost()
        
        # Weight and combine costs
        self.cost_map = (
            self.weights['safety'] * obstacle_cost +
            self.weights['terrain'] * terrain_cost +
            self.weights['smoothness'] * elevation_cost
        )
    
    def _needs_replanning(self) -> bool:
        """Determine if replanning is necessary"""
        if not self.current_path:
            return True
            
        # Check for new obstacles in path
        if self._path_blocked(self.current_path):
            return True
            
        # Check if we're too far from planned path
        if self._deviation_from_path() > self.safety_margin:
            return True
            
        return False
    
    def _process_idle(self) -> Dict:
        """Process idle state"""
        return {
            'command': 'IDLE',
            'status': 'waiting_for_goal',
            'path': None
        }
    
    def _process_planning(self) -> Dict:
        """Process planning state"""
        # Attempt global planning
        path = self._astar_planner(
            self.current_pose.position,
            self.goal_pose.position
        )
        
        if path:
            self.current_path = path
            self.state = NavigationState.EXECUTING
            return {
                'command': 'FOLLOW_PATH',
                'path': path,
                'status': 'path_found'
            }
        else:
            self.state = NavigationState.RECOVERY
            return {
                'command': 'STOP',
                'status': 'planning_failed',
                'path': None
            }
    
    def _process_executing(self) -> Dict:
        """Process path execution state"""
        # Get local plan
        local_path = self._dwa_planner(
            self.current_pose,
            self._get_local_goal()
        )
        
        if not local_path:
            self.state = NavigationState.REPLANNING
            return {
                'command': 'STOP',
                'status': 'local_planning_failed',
                'path': self.current_path
            }
            
        return {
            'command': 'FOLLOW_PATH',
            'path': local_path,
            'status': 'executing',
            'progress': self._calculate_progress()
        }
    
    def _process_replanning(self) -> Dict:
        """Process replanning state"""
        # Similar to planning but maintains more state
        return self._process_planning()
    
    def _process_emergency(self) -> Dict:
        """Process emergency stop state"""
        return {
            'command': 'EMERGENCY_STOP',
            'status': 'emergency',
            'path': None
        }
    
    def _process_recovery(self) -> Dict:
        """Process recovery state"""
        recovery_path = self._recovery_planner()
        
        if recovery_path:
            return {
                'command': 'FOLLOW_PATH',
                'path': recovery_path,
                'status': 'recovering'
            }
        else:
            return {
                'command': 'STOP',
                'status': 'recovery_failed',
                'path': None
            }
    
    def _generate_emergency_response(self) -> Dict:
        """Generate emergency response on error"""
        self.state = NavigationState.EMERGENCY_STOP
        return {
            'command': 'EMERGENCY_STOP',
            'status': 'error',
            'path': None
        }

    # Helper methods for path planning
    def _point_to_key(self, point: Point3D) -> str:
        """Convert point to string key for dictionaries"""
        return f"{point.x:.3f},{point.y:.3f},{point.z:.3f}"
    
    def _points_close(self, a: Point3D, b: Point3D, threshold: float = 0.1) -> bool:
        """Check if two points are close enough"""
        return np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2) < threshold
    
    def _get_neighbors(self, point: Point3D) -> List[Point3D]:
        """Get valid neighboring points for path planning"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = Point3D(
                        point.x + dx * self.map_resolution,
                        point.y + dy * self.map_resolution,
                        point.z + dz * self.map_resolution
                    )
                    if self._is_point_valid(neighbor):
                        neighbors.append(neighbor)
        return neighbors
    
    def _movement_cost(self, a: Point3D, b: Point3D) -> float:
        """Calculate cost of movement between points"""
        # Base cost is distance
        cost = np.sqrt((b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2)
        
        # Add terrain cost
        terrain_cost = self._get_terrain_cost(b)
        
        # Add elevation cost
        elevation_cost = self._get_elevation_cost(a, b)
        
        return cost * (1 + terrain_cost + elevation_cost)
    
    def _reconstruct_path(self, came_from: Dict, start: Point3D, goal: Point3D) -> Path:
        """Reconstruct path from A* search result"""
        current = goal
        path_points = []
        costs = []
        terrain_types = []
        safety_scores = []
        
        while current != start:
            path_points.append(current)
            costs.append(self._movement_cost(current, came_from[self._point_to_key(current)]))
            terrain_types.append(self._get_terrain_type(current))
            safety_scores.append(self._calculate_safety_score(current))
            current = came_from[self._point_to_key(current)]
            
        path_points.append(start)
        path_points.reverse()
        costs.reverse()
        terrain_types.reverse()
        safety_scores.reverse()
        
        return Path(
            points=path_points,
            costs=costs,
            terrain_types=terrain_types,
            safety_scores=safety_scores,
            smoothness=self._calculate_path_smoothness(path_points),
            total_distance=sum(costs),
            estimated_time=self._estimate_traversal_time(path_points, costs)
        ) 