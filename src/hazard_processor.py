from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import cv2
from sklearn.ensemble import IsolationForest

class HazardType(Enum):
    CHEMICAL = "chemical"
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    RADIATION = "radiation"
    BIOLOGICAL = "biological"
    PRESSURE = "pressure"
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"
    MAGNETIC = "magnetic"

class HazardSeverity(Enum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CRITICAL = 5

class ExposureLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

@dataclass
class ChemicalProperties:
    ph_level: float
    reactivity: float
    flash_point: float
    auto_ignition_temp: float
    explosive_limits: Tuple[float, float]
    oxidizing_potential: float

@dataclass
class ThermalProperties:
    temperature: float
    heat_flux: float
    thermal_conductivity: float
    specific_heat: float
    thermal_expansion: float

@dataclass
class RadiationProperties:
    radiation_type: str
    intensity: float
    wavelength: Optional[float]
    exposure_time: float
    cumulative_dose: float

@dataclass
class HazardZone:
    center: Tuple[float, float, float]
    radius: float
    hazard_type: HazardType
    severity: HazardSeverity
    exposure_level: ExposureLevel
    timestamp: datetime
    properties: Dict

@dataclass
class HazardAlert:
    hazard_type: HazardType
    severity: HazardSeverity
    location: Tuple[float, float, float]
    description: str
    recommended_action: str
    timestamp: datetime
    ttl: float  # Time-to-live in seconds

class HazardProcessor:
    def __init__(self, 
                 config: Dict = None,
                 enable_ml: bool = True,
                 alert_threshold: float = 0.75):
        """
        Initialize HazardProcessor with configuration parameters.
        
        Args:
            config: Configuration dictionary
            enable_ml: Enable machine learning for anomaly detection
            alert_threshold: Threshold for hazard alerts (0-1)
        """
        self.config = config or {}
        self.enable_ml = enable_ml
        self.alert_threshold = alert_threshold
        
        # Initialize hazard tracking
        self.active_hazards: Dict[str, HazardZone] = {}
        self.hazard_history: List[HazardZone] = []
        self.active_alerts: List[HazardAlert] = []
        
        # Initialize detection thresholds
        self.thresholds = self._init_thresholds()
        
        # Initialize ML models
        if self.enable_ml:
            self._init_ml_models()
            
        # Initialize sensor calibration data
        self.sensor_calibration = {}
        
        # Initialize safety limits
        self.safety_limits = self._init_safety_limits()
        
        # Track cumulative exposure
        self.exposure_tracking = {}

    def process(self, sensor_data: Dict) -> Dict:
        """
        Process sensor data to identify and analyze hazards
        
        Args:
            sensor_data: Dictionary containing sensor readings including:
                - chemical_sensors: Chemical sensor data
                - thermal_sensors: Temperature and heat flux data
                - radiation_sensors: Radiation detector data
                - pressure_sensors: Pressure sensor data
                - electromagnetic_sensors: EM field sensor data
                - acoustic_sensors: Sound level data
                - optical_sensors: Light level and laser detection data
                
        Returns:
            Dictionary containing hazard analysis
        """
        try:
            # Update sensor calibration
            self._update_calibration(sensor_data)
            
            # Detect hazards from various sensors
            chemical_hazards = self._process_chemical_hazards(sensor_data)
            thermal_hazards = self._process_thermal_hazards(sensor_data)
            radiation_hazards = self._process_radiation_hazards(sensor_data)
            mechanical_hazards = self._process_mechanical_hazards(sensor_data)
            electrical_hazards = self._process_electrical_hazards(sensor_data)
            
            # Combine all detected hazards
            all_hazards = self._combine_hazards([
                chemical_hazards,
                thermal_hazards,
                radiation_hazards,
                mechanical_hazards,
                electrical_hazards
            ])
            
            # Update hazard tracking
            self._update_hazard_tracking(all_hazards)
            
            # Perform risk assessment
            risk_assessment = self._assess_risks()
            
            # Generate alerts
            alerts = self._generate_alerts(risk_assessment)
            
            # Update exposure tracking
            self._update_exposure_tracking()
            
            return {
                'active_hazards': self.active_hazards,
                'risk_assessment': risk_assessment,
                'alerts': alerts,
                'safe_zones': self._identify_safe_zones(),
                'exposure_levels': self.exposure_tracking,
                'recommended_actions': self._get_recommended_actions()
            }
            
        except Exception as e:
            print(f"Hazard processing error: {e}")
            return None

    def _init_ml_models(self):
        """Initialize machine learning models for anomaly detection"""
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.pattern_recognizer = None  # Would be initialized with actual ML model
        self.trend_analyzer = None      # Would be initialized with actual ML model

    def _init_thresholds(self) -> Dict:
        """Initialize detection thresholds for different hazard types"""
        return {
            HazardType.CHEMICAL: {
                'ph_min': 6.0,
                'ph_max': 8.0,
                'reactivity_threshold': 0.7,
                'toxicity_threshold': 0.5
            },
            HazardType.THERMAL: {
                'max_temp': 60.0,  # °C
                'max_heat_flux': 1000.0,  # W/m²
                'rate_of_change_threshold': 5.0  # °C/s
            },
            HazardType.RADIATION: {
                'max_dose_rate': 0.1,  # Sv/h
                'cumulative_dose_limit': 1.0,  # Sv
                'neutron_flux_threshold': 1000.0  # n/cm²/s
            },
            # Add more hazard thresholds...
        }

    def _init_safety_limits(self) -> Dict:
        """Initialize safety limits for various parameters"""
        return {
            'exposure_time_limits': {
                HazardType.RADIATION: 3600,  # 1 hour
                HazardType.CHEMICAL: 300,    # 5 minutes
                HazardType.THERMAL: 60       # 1 minute
            },
            'distance_limits': {
                HazardType.RADIATION: 5.0,  # meters
                HazardType.CHEMICAL: 3.0,
                HazardType.THERMAL: 2.0
            },
            'concentration_limits': {
                'oxygen_min': 19.5,  # %
                'oxygen_max': 23.5,
                'co2_max': 5000.0,   # ppm
                'co_max': 50.0       # ppm
            }
        }

    def _process_chemical_hazards(self, sensor_data: Dict) -> List[HazardZone]:
        """Process chemical sensor data for hazards"""
        hazards = []
        chemical_data = sensor_data.get('chemical_sensors', {})
        
        if not chemical_data:
            return hazards
            
        for location, readings in chemical_data.items():
            # Check pH levels
            if 'ph' in readings:
                if readings['ph'] < self.thresholds[HazardType.CHEMICAL]['ph_min'] or \
                   readings['ph'] > self.thresholds[HazardType.CHEMICAL]['ph_max']:
                    hazards.append(self._create_chemical_hazard(location, readings))
                    
            # Check reactivity
            if 'reactivity' in readings and \
               readings['reactivity'] > self.thresholds[HazardType.CHEMICAL]['reactivity_threshold']:
                hazards.append(self._create_chemical_hazard(location, readings))
                
        return hazards

    def _process_thermal_hazards(self, sensor_data: Dict) -> List[HazardZone]:
        """Process thermal sensor data for hazards"""
        hazards = []
        thermal_data = sensor_data.get('thermal_sensors', {})
        
        if not thermal_data:
            return hazards
            
        for location, readings in thermal_data.items():
            if readings['temperature'] > self.thresholds[HazardType.THERMAL]['max_temp']:
                hazards.append(self._create_thermal_hazard(location, readings))
                
            # Check rate of change
            if 'temperature_history' in readings:
                rate = self._calculate_rate_of_change(readings['temperature_history'])
                if abs(rate) > self.thresholds[HazardType.THERMAL]['rate_of_change_threshold']:
                    hazards.append(self._create_thermal_hazard(location, readings))
                    
        return hazards

    def _process_radiation_hazards(self, sensor_data: Dict) -> List[HazardZone]:
        """Process radiation sensor data for hazards"""
        hazards = []
        radiation_data = sensor_data.get('radiation_sensors', {})
        
        if not radiation_data:
            return hazards
            
        for location, readings in radiation_data.items():
            if readings['dose_rate'] > self.thresholds[HazardType.RADIATION]['max_dose_rate']:
                hazards.append(self._create_radiation_hazard(location, readings))
                
        return hazards

    def _update_hazard_tracking(self, hazards: List[HazardZone]):
        """Update tracking of active hazards"""
        current_time = datetime.now()
        
        # Update active hazards
        new_active_hazards = {}
        for hazard in hazards:
            hazard_id = self._generate_hazard_id(hazard)
            new_active_hazards[hazard_id] = hazard
            
        # Archive old hazards
        for hazard_id, hazard in self.active_hazards.items():
            if hazard_id not in new_active_hazards:
                self.hazard_history.append(hazard)
                
        self.active_hazards = new_active_hazards

    def _assess_risks(self) -> Dict:
        """Perform comprehensive risk assessment"""
        risk_assessment = {
            'overall_risk': self._calculate_overall_risk(),
            'hazard_interactions': self._analyze_hazard_interactions(),
            'exposure_risks': self._analyze_exposure_risks(),
            'trend_analysis': self._analyze_hazard_trends()
        }
        
        return risk_assessment

    def _generate_alerts(self, risk_assessment: Dict) -> List[HazardAlert]:
        """Generate hazard alerts based on risk assessment"""
        new_alerts = []
        
        for hazard_id, hazard in self.active_hazards.items():
            if hazard.severity.value >= HazardSeverity.MODERATE.value:
                alert = HazardAlert(
                    hazard_type=hazard.hazard_type,
                    severity=hazard.severity,
                    location=hazard.center,
                    description=self._generate_hazard_description(hazard),
                    recommended_action=self._get_hazard_action(hazard),
                    timestamp=datetime.now(),
                    ttl=300.0  # 5 minutes
                )
                new_alerts.append(alert)
                
        return new_alerts

    def _identify_safe_zones(self) -> List[Dict]:
        """Identify safe zones away from hazards"""
        # Implementation would depend on spatial analysis algorithm
        return []

    def _update_exposure_tracking(self):
        """Update cumulative exposure tracking"""
        current_time = datetime.now()
        
        for hazard in self.active_hazards.values():
            if hazard.hazard_type not in self.exposure_tracking:
                self.exposure_tracking[hazard.hazard_type] = 0.0
                
            # Update cumulative exposure
            exposure_duration = (current_time - hazard.timestamp).total_seconds()
            self.exposure_tracking[hazard.hazard_type] += \
                exposure_duration * hazard.exposure_level.value

    def _get_recommended_actions(self) -> List[str]:
        """Get recommended actions based on current hazards"""
        actions = []
        
        for hazard in self.active_hazards.values():
            if hazard.severity >= HazardSeverity.MAJOR:
                actions.append(f"EVACUATE from {hazard.hazard_type.value} hazard")
            elif hazard.severity >= HazardSeverity.MODERATE:
                actions.append(f"CAUTION: Avoid {hazard.hazard_type.value} area")
                
        return actions

    def _generate_hazard_id(self, hazard: HazardZone) -> str:
        """Generate unique identifier for hazard"""
        return f"{hazard.hazard_type.value}_{hash(hazard.center)}_{hazard.timestamp.timestamp()}"

    def _calculate_overall_risk(self) -> float:
        """Calculate overall risk level from all active hazards"""
        if not self.active_hazards:
            return 0.0
            
        return max(hazard.severity.value for hazard in self.active_hazards.values()) / 5.0

    def _analyze_hazard_interactions(self) -> List[Dict]:
        """Analyze potential interactions between different hazards"""
        interactions = []
        
        hazard_list = list(self.active_hazards.values())
        for i, hazard1 in enumerate(hazard_list):
            for hazard2 in hazard_list[i+1:]:
                if self._check_hazard_interaction(hazard1, hazard2):
                    interactions.append({
                        'hazard1': hazard1,
                        'hazard2': hazard2,
                        'risk_multiplier': 1.5
                    })
                    
        return interactions

    def _check_hazard_interaction(self, hazard1: HazardZone, hazard2: HazardZone) -> bool:
        """Check if two hazards can interact dangerously"""
        # Implementation would include specific interaction rules
        return False

    def _generate_hazard_description(self, hazard: HazardZone) -> str:
        """Generate human-readable hazard description"""
        return f"{hazard.severity.value} {hazard.hazard_type.value} hazard detected at location {hazard.center}"

    def _get_hazard_action(self, hazard: HazardZone) -> str:
        """Get recommended action for specific hazard"""
        if hazard.severity >= HazardSeverity.CRITICAL:
            return "IMMEDIATE EVACUATION REQUIRED"
        elif hazard.severity >= HazardSeverity.MAJOR:
            return "CLEAR AREA IMMEDIATELY"
        elif hazard.severity >= HazardSeverity.MODERATE:
            return "MAINTAIN SAFE DISTANCE"
        return "MONITOR SITUATION" 