import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

class GasType(Enum):
    CO2 = "Carbon Dioxide"
    CO = "Carbon Monoxide"
    NO2 = "Nitrogen Dioxide"
    SO2 = "Sulfur Dioxide"
    O3 = "Ozone"
    NH3 = "Ammonia"
    CH4 = "Methane"
    H2S = "Hydrogen Sulfide"
    VOC = "Volatile Organic Compounds"

@dataclass
class SensorReading:
    timestamp: datetime
    concentration: float  # ppm
    temperature: float   # °C
    humidity: float     # %
    pressure: float     # hPa
    flow_rate: float    # L/min
    status: str
    confidence: float   # 0-1

class SensorStatus(Enum):
    NORMAL = "Normal Operation"
    WARMING_UP = "Warming Up"
    CALIBRATING = "Calibrating"
    ERROR = "Error"
    MAINTENANCE_NEEDED = "Maintenance Needed"

class GasSensorArray:
    def __init__(self, 
                 sampling_rate: float = 1.0,  # Hz
                 temperature_compensation: bool = True,
                 self_calibration: bool = True):
        """
        Initialize Gas Sensor Array with multiple gas sensors.
        
        Args:
            sampling_rate: Measurements per second
            temperature_compensation: Enable temperature compensation
            self_calibration: Enable automatic calibration
        """
        self.sampling_rate = sampling_rate
        self.temperature_compensation = temperature_compensation
        self.self_calibration = self_calibration
        
        # Initialize sensors for different gases
        self.sensors = {gas: self._initialize_sensor(gas) for gas in GasType}
        
        # Environmental conditions
        self.ambient_temperature = 25.0  # °C
        self.ambient_humidity = 45.0     # %
        self.ambient_pressure = 1013.25  # hPa
        
        # Calibration data
        self.last_calibration = datetime.now()
        self.calibration_due = False
        
        # Sensor characteristics
        self.detection_limits = {
            GasType.CO2: (0.0, 5000.0),    # ppm
            GasType.CO: (0.0, 1000.0),     # ppm
            GasType.NO2: (0.0, 20.0),      # ppm
            GasType.SO2: (0.0, 20.0),      # ppm
            GasType.O3: (0.0, 10.0),       # ppm
            GasType.NH3: (0.0, 100.0),     # ppm
            GasType.CH4: (0.0, 50000.0),   # ppm
            GasType.H2S: (0.0, 100.0),     # ppm
            GasType.VOC: (0.0, 1000.0)     # ppm
        }

    def get_readings(self, 
                    gases: Optional[List[GasType]] = None) -> Dict[str, SensorReading]:
        """
        Get current readings from specified gas sensors.
        
        Args:
            gases: List of gases to measure, or None for all gases
            
        Returns:
            Dictionary of gas readings with concentrations and metadata
        """
        if gases is None:
            gases = list(GasType)
            
        readings = {}
        current_time = datetime.now()
        
        for gas in gases:
            # Apply temperature compensation if enabled
            temp_factor = self._calculate_temperature_compensation() if self.temperature_compensation else 1.0
            
            # Generate realistic reading with noise and drift
            base_concentration = self._generate_realistic_concentration(gas)
            adjusted_concentration = base_concentration * temp_factor
            
            # Calculate flow rate with fluctuations
            flow_rate = 2.0 + np.random.normal(0, 0.1)
            
            readings[gas.name] = SensorReading(
                timestamp=current_time,
                concentration=adjusted_concentration,
                temperature=self.ambient_temperature + np.random.normal(0, 0.1),
                humidity=self.ambient_humidity + np.random.normal(0, 0.5),
                pressure=self.ambient_pressure + np.random.normal(0, 0.1),
                flow_rate=max(0, flow_rate),
                status=self._get_sensor_status(gas).value,
                confidence=self._calculate_confidence(gas, adjusted_concentration)
            )
            
        return readings

    def analyze_composition(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze gas composition with detailed breakdown.
        
        Returns:
            Dictionary containing composition analysis and quality metrics
        """
        readings = self.get_readings()
        total_concentration = sum(reading.concentration for reading in readings.values())
        
        composition = {
            'composition': {
                gas: (reading.concentration / total_concentration * 100 if total_concentration > 0 else 0)
                for gas, reading in readings.items()
            },
            'metrics': {
                'total_concentration': total_concentration,
                'measurement_quality': self._calculate_measurement_quality(),
                'cross_sensitivity': self._estimate_cross_sensitivity(),
                'stability': self._calculate_stability()
            }
        }
        
        return composition

    def calibrate(self, gas_type: Optional[GasType] = None) -> bool:
        """
        Calibrate specified or all gas sensors.
        
        Args:
            gas_type: Specific gas to calibrate, or None for all gases
            
        Returns:
            Success status of calibration
        """
        try:
            if gas_type:
                self._calibrate_sensor(gas_type)
            else:
                for gas in GasType:
                    self._calibrate_sensor(gas)
                    
            self.last_calibration = datetime.now()
            self.calibration_due = False
            return True
            
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            return False

    def set_environmental_conditions(self, 
                                  temperature: float,
                                  humidity: float,
                                  pressure: float) -> None:
        """Set ambient environmental conditions."""
        self.ambient_temperature = temperature
        self.ambient_humidity = humidity
        self.ambient_pressure = pressure

    def get_sensor_status(self) -> Dict[str, Dict[str, any]]:
        """Get detailed status of all sensors."""
        return {
            gas.name: {
                'status': self._get_sensor_status(gas).value,
                'last_calibration': self.last_calibration,
                'calibration_due': self.calibration_due,
                'detection_limits': self.detection_limits[gas],
                'drift': self._calculate_drift(gas),
                'noise_level': self._calculate_noise_level(gas)
            }
            for gas in GasType
        }

    def _initialize_sensor(self, gas_type: GasType) -> Dict:
        """Initialize individual gas sensor with default parameters."""
        return {
            'baseline': self._get_default_baseline(gas_type),
            'sensitivity': self._get_default_sensitivity(gas_type),
            'cross_sensitivity': self._get_cross_sensitivity_matrix(gas_type),
            'drift': 0.0,
            'noise': 0.0
        }

    def _generate_realistic_concentration(self, gas_type: GasType) -> float:
        """Generate realistic gas concentration with appropriate noise and trends."""
        base_level = {
            GasType.CO2: 400.0,
            GasType.CO: 0.2,
            GasType.NO2: 0.02,
            GasType.SO2: 0.01,
            GasType.O3: 0.03,
            GasType.NH3: 0.1,
            GasType.CH4: 1.8,
            GasType.H2S: 0.01,
            GasType.VOC: 0.1
        }[gas_type]
        
        # Add random fluctuations and drift
        noise = np.random.normal(0, base_level * 0.05)
        drift = self.sensors[gas_type]['drift']
        
        concentration = base_level + noise + drift
        return max(0, min(concentration, self.detection_limits[gas_type][1]))

    def _calculate_temperature_compensation(self) -> float:
        """Calculate temperature compensation factor."""
        reference_temp = 25.0
        temp_coefficient = 0.015  # %/°C
        return 1.0 + (self.ambient_temperature - reference_temp) * temp_coefficient

    def _calculate_confidence(self, gas_type: GasType, concentration: float) -> float:
        """Calculate measurement confidence based on various factors."""
        confidence = 1.0
        
        # Reduce confidence near detection limits
        lower_limit, upper_limit = self.detection_limits[gas_type]
        if concentration < lower_limit * 1.1 or concentration > upper_limit * 0.9:
            confidence *= 0.7
            
        # Reduce confidence based on environmental conditions
        if abs(self.ambient_temperature - 25.0) > 10:
            confidence *= 0.9
        if self.ambient_humidity > 80:
            confidence *= 0.8
            
        # Reduce confidence if calibration is due
        if self.calibration_due:
            confidence *= 0.7
            
        return max(0.0, min(1.0, confidence))

    def _get_sensor_status(self, gas_type: GasType) -> SensorStatus:
        """Determine current sensor status."""
        if (datetime.now() - self.last_calibration).days > 30:
            return SensorStatus.MAINTENANCE_NEEDED
        if self._calculate_noise_level(gas_type) > 0.1:
            return SensorStatus.ERROR
        return SensorStatus.NORMAL

    def _calibrate_sensor(self, gas_type: GasType) -> None:
        """Perform sensor calibration routine."""
        time.sleep(2)  # Simulate calibration time
        self.sensors[gas_type]['baseline'] = self._get_default_baseline(gas_type)
        self.sensors[gas_type]['drift'] = 0.0

    def _calculate_measurement_quality(self) -> float:
        """Calculate overall measurement quality score."""
        return np.mean([
            self._calculate_confidence(gas, 0.0)
            for gas in GasType
        ])

    def _estimate_cross_sensitivity(self) -> float:
        """Estimate cross-sensitivity between gases."""
        return np.random.uniform(0.02, 0.05)

    def _calculate_stability(self) -> float:
        """Calculate measurement stability over time."""
        return np.random.uniform(0.90, 0.99)

    def _calculate_drift(self, gas_type: GasType) -> float:
        """Calculate sensor drift since last calibration."""
        days_since_calibration = (datetime.now() - self.last_calibration).days
        return days_since_calibration * 0.001 * self.detection_limits[gas_type][1]

    def _calculate_noise_level(self, gas_type: GasType) -> float:
        """Calculate current noise level in measurements."""
        return np.random.uniform(0.01, 0.05)

    def _get_default_baseline(self, gas_type: GasType) -> float:
        """Get default baseline value for gas type."""
        return self.detection_limits[gas_type][0]

    def _get_default_sensitivity(self, gas_type: GasType) -> float:
        """Get default sensitivity for gas type."""
        return 1.0

    def _get_cross_sensitivity_matrix(self, gas_type: GasType) -> Dict[GasType, float]:
        """Get cross-sensitivity coefficients for other gases."""
        return {
            other_gas: np.random.uniform(0.01, 0.05)
            for other_gas in GasType if other_gas != gas_type
        }