import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

@dataclass
class SpectralPeak:
    wavenumber: float  # cm^-1
    intensity: float
    fwhm: float  # Full Width at Half Maximum
    area: float
    snr: float  # Signal-to-Noise Ratio

class LaserType(Enum):
    UV_266nm = 266
    VIOLET_405nm = 405
    BLUE_488nm = 488
    GREEN_532nm = 532
    RED_785nm = 785
    NIR_1064nm = 1064

class RamanSpectrometer:
    def __init__(self, 
                 laser_type: LaserType = LaserType.GREEN_532nm,
                 power_mw: float = 100.0,
                 spectral_range: Tuple[float, float] = (100, 3200),
                 resolution: float = 0.5):
        """
        Initialize Raman Spectrometer with specific configuration.
        
        Args:
            laser_type: Excitation laser wavelength
            power_mw: Laser power in milliwatts
            spectral_range: Range of wavenumbers to scan (cm^-1)
            resolution: Spectral resolution in cm^-1
        """
        self.laser_type = laser_type
        self.power_mw = power_mw
        self.spectral_range = spectral_range
        self.resolution = resolution
        self.spectrum_data = {}
        self.is_calibrated = False
        self.temperature = 20.0  # °C
        
        # Initialize wavelength axis
        self.wavenumbers = np.arange(
            spectral_range[0],
            spectral_range[1],
            resolution
        )
        
        # Known reference peaks for calibration (e.g., silicon)
        self.reference_peaks = {
            'silicon': 520.7,
            'diamond': 1332.5,
            'polystyrene': [620.9, 1001.4, 1031.8, 1602.3]
        }

    def calibrate(self) -> bool:
        """Perform wavelength calibration using reference standards."""
        try:
            # Simulate calibration process
            time.sleep(2)  # Simulated measurement time
            self.is_calibrated = True
            return True
        except Exception as e:
            print(f"Calibration failed: {str(e)}")
            return False

    def capture_spectrum(self, 
                        integration_time: float = 1.0,
                        averages: int = 10) -> Dict[str, np.ndarray]:
        """
        Capture Raman spectrum with specified parameters.
        
        Args:
            integration_time: Exposure time in seconds
            averages: Number of spectra to average
            
        Returns:
            Dictionary containing wavelength and intensity arrays
        """
        if not self.is_calibrated:
            raise RuntimeError("Spectrometer must be calibrated before measurement")

        # Simulate realistic Raman spectrum
        intensities = self._generate_simulated_spectrum()
        
        # Add noise based on integration time and averaging
        noise = np.random.normal(0, 1/np.sqrt(averages), len(self.wavenumbers))
        noise *= 0.02 * np.max(intensities) * np.sqrt(1/integration_time)
        
        final_spectrum = intensities + noise
        
        self.spectrum_data = {
            'wavenumber': self.wavenumbers,
            'intensity': final_spectrum
        }
        
        return self.spectrum_data

    def analyze_composition(self) -> Dict[str, float]:
        """
        Analyze the chemical composition based on the captured spectrum.
        
        Returns:
            Dictionary of identified compounds and their concentrations
        """
        if not self.spectrum_data:
            raise RuntimeError("No spectrum data available for analysis")
            
        # Simulate compound identification and quantification
        compounds = {
            'Water': 0.65,
            'Ethanol': 0.20,
            'Glucose': 0.10,
            'Unknown': 0.05
        }
        
        return compounds

    def find_peaks(self, 
                  threshold: float = 0.1, 
                  min_distance: float = 5.0) -> List[SpectralPeak]:
        """
        Identify peaks in the spectrum above threshold.
        
        Args:
            threshold: Minimum intensity threshold relative to maximum
            min_distance: Minimum separation between peaks in cm^-1
            
        Returns:
            List of SpectralPeak objects
        """
        if not self.spectrum_data:
            raise RuntimeError("No spectrum data available for peak finding")
            
        peaks = []
        intensities = self.spectrum_data['intensity']
        max_intensity = np.max(intensities)
        
        # Simple peak finding algorithm
        for i in range(1, len(intensities)-1):
            if (intensities[i] > threshold * max_intensity and
                intensities[i] > intensities[i-1] and
                intensities[i] > intensities[i+1]):
                
                # Calculate peak parameters
                peak = SpectralPeak(
                    wavenumber=self.wavenumbers[i],
                    intensity=intensities[i],
                    fwhm=self._calculate_fwhm(i),
                    area=self._calculate_peak_area(i),
                    snr=self._calculate_snr(i)
                )
                peaks.append(peak)
        
        return peaks

    def set_laser_power(self, power_mw: float) -> None:
        """Set laser power in milliwatts."""
        if not 0 <= power_mw <= 500:
            raise ValueError("Power must be between 0 and 500 mW")
        self.power_mw = power_mw

    def _generate_simulated_spectrum(self) -> np.ndarray:
        """Generate a realistic simulated Raman spectrum."""
        # Create base spectrum with multiple Gaussian peaks
        spectrum = np.zeros_like(self.wavenumbers)
        
        # Add some typical Raman peaks
        peaks = [
            (520.7, 1.0, 5.0),    # Silicon
            (1332.5, 0.8, 3.0),   # Diamond
            (1001.4, 0.6, 4.0),   # Polystyrene
            (1602.3, 0.4, 6.0),   # Aromatic ring
        ]
        
        for center, amplitude, width in peaks:
            spectrum += amplitude * np.exp(
                -(self.wavenumbers - center)**2 / (2 * width**2)
            )
            
        return spectrum

    def _calculate_fwhm(self, peak_index: int) -> float:
        """Calculate Full Width at Half Maximum for a peak."""
        intensities = self.spectrum_data['intensity']
        half_max = intensities[peak_index] / 2
        
        # Find left and right indices where intensity crosses half max
        left_idx = right_idx = peak_index
        while left_idx > 0 and intensities[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(intensities)-1 and intensities[right_idx] > half_max:
            right_idx += 1
            
        return self.wavenumbers[right_idx] - self.wavenumbers[left_idx]

    def _calculate_peak_area(self, peak_index: int) -> float:
        """Calculate integrated area under a peak."""
        # Simple trapezoidal integration
        return np.trapz(
            self.spectrum_data['intensity'][peak_index-5:peak_index+6],
            self.wavenumbers[peak_index-5:peak_index+6]
        )

    def _calculate_snr(self, peak_index: int) -> float:
        """Calculate Signal-to-Noise Ratio for a peak."""
        signal = self.spectrum_data['intensity'][peak_index]
        noise = np.std(self.spectrum_data['intensity'][max(0, peak_index-20):peak_index-5])
        return signal / noise if noise != 0 else float('inf')

    def get_status(self) -> Dict[str, any]:
        """Get current status of the spectrometer."""
        return {
            'laser_type': self.laser_type,
            'power_mw': self.power_mw,
            'is_calibrated': self.is_calibrated,
            'temperature': self.temperature,
            'spectral_range': self.spectral_range,
            'resolution': self.resolution
        } 