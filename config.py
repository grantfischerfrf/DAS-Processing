from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=False)
class DASConfig():

    '''
    This class holds all necessary hardcoded variables and paths needed in the various scripts in this repository
    frozen=True makes the dataclass immutable, so that the values cannot be changed after initialization.
    '''

    # --- DAS CABLE PARAMETERS ---
    gauge_length: float = 1.6 #meters
    channel_spacing: float = 1.6 #meters
    original_sampling_frequency: float = 500.0 #Hz
    decimated_sampling_frequency: float = 2.0 #Hz
    N_channels: int = 1592 #number of channels along the DAS
    alongshore_channel_start: int = 501 #channel that the alongshore leg should approximately start at
    alongshore_channel_end: int = 1091 #channel that the alongshore leg should approximately end at

    # --- AWAC PARAMETERS ---
    awac600_channel_location: int = 402 #channel number of the AWAC 600m xFRF location along the DAS cable
    awac300_channel_location: int = 208 #channel number of the AWAC 300m xFRF location along the DAS cable
    burst_duration: int = 35 #minutes - duration of the data collection bursts for the AWACs

    # --- FRF SPECIFICS ---
    FRF_orientation: float = 71.8535 # degrees - orientation of the FRF beach to true north in wgs84

    # --- FILTERING ---
    temporal_window: int = 10 #minutes - size of the temporal window used in the temporal detrend in FRF_DAS_processing.py

    # --- PHYSICAL CONSTANTS ---
    g: int = 9.81 #m/s^2 - acceleration due to gravity
    rho: int = 1025 #kg/m^3 - density of seawater
    gamma: float = 0.43 #breaker index - used to scale the maximum height a wave can be before it breaks at a local water depth. gamma = Hb / h

    # --- SPECTRAL PARAMETERS ---
    wave_band_min: float = 0.05 # Hz - lower limit of the wave band
    wave_band_max: float = 0.3 # Hz - upper limit of the wave band
    nperseg: int = 512 # number of points in each segment for the Welch estimate
    noverlap: int = field(init=False) # number of points to overlap between segments for the Welch estimate
    directional_weight: int = 2 # exponent used in the directional weighting of the model spectra in Calibration.py

    # --- DIRECTORIES ---
    root: Path = Path('.') #'.' means current working directory
    data_directory: Path = field(init=False) #main directory where ALL data will be stored, subfolders are left purposely unstructured to allow for flexibility/personalization
    figure_directory: Path = field(init=False) #directory where ALL figures will be saved
    documentation_directory: Path = field(init=False) #directory where ALL documentation will be saved

    def __post_init__(self):
        self.root = Path(self.root) #support for string input or path input for root
        self.data_directory = self.root / 'data'
        self.figure_directory = self.root / 'figures'
        self.documentation_directory = self.root / 'documentation'

        # Calculate spectral overlap
        self.noverlap = self.nperseg // 2





