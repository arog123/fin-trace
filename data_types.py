from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationParams:
    rocket_length: float
    initial_mass: float
    mass_flow_rate: float
    fin_chord: float
    fin_span: float
    thrust: float
    burn_time: float
    initial_speed: float

@dataclass
class WaypointResult:
    """Results from waypoint tracking"""
    waypoint_index: int
    reached: bool
    time_reached: float = None
    position_reached: np.ndarray = None
    distance_error: float = None
    target_position: np.ndarray = None

@dataclass
class TrajectoryPoint:
    """Single point along the rocket's path"""
    time: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    altitude: float

@dataclass
class PIDGains:
    """PID controller gains for each axis"""
    Kp: float  # Proportional gain
    Ki: float  # Integral gain
    Kd: float  # Derivative gain

@dataclass
class StabilizationConfig:
    """Configuration for active stabilization"""
    pitch_gains: PIDGains
    yaw_gains: PIDGains
    roll_gains: PIDGains
    max_deflection: float = np.deg2rad(15)  # Maximum fin deflection (rad)
    target_pitch: float = 0.0  # Target pitch angle (rad)
    target_yaw: float = 0.0    # Target yaw angle (rad)
    target_roll: float = 0.0   # Target roll angle (rad)
