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