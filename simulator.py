import numpy as np
from typing import List, Tuple, Dict, Any
from physics_generators import apogee_calculation_generator
from constants import *
from data_types import *

class WaypointTracker:
    """Tracks waypoints reached during rocket flight"""
    
    def __init__(self, waypoints: np.ndarray, capture_radius: float = 50.0):
        """
        Args:
            waypoints: Nx3 array of waypoint positions [[x1,y1,z1], [x2,y2,z2], ...]
            capture_radius: Distance (m) within which waypoint is considered reached
        """
        self.waypoints = waypoints
        self.capture_radius = capture_radius
        self.current_waypoint_idx = 0
        self.waypoint_results: List[WaypointResult] = []
        self.trajectory: List[TrajectoryPoint] = []
    
    def update(self, t: float, position: np.ndarray, velocity: np.ndarray) -> bool:
        """
        Update tracker with current state. Returns True if new waypoint reached.
        
        Args:
            t: Current time (s)
            position: Current position [x, y, z] (m)
            velocity: Current velocity [vx, vy, vz] (m/s)
        """
        # Store trajectory point
        altitude = position[2] if len(position) > 2 else position[0]
        self.trajectory.append(TrajectoryPoint(
            time=t,
            position=position.copy(),
            velocity=velocity.copy(),
            altitude=altitude
        ))
        
        # Check if all waypoints reached
        if self.current_waypoint_idx >= len(self.waypoints):
            return False
        
        # Check distance to current waypoint
        target = self.waypoints[self.current_waypoint_idx]
        distance = np.linalg.norm(position - target)
        
        if distance < self.capture_radius:
            # Waypoint reached!
            self.waypoint_results.append(WaypointResult(
                waypoint_index=self.current_waypoint_idx,
                reached=True,
                time_reached=t,
                position_reached=position.copy(),
                distance_error=distance,
                target_position=target.copy()
            ))
            self.current_waypoint_idx += 1
            return True
        
        return False
    
    def finalize(self) -> np.ndarray:
        """
        Finalize tracking and mark unreached waypoints.
        Returns: Boolean array indicating which waypoints were reached.
        """
        # Mark any remaining waypoints as not reached
        for idx in range(self.current_waypoint_idx, len(self.waypoints)):
            self.waypoint_results.append(WaypointResult(
                waypoint_index=idx,
                reached=False,
                target_position=self.waypoints[idx].copy()
            ))
        
        # Create boolean array
        reached_array = np.zeros(len(self.waypoints), dtype=bool)
        for result in self.waypoint_results:
            if result.reached:
                reached_array[result.waypoint_index] = True
        
        return reached_array
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        reached = sum(1 for r in self.waypoint_results if r.reached)
        total = len(self.waypoints)
        
        tracking_errors = [r.distance_error for r in self.waypoint_results if r.reached]
        avg_error = np.mean(tracking_errors) if tracking_errors else 0.0
        
        return {
            'waypoints_reached': reached,
            'total_waypoints': total,
            'completion_percentage': (reached / total * 100) if total > 0 else 0,
            'average_tracking_error': avg_error,
            'max_altitude': max(p.altitude for p in self.trajectory) if self.trajectory else 0,
            'trajectory_points': len(self.trajectory)
        }
    
    def print_summary(self):
        """Print waypoint mission summary"""
        summary = self.get_summary()
        
        print("\n=== WAYPOINT MISSION PERFORMANCE ===")
        print(f"Mission Completion: {summary['completion_percentage']:.1f}% "
              f"({summary['waypoints_reached']}/{summary['total_waypoints']} waypoints)")
        print(f"Average Tracking Error: {summary['average_tracking_error']:.2f} m")
        print(f"Max Altitude: {summary['max_altitude']:.2f} m")
        print(f"\nWaypoints:")
        
        for result in self.waypoint_results:
            status = "✓ REACHED" if result.reached else "✗ MISSED"
            print(f"  WP{result.waypoint_index + 1}: {status}", end="")
            
            if result.reached:
                print(f" at t={result.time_reached:.1f}s, error={result.distance_error:.1f}m")
            else:
                target = result.target_position
                print(f" (target: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}])")
    
    def get_trajectory_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get trajectory as arrays for plotting.
        Returns: (times, positions, velocities, altitudes)
        """
        if not self.trajectory:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        times = np.array([p.time for p in self.trajectory])
        positions = np.array([p.position for p in self.trajectory])
        velocities = np.array([p.velocity for p in self.trajectory])
        altitudes = np.array([p.altitude for p in self.trajectory])
        
        return times, positions, velocities, altitudes


def simulate_with_waypoint_tracking(
    apogee_gen,
    waypoints: np.ndarray,
    capture_radius: float = 50.0,
    verbose: bool = True
) -> Tuple[WaypointTracker, float]:
    """
    Run simulation with waypoint tracking.
    
    Args:
        apogee_gen: Generator from apogee_calculation_generator
        waypoints: Nx3 array of waypoint positions (or Nx1 for 1D vertical)
        capture_radius: Distance threshold for waypoint capture (m)
        verbose: Print progress updates
    
    Returns:
        (WaypointTracker, apogee_altitude)
    """
    tracker = WaypointTracker(waypoints, capture_radius)
    apogee_altitude = 0.0
    
    for state in apogee_gen:
        t = state['t']
        z = state['z']
        w = state['w']
        
        # For 1D vertical simulation, position is [0, 0, z]
        position = np.array([0.0, 0.0, z])
        velocity = np.array([0.0, 0.0, w])
        
        # Update tracker
        if tracker.update(t, position, velocity):
            if verbose:
                result = tracker.waypoint_results[-1]
                print(f"✓ Reached WP{result.waypoint_index + 1} at t={t:.1f}s, "
                      f"altitude={z:.1f}m, error={result.distance_error:.1f}m")
        
        # Check for apogee
        if state['apogee_reached']:
            apogee_altitude = state['apogee_altitude']
            if verbose:
                print(f"\nApogee reached: {apogee_altitude:.2f} m at t={t:.1f}s")
            break
    
    # Finalize tracking
    tracker.finalize()
    
    return tracker, apogee_altitude

def run_full_simulation(params: SimulationParams):
    states = []
    # print(f"params = {params}")
    # Phase 1: BURN (T > 0, mass decreasing)
    current_z = 0.0
    current_w = params.initial_speed
    current_mass = params.initial_mass
    burn_gen = apogee_calculation_generator(
        initial_z=current_z,
        initial_w=current_w,
        g=GRAVITY,
        m=current_mass,
        T=params.thrust,
        D=DRAG_MAGNITUDE,
        alpha= ANGLE_OF_ATTACK,
        dt=DEFAULT_TIME_INCREMENT,
        num_steps=int(params.burn_time / DEFAULT_TIME_INCREMENT)  # Limit to burn time
    )
    # print(f"burn gen = {burn_gen}")
    
    for state in burn_gen:
        # print(f"thrust state is {state}")
        # Update mass: m(t) = m0 - mass_flow_rate * t
        # current_mass = params.initial_mass - params.mass_flow_rate * state['t']
        states.append(state)
    
    # Phase 2: COAST (T = 0, constant mass)
    # print("coast phase")
    coast_gen = apogee_calculation_generator(
        initial_z=states[-1]['z'],
        initial_w=states[-1]['w'],
        g=GRAVITY,
        m=current_mass,  # Final mass after burn
        alpha= ANGLE_OF_ATTACK,
        T=0.0,  # No thrust
        D=DRAG_MAGNITUDE,
        dt=DEFAULT_TIME_INCREMENT,
    )
    
    for state in coast_gen:
        # print(f"coast state is {state}")
        states.append(state)
        if state['apogee_reached']:
            break
    
    return states

# Example usage
if __name__ == "__main__":
    # Example: Vertical waypoints at different altitudes
    waypoints = np.array([
        [0, 0, 100],   # WP1: 100m altitude
        [0, 0, 250],   # WP2: 250m altitude
        [0, 0, 400],   # WP3: 400m altitude
        [0, 0, 500],   # WP4: 500m altitude (apogee target)
    ])
    
    # apogee_gen = apogee_calculation_generator(...)
    
    # Then run with tracking:
    # tracker, apogee = simulate_with_waypoint_tracking(apogee_gen, waypoints)
    # tracker.print_summary()
    # reached_array = tracker.finalize()
    # print(f"\nReached waypoints: {reached_array}")