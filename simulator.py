import numpy as np
from typing import List, Tuple, Dict, Any
from physics_generators import apogee_calculation_generator
from constants import *
from data_types import *
from pid_controller import ActiveStabilizer

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

def stabilized_flight_simulation(
    duration: float = 10.0,
    dt: float = 0.01,
    disturbance_enabled: bool = True
):
    """
    Example of a stabilized flight simulation.
    
    This integrates rotational dynamics with active stabilization.
    """
    # Create stabilization configuration
    # Moderate gains with lower rate damping so PID has effect
    config = StabilizationConfig(
        pitch_gains=PIDGains(Kp=3.001, Ki=3.001, Kd=3.002),  # Moderate gains
        yaw_gains=PIDGains(Kp=3.001, Ki=3.001, Kd=3.002),
        roll_gains=PIDGains(Kp=1.00001, Ki=1.00001, Kd=1.0001),
        max_deflection=np.deg2rad(2),  # Max: 5 degrees
        target_pitch=0.0,  # Level flight
        target_yaw=0.0,
        target_roll=0.0
    )
    
    stabilizer = ActiveStabilizer(config)
    
    # Initial conditions with disturbance
    initial_omega = [0.1, 0.05, 0.08] if disturbance_enabled else [0.0, 0.0, 0.0]
    initial_q = [0.98, 0.05, 0.05, 0.02]  # Slightly off vertical
    initial_q = initial_q / np.linalg.norm(initial_q)
    
    # Rocket properties
    Ixx, Iyy, Izz = 0.01, 0.01, 0.005  # kg.m^2
    
    # Simulation
    print("Starting stabilized flight simulation...")
    print(f"Target attitude: pitch=0°, yaw=0°, roll=0°")
    print(f"Initial disturbance: {np.rad2deg(initial_omega)} deg/s\n")
    print("DEBUGGING: Watch for moments (M) and angular accelerations (dq)")
    print("If M is tiny compared to attitude error, stability derivatives are too weak!\n")
    
    current_omega = np.array(initial_omega)
    current_q = np.array(initial_q)
    t = 0.0
    
    results = {
        'time': [],
        'roll': [],
        'pitch': [],
        'yaw': [],
        'delta_pitch': [],
        'delta_yaw': [],
        'delta_roll': []
    }
    
    num_steps = int(duration / dt)
    
    for step in range(num_steps):
        # Get control commands
        delta_pitch, delta_yaw, delta_roll = stabilizer.compute_control(current_q, current_omega, t)
        
        # Compute moments from control deflections (simplified)
        # These would come from aero_forces_moments_generator in full sim
        rho = 1.225  # kg/m^3
        V = 50.0  # m/s (assumed constant for this example)
        S = 0.01  # m^2
        l_ref = 0.3  # m
        q_dyn = 0.5 * rho * V**2
        
        # Control moments
        L_mom = q_dyn * S * l_ref * 0.2 * delta_roll
        M = q_dyn * S * l_ref * (-1.5) * delta_pitch
        N = q_dyn * S * l_ref * (-1.5) * delta_yaw
        
        # Without natural stability, the rocket will never converge
        p, q_rate, r = current_omega
        
        # Calculate angle of attack and sideslip for stability
        u = V  # Assume mostly forward velocity
        w_aero = 5.0 * np.sin(stabilizer.euler_from_quaternion(current_q)[1])  # From pitch
        v_aero = 5.0 * np.sin(stabilizer.euler_from_quaternion(current_q)[2])  # From yaw
        
        alpha = np.arctan2(w_aero, u) if abs(u) > 1e-6 else 0.0
        beta = np.arctan2(v_aero, u) if abs(u) > 1e-6 else 0.0
        
        # Add natural aerodynamic stability (VERY IMPORTANT!)
        # Without this, control alone cannot stabilize
        C_malpha = -40.0  # Pitch stability (must be negative and strong)
        C_nbeta = -40.0   # Yaw stability (must be negative and strong)
        C_lp = -15.5      # Roll damping
        C_mq = -11.0      # Pitch damping
        C_nr = -11.0      # Yaw damping
        
        M += q_dyn * S * l_ref * (C_malpha * alpha + C_mq * q_rate * l_ref / V)
        N += q_dyn * S * l_ref * (C_nbeta * beta + C_nr * r * l_ref / V)
        L_mom += q_dyn * S * l_ref * (C_lp * p * l_ref / V)
        
        # Angular accelerations (Euler's equations)
        dp = ((Iyy - Izz) * q_rate * r) / Ixx + L_mom / Ixx
        dq = ((Izz - Ixx) * p * r) / Iyy + M / Iyy
        dr = ((Ixx - Iyy) * p * q_rate) / Izz + N / Izz
        
        # Update omega
        current_omega += np.array([dp, dq, dr]) * dt
        
        # Update quaternion
        qw, qx, qy, qz = current_q
        dq_w = -0.5 * (qx * p + qy * q_rate + qz * r)
        dq_x = 0.5 * (qw * p - qz * q_rate + qy * r)
        dq_y = 0.5 * (qz * p + qw * q_rate - qx * r)
        dq_z = 0.5 * (-qy * p + qx * q_rate + qw * r)
        current_q += np.array([dq_w, dq_x, dq_y, dq_z]) * dt
        current_q /= np.linalg.norm(current_q)
        
        # Store results
        phi, theta, psi = stabilizer.euler_from_quaternion(current_q)
        results['time'].append(t)
        results['roll'].append(np.rad2deg(phi))
        results['pitch'].append(np.rad2deg(theta))
        results['yaw'].append(np.rad2deg(psi))
        results['delta_pitch'].append(np.rad2deg(delta_pitch))
        results['delta_yaw'].append(np.rad2deg(delta_yaw))
        results['delta_roll'].append(np.rad2deg(delta_roll))
        
        # Print status every second with detailed diagnostics
        if step % int(1.0 / dt) == 0:
            status = stabilizer.get_status(current_q, current_omega)
            print(f"\n=== t={t:.1f}s ===")
            print(f"Attitude: Pitch={status['pitch_deg']:6.2f}° Yaw={status['yaw_deg']:6.2f}° Roll={status['roll_deg']:6.2f}°")
            print(f"Rates:    p={status['roll_rate_deg_s']:6.2f}°/s q={status['pitch_rate_deg_s']:6.2f}°/s r={status['yaw_rate_deg_s']:6.2f}°/s")
            print(f"Control:  δp={np.rad2deg(delta_pitch):5.2f}° δy={np.rad2deg(delta_yaw):5.2f}° δr={np.rad2deg(delta_roll):5.2f}°")
            print(f"Moments:  L={L_mom:.4f} M={M:.4f} N={N:.4f}")
            print(f"Accel:    dp={dp:.4f} dq={dq:.4f} dr={dr:.4f}")
        
        t += dt
    
    print("\nSimulation complete!")
    return results

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