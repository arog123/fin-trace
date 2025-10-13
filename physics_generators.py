import numpy as np
from typing import Generator, Dict, Any

def apogee_calculation_generator(
    initial_z: float,
    initial_w: float,
    g: float = 9.81,
    m: float = 1.0,
    T: float = 0.0,  # Thrust (N), 0 for coast phase
    D: float = 0.0,  # Drag (N), can be updated externally if needed
    alpha: float = 0.0,  # Angle of attack (rad), affects lift/drag projection
    L: float = 0.0,  # Lift (N)
    dt: float = 0.01,
    num_steps: int = None,
    tolerance: float = 1e-3  # For detecting apogee (velocity sign change)
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator for simulating vertical motion to calculate apogee (max altitude).
    
    This is a simplified 1D vertical dynamics model (z-down convention, w positive down? Wait, adjust:
    Actually, for rockets, z often up, w positive up. Here: assume z up, w positive up.
    Adapted from translational eq for vertical (w dot = -g + (T - D cos alpha - L sin alpha)/m, but simplified.
    For coast: T=0, if alpha=0, L=0, then dot_w = -g - D/m (if D opposes motion).
    
    Full integration until w <= 0 (apogee when w crosses from + to -).
    Yields dict with 'z' (altitude), 'w' (vertical velocity), 't' (time), 'apogee_reached' (bool),
    'apogee_altitude' (z at apogee, updated when reached).
    
    For burn phase, set T >0; for coast, T=0.
    Drag D and lift L can be fixed or passed anew in full sim.
    
    Quick estimate on first yield if desired, but here focuses on simulation.
    
    Args:
        initial_z: Initial altitude (m, positive up)
        initial_w: Initial vertical velocity (m/s, positive up)
        g: Gravity (m/s²)
        m: Mass (kg)
        T: Thrust (N, along body; projected to vertical if theta=0)
        D: Drag magnitude (N)
        alpha: Angle of attack (rad)
        L: Lift magnitude (N)
        dt: Time increment (s)
        num_steps: Max steps (default infinite)
        tolerance: Velocity threshold for apogee detection
    
    Example:
        gen = apogee_calculation_generator(0.0, 50.0, 9.81, 0.5, 0.0, 2.0, 0.0, 0.0, 0.1)
        apogee_z = None
        for state in gen:
            print(f"t={state['t']:.2f}s, z={state['z']:.2f}m, w={state['w']:.2f}m/s")
            if state['apogee_reached']:
                apogee_z = state['apogee_altitude']
                break
    """
    current_z = initial_z
    current_w = initial_w
    t = 0.0
    apogee_reached = False
    apogee_altitude = None
    prev_w = current_w
    step = 0
    
    while num_steps is None or step < num_steps:
        # Vertical acceleration (simplified, assuming aligned flight theta=0, no coriolis)
        # For full: include q u - p v terms, but omitted for 1D
        dw = -g + T / m - (D / m) * np.cos(alpha) - (L / m) * np.sin(alpha)  # w positive up
        # If w <0 and prev_w >0, apogee (but since up, apogee when w=0 from positive)
        w_new = current_w + dw * dt
        z_new = current_z + current_w * dt + 0.5 * dw * dt**2  # Trapezoidal for better accuracy
        
        t_new = t + dt
        
        # Check for apogee: w changes from positive to negative
        if not apogee_reached and prev_w > 0 and w_new < 0:
            # Interpolate apogee time/altitude
            t_apogee = t + (0 - current_w) / dw * dt if abs(dw) > tolerance else t
            z_apogee = current_z + current_w * (t_apogee - t) + 0.5 * dw * (t_apogee - t)**2
            apogee_reached = True
            apogee_altitude = z_apogee
        
        yield {
            'z': z_new,
            'w': w_new,
            't': t_new,
            'apogee_reached': apogee_reached,
            'apogee_altitude': apogee_altitude
        }
        
        # Update
        current_z = z_new
        current_w = w_new
        prev_w = w_new
        t = t_new
        step += 1
        
        # Optional early stop if apogee reached and w < -tolerance
        if apogee_reached and w_new < -tolerance:
            break

if __name__ == "__main__":
    gen_apogee = apogee_calculation_generator(0.0, 50.0, 9.81, 0.5, 0.0, 2.0, 0.0, 0.0, 0.1)

    print("Generating apogee...")
    apogee_z = None
    for state in gen_apogee:
        print(f"t={state['t']:.2f}s, z={state['z']:.2f}m, w={state['w']:.2f}m/s")
        if state['apogee_reached']:
            apogee_z = state['apogee_altitude']
            break