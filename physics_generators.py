import numpy as np
from typing import Generator, Dict, Any, Tuple

def rotation_matrix_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Compute the rotation matrix from 3-2-1 Euler angles (roll, pitch, yaw).
    Convention: x forward, y right, z down.
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    R = np.array([
        [ctheta * cpsi, sphi * stheta * cpsi - cphi * spsi, cphi * stheta * cpsi + sphi * spsi],
        [ctheta * spsi, sphi * stheta * spsi + cphi * cpsi, cphi * stheta * spsi - sphi * cpsi],
        [-stheta, sphi * ctheta, cphi * ctheta]
    ])
    return R

def translational_dynamics_generator(
    initial_r: Tuple[float, float, float],
    initial_v: Tuple[float, float, float],
    p: float,
    q: float,
    r: float,
    phi: float,
    theta: float,
    psi: float,
    g: float,
    m: float,
    T: float,
    D: float,
    L: float,
    alpha: float,
    Y: float,
    Z: float,
    dt: float,
    num_steps: int = None
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Generator for simulating translational dynamics (position and velocity) using Euler integration.
    
    This implements the body-frame acceleration equations:
    
    \dot{u} = r v - q w - g \sin \theta + T/m - (D/m) \cos \alpha - (L/m) \sin \alpha
    \dot{v} = p w - r u + g \sin \phi \cos \theta - Y/m
    \dot{w} = q u - p v + g \cos \phi \cos \theta - Z/m
    
    And inertial position update:
    \dot{\mathbf{r}} = \mathbf{R}(\phi, \theta, \psi) \mathbf{v}
    
    Assumes fixed angular rates (p, q, r) and Euler angles (\phi, \theta, \psi) for this translational-only simulation.
    For a full 6DOF sim, these would be updated separately and passed anew each step.
    
    Yields a dict with 'r' (inertial position [x, y, z]) and 'v' (body velocity [u, v, w]) at each time step.
    
    Args:
        initial_r: Initial inertial position (x, y, z)
        initial_v: Initial body velocity (u, v, w)
        p, q, r: Angular rates (roll, pitch, yaw)
        phi, theta, psi: Euler angles (roll, pitch, yaw)
        g: Gravity acceleration (m/s²)
        m: Mass (kg)
        T: Thrust (N, along body x)
        D: Drag magnitude (N)
        L: Lift magnitude (N)
        alpha: Angle of attack (rad)
        Y: Side force (N, body y)
        Z: Normal force (N, body z, positive down)
        dt: Time increment (s)
        num_steps: Number of steps (default infinite; break externally if needed)
    
    Example usage:
        gen = translational_dynamics_generator([0,0,0], [0,0,0], 0,0,0, 0,0,0, 9.81, 1.0, 10.0, 0.1, 0.1, 0.1, 0,0, 0.1)
        for i, state in enumerate(gen):
            print(f"Step {i}: position={state['r']}, velocity={state['v']}")
            if i >= 4: break
    """
    current_r = np.array(initial_r, dtype=float)
    current_v = np.array(initial_v, dtype=float)
    step = 0
    while num_steps is None or step < num_steps:
        # Body-frame accelerations
        u, v_b, w = current_v  # v_b to avoid name conflict
        du = r * v_b - q * w - g * np.sin(theta) + T / m - (D / m) * np.cos(alpha) - (L / m) * np.sin(alpha)
        dv = p * w - r * u + g * np.sin(phi) * np.cos(theta) - Y / m
        dw = q * u - p * v_b + g * np.cos(phi) * np.cos(theta) - Z / m
        dot_v = np.array([du, dv, dw])
        v_new = current_v + dot_v * dt
        
        # Inertial position rate
        R = rotation_matrix_from_euler(phi, theta, psi)
        dot_r = R @ current_v
        r_new = current_r + dot_r * dt
        
        # Update
        current_r = r_new
        current_v = v_new
        
        yield {'r': r_new.copy(), 'v': v_new.copy()}
        step += 1

def quaternion_dot(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the time derivative of the quaternion using angular rates.
    
    \dot{q} = 1/2 * [ -omega × q ] in vector form, or explicit components.
    """
    qw, qx, qy, qz = q
    p, q_rate, r = omega  # q_rate to avoid name conflict with quaternion q
    dq_w = -0.5 * (qx * p + qy * q_rate + qz * r)
    dq_x =  0.5 * (qw * p - qz * q_rate + qy * r)
    dq_y =  0.5 * ( qz * p + qw * q_rate - qx * r)
    dq_z =  0.5 * (-qy * p + qx * q_rate + qw * r)
    return np.array([dq_w, dq_x, dq_y, dq_z])

def rotational_dynamics_generator(
    initial_omega: Tuple[float, float, float],
    initial_q: Tuple[float, float, float, float],
    Ixx: float,
    Iyy: float,
    Izz: float,
    L: float,
    M: float,
    N: float,
    dt: float,
    num_steps: int = None
) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Generator for simulating rotational dynamics (angular rates and attitude) using Euler integration.
    
    This implements the angular acceleration equations:
    
    \dot{p} = \frac{(I_{yy} - I_{zz}) q r}{I_{xx}} + \frac{L}{I_{xx}}
    \dot{q} = \frac{(I_{zz} - I_{xx}) p r}{I_{yy}} + \frac{M}{I_{yy}}
    \dot{r} = \frac{(I_{xx} - I_{yy}) p q}{I_{zz}} + \frac{N}{I_{zz}}
    
    And quaternion kinematics:
    \dot{\mathbf{q}} = \frac{1}{2} \Omega(\boldsymbol{\omega}) \mathbf{q}
    
    Assumes fixed moments (L, M, N) for this rotational-only simulation.
    For a full 6DOF sim, these would be updated separately and passed anew each step.
    Quaternion is normalized after each update to maintain unit length.
    
    Yields a dict with 'omega' (body angular rates [p, q, r]) and 'q' (quaternion [q_w, q_x, q_y, q_z]) at each time step.
    
    Args:
        initial_omega: Initial angular rates (p, q, r) in rad/s
        initial_q: Initial quaternion (q_w, q_x, q_y, q_z), assumed unit norm
        Ixx, Iyy, Izz: Principal moments of inertia (kg·m²)
        L, M, N: Aerodynamic moments (roll, pitch, yaw) in N·m
        dt: Time increment (s)
        num_steps: Number of steps (default infinite; break externally if needed)
    
    Example usage:
        gen = rotational_dynamics_generator([0,0,0], [1,0,0,0], 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.1)
        for i, state in enumerate(gen):
            print(f"Step {i}: omega={state['omega']}, q={state['q']}")
            if i >= 4: break
    """
    current_omega = np.array(initial_omega, dtype=float)
    current_q = np.array(initial_q, dtype=float)
    # Ensure initial quaternion is normalized
    current_q /= np.linalg.norm(current_q)
    step = 0
    while num_steps is None or step < num_steps:
        p, q_rate, r = current_omega  # q_rate to avoid name conflict
        
        # Angular accelerations
        dp = ((Iyy - Izz) * q_rate * r) / Ixx + L / Ixx
        dq = ((Izz - Ixx) * p * r) / Iyy + M / Iyy
        dr = ((Ixx - Iyy) * p * q_rate) / Izz + N / Izz
        dot_omega = np.array([dp, dq, dr])
        omega_new = current_omega + dot_omega * dt
        
        # Quaternion update
        dot_q = quaternion_dot(current_q, current_omega)
        q_new = current_q + dot_q * dt
        # Normalize
        q_new /= np.linalg.norm(q_new)
        
        # Update
        current_omega = omega_new
        current_q = q_new
        
        yield {'omega': omega_new.copy(), 'q': q_new.copy()}
        step += 1

def aero_forces_moments_generator(
    initial_v: Tuple[float, float, float],
    delta_p: float,
    delta_y: float,
    delta_r: float,
    rho: float,
    S: float,
    l_ref: float,
    C_D0: float,
    C_Lalpha: float,
    C_Ybeta: float,
    C_Zalpha: float,
    C_ldelta_r: float,
    C_malpha: float,
    C_mdelta_p: float,
    C_nbeta: float,
    C_ndelta_y: float,
    dt: float,
    num_steps: int = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator for computing aerodynamic forces and moments, including fin control effects, at each time step.
    
    This computes based on current body velocities [u, v, w]:
    - V = ||v||
    - alpha ≈ atan2(w, u)
    - beta ≈ atan2(v, u)
    
    Forces:
    - D = q * S * C_D0  (base drag; induced drag omitted for simplicity)
    - L = q * S * C_Lalpha * (alpha + delta_p)
    - Y = q * S * C_Ybeta * (beta + delta_y)
    - Z = q * S * C_Zalpha * (alpha + delta_p)
    
    Moments (roll L, pitch M, yaw N):
    - L_mom = q * S * l_ref * C_ldelta_r * delta_r
    - M = q * S * l_ref * (C_malpha * alpha + C_mdelta_p * delta_p)
    - N = q * S * l_ref * (C_nbeta * beta + C_ndelta_y * delta_y)
    
    Where q = 0.5 * rho * V**2.
    
    Assumes fixed initial_v and fin deflections (delta_p, delta_y, delta_r) for this aero-only computation.
    In a full 6DOF sim, pass updated v and deltas each step externally.
    Yields a dict with 'forces' (dict of D, L, Y, Z), 'moments' (dict of L, M, N), 'alpha', 'beta', 'V' at each step.
    Values are constant across steps since no state update here.
    
    Typical parameters for model rocket:
    - C_D0 ≈ 0.3–0.75
    - C_Lalpha ≈ 3–5 /rad
    - C_Ybeta ≈ 3–5 /rad
    - C_Zalpha ≈ -C_Lalpha (for z-down convention)
    - C_ldelta_r ≈ 0.1–0.5 /rad
    - C_malpha ≈ -5–10 /rad (negative for stability)
    - C_mdelta_p ≈ -1–2 /rad
    - C_nbeta ≈ C_malpha
    - C_ndelta_y ≈ C_mdelta_p
    
    Args:
        initial_v: Initial/current body velocity (u, v, w) in m/s
        delta_p, delta_y, delta_r: Fin deflection angles (pitch, yaw, roll) in rad
        rho: Air density (kg/m³)
        S: Reference area (m²)
        l_ref: Reference length for moments (m)
        C_D0: Base drag coefficient
        C_Lalpha: Lift slope (1/rad)
        C_Ybeta: Side force slope (1/rad)
        C_Zalpha: Normal force slope (1/rad)
        C_ldelta_r: Roll control derivative (1/rad)
        C_malpha: Pitch stability derivative (1/rad)
        C_mdelta_p: Pitch control derivative (1/rad)
        C_nbeta: Yaw stability derivative (1/rad)
        C_ndelta_y: Yaw control derivative (1/rad)
        dt: Time increment (s, unused here but for consistency)
        num_steps: Number of steps (default infinite)
    
    Example usage:
        gen = aero_forces_moments_generator([100, 0, 5], 0.1, 0.05, 0.02, 1.225, 0.01, 0.1, 0.5, 4.0, 4.0, -4.0, 0.2, -6.0, -1.5, -6.0, -1.5, 0.1)
        for i, aero in enumerate(gen):
            print(f"Step {i}: D={aero['forces']['D']:.2f}, M={aero['moments']['M']:.2f}")
            if i >= 4: break
    """
    current_v = np.array(initial_v, dtype=float)
    step = 0
    while num_steps is None or step < num_steps:
        u, v_side, w = current_v  # v_side to avoid name conflict
        V = np.linalg.norm(current_v)
        if V < 1e-6:  # Avoid division by zero
            yield {
                'forces': {'D': 0.0, 'L': 0.0, 'Y': 0.0, 'Z': 0.0},
                'moments': {'L': 0.0, 'M': 0.0, 'N': 0.0},
                'alpha': 0.0, 'beta': 0.0, 'V': 0.0
            }
        else:
            alpha = np.arctan2(w, u)
            beta = np.arctan2(v_side, u)  # Approximation for small angles
            q_dyn = 0.5 * rho * V**2
            
            # Forces
            CD = C_D0
            D = q_dyn * S * CD
            L = q_dyn * S * C_Lalpha * (alpha + delta_p)
            Y = q_dyn * S * C_Ybeta * (beta + delta_y)
            Z = q_dyn * S * C_Zalpha * (alpha + delta_p)
            
            # Moments
            L_mom = q_dyn * S * l_ref * C_ldelta_r * delta_r
            M = q_dyn * S * l_ref * (C_malpha * alpha + C_mdelta_p * delta_p)
            N = q_dyn * S * l_ref * (C_nbeta * beta + C_ndelta_y * delta_y)
            
            yield {
                'forces': {'D': D, 'L': L, 'Y': Y, 'Z': Z},
                'moments': {'L': L_mom, 'M': M, 'N': N},
                'alpha': alpha, 'beta': beta, 'V': V
            }
        step += 1


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
    gen_translation = translational_dynamics_generator([0,0,0], [0,0,0], 0,0,0, 0,0,0, 9.81, 1.0, 10.0, 0.1, 0.1, 0.1, 0,0, 0.1)
    gen_rotation = rotational_dynamics_generator([0,0,0], [1,0,0,0], 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.1)
    gen_forces = aero_forces_moments_generator([100, 0, 5], 0.1, 0.05, 0.02, 1.225, 0.01, 0.1, 0.5, 4.0, 4.0, -4.0, 0.2, -6.0, -1.5, -6.0, -1.5, 0.1)
    gen_apogee = apogee_calculation_generator(0.0, 50.0, 9.81, 0.5, 0.0, 2.0, 0.0, 0.0, 0.1)

    print("Generating translational changes...")
    for i, state in enumerate(gen_translation):
        print(f"Step {i}: position={state['r']}, velocity={state['v']}")
        if i >= 4: break

    print("Generating rotational changes...")
    for i, state in enumerate(gen_rotation):
        print(f"Step {i}: omega={state['omega']}, q={state['q']}")
        if i >= 4: break

    print("Generating aerodynamic forces...")
    for i, aero in enumerate(gen_forces):
        print(f"Step {i}: D={aero['forces']['D']:.2f}, M={aero['moments']['M']:.2f}")
        if i >= 4: break

    print("Generating apogee...")
    apogee_z = None
    for state in gen_apogee:
        print(f"t={state['t']:.2f}s, z={state['z']:.2f}m, w={state['w']:.2f}m/s")
        if state['apogee_reached']:
            apogee_z = state['apogee_altitude']
            break