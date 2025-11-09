import numpy as np
from typing import Dict, Tuple
from data_types import *
import simulator

class ActiveStabilizer:
    """Active stabilization controller using PID control"""
    
    def __init__(self, config: StabilizationConfig):
        self.config = config
        
        # Integral error accumulators
        self.pitch_integral = 0.0
        self.yaw_integral = 0.0
        self.roll_integral = 0.0
        
        # Previous errors for derivative
        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0
        self.prev_roll_error = 0.0
        
        # Time tracking
        self.prev_time = 0.0
    
    def reset(self):
        """Reset controller state"""
        self.pitch_integral = 0.0
        self.yaw_integral = 0.0
        self.roll_integral = 0.0
        self.prev_pitch_error = 0.0
        self.prev_yaw_error = 0.0
        self.prev_roll_error = 0.0
        self.prev_time = 0.0
    
    def euler_from_quaternion(self, q: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        q = [qw, qx, qy, qz]
        Returns: (phi, theta, psi) in radians
        """
        qw, qx, qy, qz = q
        
        # Roll (phi)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        phi = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (theta)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            theta = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            theta = np.arcsin(sinp)
        
        # Yaw (psi)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        psi = np.arctan2(siny_cosp, cosy_cosp)
        
        return phi, theta, psi
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def compute_control(
        self, 
        q: np.ndarray, 
        omega: np.ndarray, 
        t: float
    ) -> Tuple[float, float, float]:
        """
        Compute control deflections based on current attitude and rates.
        
        Args:
            q: Current quaternion [qw, qx, qy, qz]
            omega: Current angular rates [p, q, r] (rad/s)
            t: Current time (s)
        
        Returns:
            (delta_pitch, delta_yaw, delta_roll) fin deflections in radians
        """
        # Get current Euler angles
        phi, theta, psi = self.euler_from_quaternion(q)
        
        # Time step
        dt = t - self.prev_time if self.prev_time > 0 else 0.01
        self.prev_time = t
        
        # Compute errors (normalized to [-pi, pi])
        pitch_error = self.normalize_angle(self.config.target_pitch - theta)
        yaw_error = self.normalize_angle(self.config.target_yaw - psi)
        roll_error = self.normalize_angle(self.config.target_roll - phi)
        
        # Update integrals with anti-windup (prevent integral buildup)
        # Only accumulate integral when error is small (prevents windup during large disturbances)
        max_integral = 0.1  # Reduced limit for integral contribution
        error_threshold = np.deg2rad(10)  # Only integrate when error < 10 degrees
        
        if abs(pitch_error) < error_threshold:
            self.pitch_integral += pitch_error * dt
        self.pitch_integral = np.clip(self.pitch_integral, -max_integral, max_integral)
        
        if abs(yaw_error) < error_threshold:
            self.yaw_integral += yaw_error * dt
        self.yaw_integral = np.clip(self.yaw_integral, -max_integral, max_integral)
        
        if abs(roll_error) < error_threshold:
            self.roll_integral += roll_error * dt
        self.roll_integral = np.clip(self.roll_integral, -max_integral, max_integral)
        
        # Compute derivatives with low-pass filter to reduce noise amplification
        derivative_filter = 0.1  # Lower = more filtering (0.1-0.3 typical)
        
        if dt > 0:
            pitch_derivative_raw = (pitch_error - self.prev_pitch_error) / dt
            yaw_derivative_raw = (yaw_error - self.prev_yaw_error) / dt
            roll_derivative_raw = (roll_error - self.prev_roll_error) / dt
            
            # Apply exponential filter to derivatives
            if not hasattr(self, 'filtered_pitch_deriv'):
                self.filtered_pitch_deriv = 0.0
                self.filtered_yaw_deriv = 0.0
                self.filtered_roll_deriv = 0.0
            
            self.filtered_pitch_deriv = derivative_filter * pitch_derivative_raw + (1 - derivative_filter) * self.filtered_pitch_deriv
            self.filtered_yaw_deriv = derivative_filter * yaw_derivative_raw + (1 - derivative_filter) * self.filtered_yaw_deriv
            self.filtered_roll_deriv = derivative_filter * roll_derivative_raw + (1 - derivative_filter) * self.filtered_roll_deriv
            
            pitch_derivative = self.filtered_pitch_deriv
            yaw_derivative = self.filtered_yaw_deriv
            roll_derivative = self.filtered_roll_deriv
        else:
            pitch_derivative = 0.0
            yaw_derivative = 0.0
            roll_derivative = 0.0
        
        # Store current errors
        self.prev_pitch_error = pitch_error
        self.prev_yaw_error = yaw_error
        self.prev_roll_error = roll_error
        
        # PID control law with adjustable rate damping
        p, q_rate, r = omega
        
        # Rate damping - try reducing this significantly
        rate_damping = 0.5  # Lower (0.1-0.5) so PID terms have more effect
        
        # Calculate individual PID components for debugging
        pitch_p = self.config.pitch_gains.Kp * pitch_error
        pitch_i = self.config.pitch_gains.Ki * self.pitch_integral
        pitch_d = self.config.pitch_gains.Kd * pitch_derivative
        pitch_rate = rate_damping * q_rate
        
        yaw_p = self.config.yaw_gains.Kp * yaw_error
        yaw_i = self.config.yaw_gains.Ki * self.yaw_integral
        yaw_d = self.config.yaw_gains.Kd * yaw_derivative
        yaw_rate = rate_damping * r
        
        roll_p = self.config.roll_gains.Kp * roll_error
        roll_i = self.config.roll_gains.Ki * self.roll_integral
        roll_d = self.config.roll_gains.Kd * roll_derivative
        roll_rate = rate_damping * p
        
        delta_pitch = pitch_p + pitch_i + pitch_d - pitch_rate
        delta_yaw = yaw_p + yaw_i + yaw_d - yaw_rate
        delta_roll = roll_p + roll_i + roll_d - roll_rate
        
        # Store for debugging (uncomment to see contributions)
        # if t % 1.0 < 0.01:
        #     print(f"Pitch: P={np.rad2deg(pitch_p):.2f}° I={np.rad2deg(pitch_i):.2f}° D={np.rad2deg(pitch_d):.2f}° Rate={np.rad2deg(pitch_rate):.2f}°")
        
        # Limit deflections
        delta_pitch = np.clip(delta_pitch, -self.config.max_deflection, self.config.max_deflection)
        delta_yaw = np.clip(delta_yaw, -self.config.max_deflection, self.config.max_deflection)
        delta_roll = np.clip(delta_roll, -self.config.max_deflection, self.config.max_deflection)
        
        return delta_pitch, delta_yaw, delta_roll
    
    def get_status(self, q: np.ndarray, omega: np.ndarray) -> Dict[str, float]:
        """Get current stabilization status"""
        phi, theta, psi = self.euler_from_quaternion(q)
        p, q_rate, r = omega
        
        return {
            'roll_deg': np.rad2deg(phi),
            'pitch_deg': np.rad2deg(theta),
            'yaw_deg': np.rad2deg(psi),
            'roll_rate_deg_s': np.rad2deg(p),
            'pitch_rate_deg_s': np.rad2deg(q_rate),
            'yaw_rate_deg_s': np.rad2deg(r),
            'pitch_error_deg': np.rad2deg(self.prev_pitch_error),
            'yaw_error_deg': np.rad2deg(self.prev_yaw_error),
            'roll_error_deg': np.rad2deg(self.prev_roll_error)
        }


if __name__ == "__main__":
    results = simulator.stabilized_flight_simulation(duration=10.0, disturbance_enabled=True)
    
    # Plot results if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Attitude plot
        ax1.plot(results['time'], results['pitch'], 'r-', label='Pitch', linewidth=2)
        ax1.plot(results['time'], results['yaw'], 'g-', label='Yaw', linewidth=2)
        ax1.plot(results['time'], results['roll'], 'b-', label='Roll', linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Angle [deg]')
        ax1.set_title('Rocket Attitude (Active Stabilization)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Control deflections
        ax2.plot(results['time'], results['delta_pitch'], 'r-', label='Pitch deflection', linewidth=2)
        ax2.plot(results['time'], results['delta_yaw'], 'g-', label='Yaw deflection', linewidth=2)
        ax2.plot(results['time'], results['delta_roll'], 'b-', label='Roll deflection', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Deflection [deg]')
        ax2.set_title('Control Surface Deflections')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
