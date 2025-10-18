import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import numpy as np
from waypoints import generate_waypoints

# GUI for input parameters
class RocketSimulatorGUI:
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("Rocket Simulator")
        self.create_widgets()
    
    def create_widgets(self) -> None:
        # Input fields for parameters
        params = [
            ("Rocket Length (m)", "3.0"),
            ("Initial Mass (kg)", "1.0"),
            ("Mass Flow Rate (kg/s)", "0.05"),
            ("Fin Chord (m)", "0.1"),
            ("Fin Span (m)", "0.15"),
            ("Thrust (N)", "150.0"),
            ("Burn Time (s)", "8.0"),
            ("Initial Speed (m/s)", "5.0")
        ]
        
        self.entries = {}
        for i, (label, default) in enumerate(params):
            tk.Label(self.root, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(self.root)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[label] = entry
        
        # Run button
        ttk.Button(self.root, text="Run Simulation", command=self.run_simulation).grid(row=len(params), column=0, columnspan=2, pady=10)

    def run_simulation(self) -> None:
        generate_plots()

def generate_attitude_plot() -> None:
    """Generate plot for attitude angles (roll, pitch, yaw)."""
    print("\nDisplaying Plot 1: Attitude Angles...")
    fig1, axs = plt.subplots(3, 1, figsize=(12, 10), num=1)
    fig1.suptitle('Rocket Attitude vs Time', fontsize=14)
    axs[0].set_ylabel('Roll Angle [°]')
    axs[0].set_ylim(-359, 359)
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set_ylabel('Pitch Angle [°]')
    axs[1].set_ylim(-45, 45)
    axs[1].grid(True)
    axs[1].legend()
    axs[2].set_ylabel('Yaw Angle [°]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylim(-45, 45)
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()


def generate_trajectory_plot() -> None:
    """Generate 3D trajectory plot."""
    
    WAYPOINTS = np.array(list(generate_waypoints(waypoint_type='vertical', max_height=800.0, n_points=8)))
    current_waypoint_index = 0
    waypoint_capture_radius = 10.0  # Distance to consider waypoint "reached" [m]

    print("=== WAYPOINT MISSION ===")
    print(f"Generated {len(WAYPOINTS)} waypoints in vertical pattern")
    for i, wp in enumerate(WAYPOINTS):
        print(f"WP{i+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}) m")

    print("Displaying Plot 2: 3D Trajectory...")
    fig2 = plt.figure(figsize=(14, 12), num=2)
    ax3d = fig2.add_subplot(111, projection='3d')

    waypoints_reached = [j[1] for j in WAYPOINTS]
    for i, wp in enumerate(WAYPOINTS):
        color, marker = ('red', 'X')
        ax3d.scatter(wp[0], wp[1], wp[2], color=color, s=400, marker=marker, alpha=0.9, 
                        edgecolors='darkgreen' if i in waypoints_reached else 'darkred', linewidth=3,
                        label='Reached WP' if i in waypoints_reached and len([idx for idx in waypoints_reached if idx == i]) == 1 else 'Target WP' if i == waypoints_reached[-1] + 1 else "")
        ax3d.text(wp[0] + 15, wp[1] + 15, wp[2] + 25, f'WP{i+1}\n({wp[2]:.0f}m)', 
                    fontsize=11, ha='center', weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax3d.plot([0, 0], [0, 0], [0, np.max(WAYPOINTS[:, 2])], 'k--', alpha=0.7, linewidth=3, label='Planned Vertical Path')

    try:
        x_range = 50
        z_max = np.max(WAYPOINTS[:, 2]) * 1.1
        ax3d.set_xlim(-x_range, x_range)
        ax3d.set_ylim(-x_range, x_range)
        ax3d.set_zlim(0, z_max)
        print(f"Plot ranges: X=[{-x_range:.0f}, {x_range:.0f}], Y=[{-x_range:.0f}, {x_range:.0f}], Z=[0, {z_max:.0f}]")
    except Exception as e:
        print(f"Error setting axis limits: {e}")
        ax3d.set_xlim(-100, 100)
        ax3d.set_ylim(-100, 100)
        ax3d.set_zlim(0, 1000)
    ax3d.set_xlabel('X [m]', fontsize=12, weight='bold')
    ax3d.set_ylabel('Y [m]', fontsize=12, weight='bold')
    ax3d.set_zlabel('Z (Altitude) [m]', fontsize=12, weight='bold')
    ax3d.set_title('Rocket Vertical Waypoint Mission', fontsize=16, fontweight='bold', pad=20)
    ax3d.legend(loc='upper left', fontsize=11)
    ax3d.grid(True, alpha=0.4)
    ax3d.view_init(elev=20, azim=45)


def generate_velocity_altitude_plot() -> None:
    """Generate velocity and altitude plots."""
    print("Displaying Plot 3: Velocity and Altitude...")
    fig3, (ax_vel, ax_alt) = plt.subplots(2, 1, figsize=(12, 8), num=3)
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.set_title('Velocity Components vs Time')
    ax_vel.set_ylim(0, 15)
    ax_vel.set_xlim(0, 5)
    ax_alt.set_ylabel('Altitude [m]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.grid(True)
    ax_alt.set_title('Altitude vs Time')
    ax_alt.set_xlim(0, 5)
    ax_alt.set_ylim(0, 20)
    plt.tight_layout()

def display_all_plots() -> None:
    """Display all generated plots and handle interaction."""
    plt.ion()
    
    # Draw all existing figures
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.draw()
        plt.pause(0.1)
    
    print("\nAll plots created! You can now interact with all three plot windows.")
    print("Close each plot window individually when you're done viewing them.")
    
    plt.ioff()
    plt.show()

def generate_plots() -> None:
    generate_attitude_plot()
    generate_trajectory_plot()
    generate_velocity_altitude_plot()
    display_all_plots()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RocketSimulatorGUI(root)
    root.mainloop()