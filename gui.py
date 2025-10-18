import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk



# GUI for input parameters
class RocketSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rocket Simulator")
        self.create_widgets()
    
    def create_widgets(self):
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

    def run_simulation(self):
        generate_plots()

# TODO: Implement waypoint generation (might be in another file)

def generate_plots():

    # Plotting
    plt.ion()
    print("\nDisplaying Plot 1: Attitude Angles...")
    fig1, axs = plt.subplots(3, 1, figsize=(12, 10), num=1)
    fig1.suptitle('Rocket Attitude vs Time', fontsize=14)
    axs[0].set_ylabel('Roll Angle [°]')
    axs[0].set_ylim(-359,359)
    axs[0].grid(True)
    axs[0].legend()
    axs[1].set_ylabel('Pitch Angle [°]')
    axs[1].set_ylim(-45,45)
    axs[1].grid(True)
    axs[1].legend()
    axs[2].set_ylabel('Yaw Angle [°]')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylim(-45,45)
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    # TODO: Get waypoints and plot them on 3D Trajectory Graph

    print("Displaying Plot 2: 3D Trajectory...")
    fig2 = plt.figure(figsize=(14, 12), num=2)
    ax3d = fig2.add_subplot(111, projection='3d')

    try:
        x_range = 50
        z_max = 1000 # TODO: make change to this value possible or set to just bigger than highest waypoint
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
    plt.draw()
    plt.pause(0.1)

    print("Displaying Plot 3: Velocity and Altitude...")
    fig3, (ax_vel, ax_alt) = plt.subplots(2, 1, figsize=(12, 8), num=3)
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.set_title('Velocity Components vs Time')
    ax_vel.set_ylim(0,15)
    ax_vel.set_xlim(0,5)
    ax_alt.set_ylabel('Altitude [m]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.grid(True)
    ax_alt.set_title('Altitude vs Time')
    ax_alt.set_xlim(0,5)
    ax_alt.set_ylim(0,20)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

    print("\nAll plots created! You can now interact with all three plot windows.")
    print("Close each plot window individually when you're done viewing them.")
    plt.ioff()
    plt.show()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RocketSimulatorGUI(root)
    root.mainloop()