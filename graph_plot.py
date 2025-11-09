from data_types import *
import matplotlib.pyplot as plt
import simulator
from waypoints import generate_waypoints

def generate_plots(params: SimulationParams) -> None:
    generate_attitude_plot()
    generate_trajectory_plot(params)
    generate_velocity_altitude_plot(params)
    display_all_plots()

def generate_attitude_plot() -> None:
    """Generate plot for attitude angles (roll, pitch, yaw)."""
    print("\nDisplaying Plot 1: Attitude Angles...")

    results = simulator.stabilized_flight_simulation(duration=10.0, disturbance_enabled=True)

    fig1, axs = plt.subplots(3, 1, figsize=(12, 10), num=1)
    fig1.suptitle('Rocket Attitude vs Time', fontsize=14)
    axs[0].set_ylabel('Roll Angle [°]')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].plot(results['time'], results['roll'], 'b-', label='Roll', linewidth=2)
    axs[1].set_ylabel('Pitch Angle [°]')
    axs[1].set_ylim(-10, 10)
    axs[1].grid(True)
    axs[1].legend()
    axs[1].plot(results['time'], results['pitch'], 'g-', label='Yaw', linewidth=2)
    axs[2].set_ylabel('Yaw Angle [°]')
    axs[2].set_xlabel('Time [s]')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].plot(results['time'], results['yaw'], 'g-', label='Yaw', linewidth=2)
    plt.tight_layout()

def generate_trajectory_plot(params: SimulationParams) -> None:
    """Generate 3D trajectory plot."""
    
    gen_apogee = simulator.run_full_simulation(params=params)
    apogee_z = None
    for state in gen_apogee:
        print(f"t={state['t']:.2f}s, z={state['z']:.2f}m, w={state['w']:.2f}m/s")
        if state['apogee_reached']:
            apogee_z = state['apogee_altitude']
            break

    # Generate waypoints and print them out
    WAYPOINTS = np.array(list(generate_waypoints(waypoint_type='vertical', max_height=800.0, n_points=8)))
    current_waypoint_index = 0
    waypoint_capture_radius = 10.0  # Distance to consider waypoint "reached" [m]

    print("=== WAYPOINT MISSION ===")
    print(f"Generated {len(WAYPOINTS)} waypoints in vertical pattern")
    for i, wp in enumerate(WAYPOINTS):
        print(f"WP{i+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}) m")

    tracker, apogee = simulator.simulate_with_waypoint_tracking(gen_apogee, WAYPOINTS)
    tracker.print_summary()
    reached_array = tracker.finalize()
    print(f"reached array = {reached_array}")

    # Get trajectory data for plotting
    times, positions, velocities, altitudes = tracker.get_trajectory_arrays()

    print("Displaying Plot 2: 3D Trajectory...")
    fig2 = plt.figure(figsize=(14, 12), num=2)
    ax3d = fig2.add_subplot(111, projection='3d')

    # Plot trajectory
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=4, alpha=0.8, label='Trajectory')

    # Add waypoints to plot
    waypoints_reached = [j[1] for j in WAYPOINTS]
    for i, wp in enumerate(WAYPOINTS):
        color, marker = ('lime', 'o') if reached_array[i] else ('red', 'X')
        ax3d.scatter(wp[0], wp[1], wp[2], color=color, s=400, marker=marker, alpha=0.9, 
                        edgecolors='darkgreen' if i in waypoints_reached else 'darkred', linewidth=3,
                        label='Reached WP' if i in waypoints_reached and len([idx for idx in waypoints_reached if idx == i]) == 1 else 'Target WP' if i == waypoints_reached[-1] + 1 else "")
        ax3d.text(wp[0] + 15, wp[1] + 15, wp[2] + 25, f'WP{i+1}\n({wp[2]:.0f}m)', 
                    fontsize=11, ha='center', weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax3d.plot([0, 0], [0, 0], [0, np.max(WAYPOINTS[:, 2])], 'k--', alpha=0.7, linewidth=3, label='Planned Vertical Path')

    # Set axis limits and labels
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

def generate_velocity_altitude_plot(params: SimulationParams) -> None:
    """Generate velocity and altitude plots."""
    print("Displaying Plot 3: Velocity and Altitude...")
    gen_apogee = simulator.run_full_simulation(params=params)
    apogee_z = None
    
    # Collect data from generator
    times = []
    altitudes = []
    velocities = []
    
    for state in gen_apogee:
        # Deal with time restarting during transition from burn phase to coast phase
        # TODO: Correct this issue in apogee_generator to have time consistency
        if len(times) != 0:
            if state['t'] <= times[-1]:
                times.append(round(state['t'], 2) + times[-1])
            else:
                times.append(round(state['t'], 2))
        else:
            times.append(round(state['t'], 2))
        altitudes.append(round(state['z'], 2))
        velocities.append(round(state['w'], 2))
        print(f"t={state['t']:.2f}s, z={state['z']:.2f}m, w={state['w']:.2f}m/s")
        if state['apogee_reached']:
            apogee_z = state['apogee_altitude']
            break  # Stop here since apogee is the end goal
    
    fig3, (ax_vel, ax_alt) = plt.subplots(2, 1, figsize=(12, 8), num=3)
    
    # Plot velocity
    ax_vel.plot(times, velocities, 'b-', linewidth=2, label='Vertical Velocity')
    ax_vel.axvline(x=params.burn_time, color='r', linestyle='--', alpha=0.5, label='Burn end')
    ax_vel.set_ylabel('Velocity [m/s]')
    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.set_title('Velocity Components vs Time')
    ax_vel.set_ylim(min(velocities) * 1.1 if velocities else -5, max(velocities) * 1.1 if velocities else 15)
    ax_vel.set_xlim(0, max(times) * 1.05 if times else 5)
    
    # Plot altitude
    ax_alt.plot(times, altitudes, 'g-', linewidth=2, label='Altitude')
    ax_alt.axvline(x=params.burn_time, color='r', linestyle='--', alpha=0.5, label='Burn end')
    if apogee_z is not None:
        apogee_idx = altitudes.index(max(altitudes))
        ax_alt.scatter(times[apogee_idx], apogee_z, 
                      color='red', s=200, marker='*', label=f'Apogee: {apogee_z:.1f}m')
    ax_alt.set_ylabel('Altitude [m]')
    ax_alt.set_xlabel('Time [s]')
    ax_alt.legend()
    ax_alt.grid(True)
    ax_alt.set_title('Altitude vs Time')
    ax_alt.set_xlim(0, max(times) * 1.05 if times else 5)
    ax_alt.set_ylim(0, max(altitudes) * 1.1 if altitudes else 20)
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