import tkinter as tk
from tkinter import ttk
from constants import *
from data_types import *
import graph_plot

# GUI for input parameters
class RocketSimulatorGUI:
    
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("Rocket Simulator")
        self.create_widgets()

    def get_params(self) -> SimulationParams:
        return SimulationParams(
            rocket_length=float(self.entries["Rocket Length (m)"].get()),
            initial_mass=float(self.entries["Initial Mass (kg)"].get()),
            mass_flow_rate=float(self.entries["Mass Flow Rate (kg/s)"].get()),
            fin_chord=float(self.entries["Fin Chord (m)"].get()),
            fin_span=float(self.entries["Fin Span (m)"].get()),
            thrust=float(self.entries["Thrust (N)"].get()),
            burn_time=float(self.entries["Burn Time (s)"].get()),
            initial_speed=float(self.entries["Initial Speed (m/s)"].get())
        )
    
    def create_widgets(self) -> None:
        # Input fields for parameters
        params = [
            ("Rocket Length (m)", "3.0"),
            ("Initial Mass (kg)", "1.0"),
            ("Mass Flow Rate (kg/s)", "0.05"),
            ("Fin Chord (m)", "0.1"),
            ("Fin Span (m)", "0.15"),
            ("Thrust (N)", THRUST),
            ("Burn Time (s)", "8.0"),
            ("Initial Speed (m/s)", INITIAL_W)
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
        params = self.get_params()
        graph_plot.generate_plots(params)

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RocketSimulatorGUI(root)
    root.mainloop()