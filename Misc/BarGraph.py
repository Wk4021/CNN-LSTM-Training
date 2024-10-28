import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import os
import ast
import threading

class CSVGraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV BPM and RR Graphs")

        self.file_path = None
        self.data = None
        self.last_modified_time = None
        self.points_count = 0
        self.last_bpm = None
        self.last_rr = None

        self.init_ui()

    def init_ui(self):
        # Button Frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X)

        self.force_update_button = tk.Button(button_frame, text="Force Update", command=self.update_graphs)
        self.force_update_button.pack(side=tk.LEFT)

        self.exit_button = tk.Button(button_frame, text="Exit", command=self.exit_application)
        self.exit_button.pack(side=tk.LEFT)

        self.stats_label = tk.Label(button_frame, text="Points Graphed: 0 | Last BPM: None | Last RR: None")
        self.stats_label.pack(side=tk.LEFT)

        # Tab manager
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.file_selection_frame = tk.Frame(self.notebook)
        self.bpm_distribution_frame = tk.Frame(self.notebook)
        self.rr_distribution_frame = tk.Frame(self.notebook)
        self.bpm_vs_rr_frame = tk.Frame(self.notebook)
        self.heatmap_frame = tk.Frame(self.notebook)
        self.all_graphs_frame = tk.Frame(self.notebook)

        self.notebook.add(self.file_selection_frame, text="File Selection")
        self.notebook.add(self.bpm_distribution_frame, text="BPM Distribution")
        self.notebook.add(self.rr_distribution_frame, text="RR Distribution")
        self.notebook.add(self.bpm_vs_rr_frame, text="BPM vs RR")
        self.notebook.add(self.heatmap_frame, text="3D Heatmap")
        self.notebook.add(self.all_graphs_frame, text="All Graphs")

        self.init_file_selection_tab()
        self.init_all_graphs_tab()

        self.check_for_updates()

    def init_file_selection_tab(self):
        select_button = tk.Button(self.file_selection_frame, text="Select CSV File", command=self.select_file)
        select_button.pack(pady=10)

        self.loaded_points_label = tk.Label(self.file_selection_frame, text="Points Loaded: 0 | Currently Loading: 0")
        self.loaded_points_label.pack(pady=10)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.file_path:
            self.last_modified_time = os.path.getmtime(self.file_path)
            threading.Thread(target=self.load_data_and_update_graphs).start()

    def load_data_and_update_graphs(self):
        loading_count = 0  # Update this logic based on your loading process
        self.loaded_points_label.config(text=f"Points Loaded: {self.points_count} | Currently Loading: {loading_count}")
        self.load_data()
        self.root.after(0, self.update_graphs)  # Ensure update happens in main thread

    def load_data(self):
        self.data = pd.read_csv(self.file_path, header=None, skiprows=1)
        self.data = np.array(self.data[1].apply(ast.literal_eval).tolist())
        self.points_count = self.data.shape[0]

    def update_graphs(self):
        if self.data is not None and self.data.size > 0:
            bpm = self.data[:, 0]
            rr = self.data[:, 1]
            self.last_bpm = bpm[-1]
            self.last_rr = rr[-1]

            self.stats_label.config(text=f"Points Graphed: {self.points_count} | Last BPM: {self.last_bpm:.2f} | Last RR: {self.last_rr:.2f}")

            # Update individual graph tabs
            self.create_bpm_distribution(self.bpm_distribution_frame)
            self.create_rr_distribution(self.rr_distribution_frame)
            self.create_bpm_vs_rr(self.bpm_vs_rr_frame)
            self.create_heatmap(self.heatmap_frame)

    def init_all_graphs_tab(self):
        self.graph_options = ["BPM Distribution", "RR Distribution", "BPM vs RR", "3D Heatmap"]
        self.selected_graphs = [tk.StringVar(value=self.graph_options[0]) for _ in range(4)]

        for i in range(4):
            dropdown = ttk.Combobox(self.all_graphs_frame, textvariable=self.selected_graphs[i], values=self.graph_options)
            dropdown.grid(row=0, column=i, padx=5, pady=5)
            dropdown.bind("<<ComboboxSelected>>", lambda e: self.show_selected_graphs())

        self.graph_frames = [tk.Frame(self.all_graphs_frame) for _ in range(4)]
        for i, frame in enumerate(self.graph_frames):
            frame.grid(row=1, column=i, padx=5, pady=5, sticky='nsew')

        # Configure grid weights for resizing
        self.all_graphs_frame.grid_rowconfigure(1, weight=1)
        for i in range(4):
            self.all_graphs_frame.grid_columnconfigure(i, weight=1)

        # Initial graph display
        self.show_selected_graphs()

    def show_selected_graphs(self):
        # Clear the current graphs
        for frame in self.graph_frames:
            for widget in frame.winfo_children():
                widget.destroy()

        # Create graphs based on selected dropdowns
        for i, selected_graph in enumerate(self.selected_graphs):
            if self.data is not None:
                if selected_graph.get() == "BPM Distribution":
                    self.create_bpm_distribution(self.graph_frames[i])
                elif selected_graph.get() == "RR Distribution":
                    self.create_rr_distribution(self.graph_frames[i])
                elif selected_graph.get() == "BPM vs RR":
                    self.create_bpm_vs_rr(self.graph_frames[i])
                elif selected_graph.get() == "3D Heatmap":
                    self.create_heatmap(self.graph_frames[i])

    def create_bpm_distribution(self, frame):
        if self.data is None: return
        fig_bpm = plt.Figure(figsize=(6, 4), dpi=150)
        ax = fig_bpm.add_subplot(111)
        unique_bpm, bpm_counts = np.unique(self.data[:, 0], return_counts=True)
        bars = ax.bar(unique_bpm, bpm_counts, color='blue')
        ax.set_title("BPM Distribution")
        ax.set_xlabel("BPM")
        ax.set_ylabel("Count")

        mplcursors.cursor(bars, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"BPM: {int(sel.target[0])}, Count: {int(sel.target[1])}"))

        canvas_bpm = FigureCanvasTkAgg(fig_bpm, master=frame)
        canvas_bpm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_rr_distribution(self, frame):
        if self.data is None: return
        fig_rr = plt.Figure(figsize=(6, 4), dpi=150)
        ax = fig_rr.add_subplot(111)
        unique_rr, rr_counts = np.unique(self.data[:, 1], return_counts=True)
        bars = ax.bar(unique_rr, rr_counts, color='green')
        ax.set_title("RR Distribution")
        ax.set_xlabel("RR")
        ax.set_ylabel("Count")

        mplcursors.cursor(bars, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"RR: {int(sel.target[0])}, Count: {int(sel.target[1])}"))

        canvas_rr = FigureCanvasTkAgg(fig_rr, master=frame)
        canvas_rr.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_bpm_vs_rr(self, frame):
        if self.data is None: return
        fig_bpm_vs_rr = plt.Figure(figsize=(6, 4), dpi=150)
        ax = fig_bpm_vs_rr.add_subplot(111)
        scatter = ax.scatter(self.data[:, 0], self.data[:, 1], color='purple', alpha=0.6)
        ax.set_title("BPM vs RR")
        ax.set_xlabel("BPM")
        ax.set_ylabel("RR")

        mplcursors.cursor(scatter, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"BPM: {int(sel.target[0])}, RR: {int(sel.target[1])}"))

        canvas_bpm_vs_rr = FigureCanvasTkAgg(fig_bpm_vs_rr, master=frame)
        canvas_bpm_vs_rr.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_heatmap(self, frame):
        if self.data is None:
            print("No data loaded.")
            return

        # Create a grid for the 3D plot
        x = self.data[:, 0]
        y = self.data[:, 1]
        histogram, xedges, yedges = np.histogram2d(x, y, bins=30)

        # Define the grid for the surface plot
        x_grid, y_grid = np.meshgrid(xedges[:-1], yedges[:-1])
        z_grid = histogram.T  # Transpose for correct orientation

        # Set any negative values to zero
        z_grid[z_grid < 0] = 0

        # Create a figure for the 3D heatmap
        fig_heatmap = plt.Figure(figsize=(6, 4), dpi=150)
        ax = fig_heatmap.add_subplot(111, projection='3d')

        # Create the surface plot
        surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='RdYlGn', edgecolor='none')

        # Set labels
        ax.set_xlabel('BPM')
        ax.set_ylabel('RR')
        ax.set_zlabel('Count')
        ax.set_title('3D Heatmap')

        # Add color bar at the top
        cbar = fig_heatmap.colorbar(surf, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Count')

        # Pack the canvas
        canvas_heatmap = FigureCanvasTkAgg(fig_heatmap, master=frame)
        canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def check_for_updates(self):
        if self.file_path:
            try:
                current_modified_time = os.path.getmtime(self.file_path)
                if current_modified_time != self.last_modified_time:
                    self.load_data_and_update_graphs()
                    self.last_modified_time = current_modified_time
            except FileNotFoundError:
                pass
        self.root.after(1000, self.check_for_updates)

    def exit_application(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVGraphApp(root)
    root.mainloop()
