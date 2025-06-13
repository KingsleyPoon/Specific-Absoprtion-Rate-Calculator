'''
Version 2
Change log:
    - Updated how the temperature and time data is imported (using pandas - column headers)
    - fixed bug with selecting temperature

'''

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import csv
import scipy.optimize
import math
import os
import pandas as pd

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR Calculator")
        
        #Variable needed to calculate SAR
        self.heatCapacityWater = 4.184
        
        # Set window geometry size
        root.geometry("1100x600")
        root.resizable(False, False)
        
        # Open File Button
        self.open_button = tk.Button(root, text="Open File", command=self.open_file, font=("Helvetica", 10))
        self.open_button.place(relx=0.84, rely=0.03, anchor='nw', relwidth=0.15, relheight=0.07)

        # Canvas for matplotlib (graph)
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.place(relx=0.02, rely=0.03, anchor='nw', relwidth=0.8, relheight=0.92)
        
        # Labels to show which file is open
        self.file_label = tk.Label(self.root, text="Sample:", font=("Helvetica", 10,"bold"))
        self.file_label.place(anchor = 'nw', relx=0.84, rely=0.12)
        self.filepath_label = tk.Label(self.root, text="", font=("Helvetica", 10))
        self.filepath_label.place(anchor = 'nw', relx=0.84, rely=0.15)
        
        # Labels to set what temperature the gradient should be calculated on the curve
        self.gradientTemp_label = tk.Label(self.root, text="Temp of gradient (°C) ", font=("Helvetica", 10,"bold"))
        self.gradientTemp_label.place(anchor = 'nw', relx=0.84, rely=0.20)
        self.gradientTemp_number = tk.Entry(self.root, justify='center', width=5)  # Align text right
        self.gradientTemp_number.place(anchor = 'nw', relx=0.84, rely=0.24, relwidth=0.15)
        self.gradientTemp_number.insert(0,"24")
        self.gradientTemp_number.bind("<FocusOut>", self.on_concentration_exit)
        
        # Labels to change concentration variable
        self.concentration_label = tk.Label(self.root, text="Concentration (mg/mL) ", font=("Helvetica", 10,"bold"))
        self.concentration_label.place(anchor = 'nw', relx=0.84, rely=0.28)
        self.concentration_number = tk.Entry(self.root, justify='center', width=5)  # Align text right
        self.concentration_number.place(anchor = 'nw', relx=0.84, rely=0.32, relwidth=0.15)
        self.concentration_number.insert(0,"1")
        self.concentration_number.bind("<FocusOut>", self.on_concentration_exit)
        
        # Label to display time and temperature of vertical lines
        self.vline1_time_label = tk.Label(self.root,text="Time (s):",font=("Helvetica", 10, "bold"),fg="red")
        self.vline1_time_label.place(anchor = 'nw', relx=0.84, rely=0.36)
        self.vline1_time_number = tk.Label(self.root,text="0",font=("Helvetica", 10, ),fg="red")
        self.vline1_time_number.place(anchor = 'ne', relx=0.98, rely=0.36)
        self.vline1_temperature_label = tk.Label(self.root,text="Tempearture (°C):",font=("Helvetica", 10, "bold"),fg="red")
        self.vline1_temperature_label.place(anchor = 'nw', relx=0.84, rely=0.40)
        self.vline1_temperature_number = tk.Label(self.root,text="0",font=("Helvetica", 10, ),fg="red")
        self.vline1_temperature_number.place(anchor = 'ne', relx=0.98, rely=0.40)

        self.vline2_time_label = tk.Label(self.root,text="Time (s):",font=("Helvetica", 10, "bold"),fg="blue")
        self.vline2_time_label.place(anchor = 'nw', relx=0.84, rely=0.44)
        self.vline2_time_number = tk.Label(self.root,text="0",font=("Helvetica", 10, ),fg="blue")
        self.vline2_time_number.place(anchor = 'ne', relx=0.98, rely=0.44)
        self.vline2_temperature_label = tk.Label(self.root,text="Tempearture (°C):",font=("Helvetica", 10, "bold"),fg="blue")
        self.vline2_temperature_label.place(anchor = 'nw', relx=0.84, rely=0.48)
        self.vline2_temperature_number = tk.Label(self.root,text="0",font=("Helvetica", 10, ),fg="blue")
        self.vline2_temperature_number.place(anchor = 'ne', relx=0.98, rely=0.48)
        
        #Labels for SAR values
        self.SAR_Label = tk.Label(self.root,text="SAR Calculation (W/g)",font=("Helvetica", 10, "bold","underline"))
        self.SAR_Label.place(anchor = 'nw', relx=0.84, rely=0.54)
        
        self.SAR_exponential_label = tk.Label(self.root,text="Exponential:",font=("Helvetica", 10, "bold"))
        self.SAR_exponential_label.place(anchor = 'nw', relx=0.84, rely=0.58)
        self.SAR_exponential_number = tk.Label(self.root,text="NaN",font=("Helvetica", 10))
        self.SAR_exponential_number.place(anchor = 'ne', relx=0.98, rely=0.58)
        
        self.SAR_exponentialE_label = tk.Label(self.root,text="Error:",font=("Helvetica", 10, "bold"))
        self.SAR_exponentialE_label.place(anchor = 'nw', relx=0.84, rely=0.62)
        self.SAR_exponentialE_number = tk.Label(self.root,text="NaN",font=("Helvetica", 10))
        self.SAR_exponentialE_number.place(anchor = 'ne', relx=0.98, rely=0.62)
        
        self.SAR_linear_label = tk.Label(self.root,text="Linear:",font=("Helvetica", 10, "bold"))
        self.SAR_linear_label.place(anchor = 'nw', relx=0.84, rely=0.66)
        self.SAR_linear_number = tk.Label(self.root,text="NaN",font=("Helvetica", 10))
        self.SAR_linear_number.place(anchor = 'ne', relx=0.98, rely=0.66)
        
        self.SAR_linearE_label = tk.Label(self.root,text="Error:",font=("Helvetica", 10, "bold"))
        self.SAR_linearE_label.place(anchor = 'nw', relx=0.84, rely=0.70)
        self.SAR_linearE_number = tk.Label(self.root,text="NaN",font=("Helvetica", 10))
        self.SAR_linearE_number.place(anchor = 'ne', relx=0.98, rely=0.70)
        
        
        #Initialize variables for the plot and vertical lines
        self.fig = None
        self.ax = None
        self.canvas = None
        self.vline1 = None
        self.vline2 = None
        self.dragging_vline = None
        self.fig, self.ax = plt.subplots()
        
        self.last_drag_time = 0.1
        self.drag_delay = 0.1

    def open_file(self):
        
        # Open file dialog to select a .txt file
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        
        if file_path:
            
            # Remove the old canvas if it exists
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
        
            # Clear the figure and axes
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
        
            # Reset vertical lines
            self.vline1 = None
            self.vline2 = None
            self.dragging_vline = None
        
            # Reset SAR labels
            self.SAR_exponential_number.config(text="NaN")
            self.SAR_exponentialE_number.config(text="NaN")
            self.SAR_linear_number.config(text="NaN")
            self.SAR_linearE_number.config(text="NaN")
        
            # Reset vertical line labels
            self.vline1_time_number.config(text="0")
            self.vline1_temperature_number.config(text="0")
            self.vline2_time_number.config(text="0")
            self.vline2_temperature_number.config(text="0")
            
            # Read the data from the file
            with open(file_path, newline='', encoding='utf-8') as openFile:
                #file = csv.DictReader(openFile, delimiter='\t')
                file = pd.read_csv("C:\\Users\\Kingsley Poon\\Desktop\\Github\\Specific-Absoprtion-Rate-Calculator\\Sample Data.txt",skiprows = 2, header = 0, sep = '\t')
                #Display the filepath name
                self.filepath_label.config(text = os.path.basename(file_path))
                
                # Get Temperature and time data
                
                self.temperature = file["T1"].tolist()
                self.time = file["t"].tolist()
                
                self.temperature = self.temperature[5:]
                self.time = self.time[5:]
                

                # Convert lists to numpy arrays
                self.temperature = np.asarray(self.temperature, dtype=float)
                self.time = np.asarray(self.time, dtype=float)
            

            # Plot the temperature vs. time
            self.ax.plot(self.time, self.temperature, label="Temperature", color='black')
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Temperature (°C)")

                
            # Create the canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    
            # Add vertical lines
            self.vline1 = self.ax.axvline(x=2, color='red', linestyle='--')
            self.vline2 = self.ax.axvline(x=8, color='blue', linestyle='--')
    
            self.start_interaction()
            self.canvas.draw()


    def on_press(self, event):
        
        # Check if we're clicking near one of the vertical lines
        if event.inaxes == self.ax:
            
            # Get the width of the plot (x-axis limits)
            graph_width = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            
            # Set sensitivity relative to the graph width 
            sensitivity = 0.05 * graph_width  # 5% of the graph's width
            
            #Move the line if clicked near the line based on sensitivity set
            if abs(event.xdata - self.vline1.get_xdata()[0]) < sensitivity:
                self.dragging_vline = self.vline1
            elif abs(event.xdata - self.vline2.get_xdata()[0]) < sensitivity:
                self.dragging_vline = self.vline2

    def calculate_SAR(self):
        
        try:
            # Get the bounds of the vertical lines (x-values)
            x1 = self.vline1.get_xdata()[0]
            x2 = self.vline2.get_xdata()[0]
    
            # Ensure x1 is the smaller value
            if x1 > x2:
                x1, x2 = x2, x1
                
            # Filter the data within the bounds of the vertical lines
            mask = (self.time >= x1) & (self.time <= x2)
            time_fit = self.time[mask]
            temp_fit = self.temperature[mask]
            
            # Identify the points where the temperature increases
            selected_time = []
            selected_temp = []
            
            for i in range(1, len(temp_fit)):
                if temp_fit[i] > temp_fit[i-1]:  # Check if the temperature increases
                    selected_time.append(time_fit[i-1])  # Add the time before the increase
                    selected_temp.append(temp_fit[i-1])  # Add the temperature before the increase
            
            # Add the last point as well
            selected_time.append(time_fit[-1])
            selected_temp.append(temp_fit[-1])
            
            # Determine concentration of NPs from textbox
            concentration = float(self.concentration_number.get().strip())
            
            # Fit the exponential curve
            if len(selected_time) >= 2:  # Ensure enough data points to fit the model
                try:
                    popt_exp, _ = scipy.optimize.curve_fit(monoExp, np.array(selected_time), np.array(selected_temp), p0=[min(selected_temp), max(selected_temp), 0.001])
    
                    tint, tfin, invTau = popt_exp
                    gradient_exp = -invTau * (tint - tfin) * np.exp(-invTau * float(self.gradientTemp_number.get().strip()))
                    
                    #calculate SAR and error (euclidean distance between curve and data points)
                    SAR = self.heatCapacityWater/(concentration/1000)*gradient_exp
                    self.SAR_exponential_number.config(text=f"{SAR:.4f}")
                    exp_distances = point_to_curve_distance(selected_time, selected_temp, monoExp, popt_exp)
                    avg_exp_dist = np.mean(exp_distances)
                    self.SAR_exponentialE_number.config(text=f"{avg_exp_dist:.4f}")
                    
                except:
                    self.SAR_exponential_number.config(text="NaN")
                    self.SAR_exponentialE_number.config(text="NaN")
                
                try:
                    # Fit the linear curve
                    popt_lin, _ = scipy.optimize.curve_fit(monoLin, np.array(selected_time), np.array(selected_temp))
            
                    # Linear gradient at 24°C (the gradient is constant)
                    a, b = popt_lin
                    gradient_lin = a  # The gradient of the linear model is the slope 'a'
                    
                    #calculate SAR and error (euclidean distance between curve and data points)
                    SAR = self.heatCapacityWater/(concentration/1000)*gradient_lin
                    self.SAR_linear_number.config(text=f"{SAR:.4f}")
                    lin_distances = point_to_curve_distance(selected_time, selected_temp, monoLin, popt_lin)
                    avg_lin_dist = np.mean(lin_distances)
                    self.SAR_linearE_number.config(text=f"{avg_lin_dist:.4f}")
                    
                except:
                    self.SAR_linear_number.config(text="NaN")
                    self.SAR_linearE_number.config(text="NaN")
        except:
            pass
        
    def on_release(self, event):
        self.dragging_vline = None
        
        #Calculate SAR when line is released
        self.calculate_SAR()
    
    def on_concentration_exit(self,event):
        #Calculate SAR when exiting textbox to update values
        self.calculate_SAR()
        
    def on_motion(self, event):
        if self.dragging_vline and event.inaxes == self.ax:
            current_time = time.time()
            if current_time - self.last_drag_time >= self.drag_delay:  # Throttle drag redraw
                # Move the line to the current x position of the mouse
                self.dragging_vline.set_xdata([event.xdata])
                self.canvas.draw()
                self.last_drag_time = current_time  # Update time of last drag
    
                # Update the temperature and time labels
                x = event.xdata
                y = np.interp(x, self.time, self.temperature)
    
                if self.dragging_vline == self.vline1:
                    self.vline1_time_number.config(text=f"{x:.0f}")
                    self.vline1_temperature_number.config(text=f"{y:.1f}")
                elif self.dragging_vline == self.vline2:
                    self.vline2_time_number.config(text=f"{x:.0f}")
                    self.vline2_temperature_number.config(text=f"{y:.1f}")
        
    def start_interaction(self):
        # Connect the events to handle dragging
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

#Exponential function for time vs tempearture        
def monoExp(t,tint,tfin,invTau):
    return (tint-tfin)*np.exp(-invTau*t)+tfin

#Linear function for time vs temperature
def monoLin(t,a,b):
    return a*t + b

#Function that calculates the Euclidean distance between the data points and curve
def point_to_curve_distance(x_points, y_points, func, popt):
    distances = []
    for x0, y0 in zip(x_points, y_points):
        # Define a function to minimize: squared distance between (x0, y0) and (x, func(x))
        def distance_squared(x):
            y = func(x, *popt)
            return (x - x0)**2 + (y - y0)**2

        # Minimize the distance with an initial guess near x0
        result = scipy.optimize.minimize(distance_squared, x0)
        if result.success:
            distances.append(math.sqrt(result.fun))
        else:
            distances.append(float('inf'))  # In case optimization fails
    return distances

# Create and run the app
root = tk.Tk()
app = GraphApp(root)
root.mainloop()
