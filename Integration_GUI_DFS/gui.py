import os
import sys
import tkinter as tk
from tkinter import messagebox

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
INTEGRATION_DIR = os.path.join(PROJECT_ROOT, "src", "integration")

sys.path.append(INTEGRATION_DIR)

from integration import get_routes


def find_routes():
    origin = origin_entry.get().strip()
    destination = destination_entry.get().strip()
    time_value = time_entry.get().strip()

    if not origin:
        messagebox.showerror("Input Error", "Please enter an origin SCATS site.")
        return

    if not destination:
        messagebox.showerror("Input Error", "Please enter a destination SCATS site.")
        return

    if origin == destination:
        messagebox.showwarning("Input Warning", "Origin and destination cannot be the same.")
        return

    if not time_value:
        messagebox.showerror("Input Error", "Please enter prediction time, for example 08:00.")
        return

    try:
        routes = get_routes(origin, destination, time_value)

        result_box.delete("1.0", tk.END)

        if not routes:
            result_box.insert(tk.END, "No route found for the selected SCATS sites.\n")
            return

        result_box.insert(tk.END, "Top route recommendations:\n\n")

        for index, (path, travel_time_seconds) in enumerate(routes, start=1):
            travel_time_minutes = travel_time_seconds / 60

            result_box.insert(
                tk.END,
                f"Route {index}: {' -> '.join(path)}\n"
                f"Estimated Travel Time: {travel_time_minutes:.2f} minutes\n\n"
            )

    except Exception as error:
        messagebox.showerror("System Error", str(error))


window = tk.Tk()
window.title("TBRGS - Traffic Based Route Guidance System")
window.geometry("760x540")

title_label = tk.Label(
    window,
    text="Traffic Based Route Guidance System",
    font=("Arial", 17, "bold")
)
title_label.pack(pady=12)

subtitle_label = tk.Label(
    window,
    text="Member 4: Integration and GUI using DFS route search",
    font=("Arial", 10)
)
subtitle_label.pack(pady=4)

form_frame = tk.Frame(window)
form_frame.pack(pady=15)

tk.Label(form_frame, text="Origin SCATS Site:").grid(row=0, column=0, padx=8, pady=8, sticky="e")
origin_entry = tk.Entry(form_frame, width=32)
origin_entry.insert(0, "2000")
origin_entry.grid(row=0, column=1, padx=8, pady=8)

tk.Label(form_frame, text="Destination SCATS Site:").grid(row=1, column=0, padx=8, pady=8, sticky="e")
destination_entry = tk.Entry(form_frame, width=32)
destination_entry.insert(0, "3977")
destination_entry.grid(row=1, column=1, padx=8, pady=8)

tk.Label(form_frame, text="Prediction Time:").grid(row=2, column=0, padx=8, pady=8, sticky="e")
time_entry = tk.Entry(form_frame, width=32)
time_entry.insert(0, "08:00")
time_entry.grid(row=2, column=1, padx=8, pady=8)

find_button = tk.Button(
    window,
    text="Find Top Routes",
    command=find_routes,
    width=22,
    height=2
)
find_button.pack(pady=10)

result_box = tk.Text(window, height=16, width=88)
result_box.pack(pady=10)

window.mainloop()