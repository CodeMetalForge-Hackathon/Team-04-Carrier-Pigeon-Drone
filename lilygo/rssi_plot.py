"""
RSSI Live Plotter — Laptop 2
Reads CSV from LilyGo RX over serial, plots live, saves to CSV log.

Usage:
    python rssi_plot.py

Requirements:
    pip install pyserial matplotlib
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import csv
from datetime import datetime

# ---- CONFIG — change COM port to match your RX LilyGo ----
PORT   = "COM11"   # check in Arduino IDE > Tools > Port
BAUD   = 115200
WINDOW = 200       # number of data points shown on screen

# ---- Log file (new file each run) ----
log_filename = f"rssi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, "w", newline="")
writer = csv.writer(log_file)
writer.writerow(["time_ms", "rssi", "snr", "loss_pct"])

# ---- Open serial port ----
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Connected to {PORT}")
    print(f"Logging to {log_filename}")
    print("Close this window to stop.\n")
except serial.SerialException as e:
    print(f"\nERROR: Could not open {PORT}")
    print("Fix: Close Arduino Serial Monitor / Arduino IDE first, then retry.\n")
    print(f"Details: {e}")
    exit(1)

# ---- Data buffers ----
times     = deque(maxlen=WINDOW)
rssi_vals = deque(maxlen=WINDOW)
snr_vals  = deque(maxlen=WINDOW)
loss_vals = deque(maxlen=WINDOW)

# ---- Plot setup ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle("LoRa RSSI Monitor", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])


def update(frame):
    # Read all waiting lines
    while ser.in_waiting:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue

        # Skip header and non-CSV lines
        if not line or not line[0].isdigit():
            continue

        parts = line.split(",")
        if len(parts) != 4:
            continue

        try:
            t    = float(parts[0])
            rssi = int(parts[1])
            snr  = float(parts[2])
            loss = float(parts[3])
        except ValueError:
            continue

        times.append(t / 1000)   # ms -> seconds
        rssi_vals.append(rssi)
        snr_vals.append(snr)
        loss_vals.append(loss)

        writer.writerow([t, rssi, snr, loss])
        log_file.flush()

    if not times:
        return

    t = list(times)

    # ---- RSSI plot ----
    ax1.clear()
    ax1.plot(t, list(rssi_vals), color="dodgerblue", linewidth=1.5, label="RSSI")
    ax1.axhline(-100, color="red", linestyle="--", linewidth=1, label="Jam threshold (-100 dBm)")
    ax1.set_ylabel("RSSI (dBm)")
    ax1.set_ylim(-130, -30)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- SNR plot ----
    ax2.clear()
    ax2.plot(t, list(snr_vals), color="limegreen", linewidth=1.5, label="SNR")
    ax2.axhline(0, color="orange", linestyle="--", linewidth=1, label="SNR = 0 (noise floor)")
    ax2.set_ylabel("SNR (dB)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ---- Packet loss plot ----
    ax3.clear()
    ax3.fill_between(t, list(loss_vals), color="red", alpha=0.4, label="Packet loss")
    ax3.set_ylabel("Loss (%)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ---- Jamming alert ----
    jammed = (
        len(rssi_vals) > 0
        and rssi_vals[-1] < -100
        and snr_vals[-1] < 0
        and loss_vals[-1] > 10
    )
    if jammed:
        fig.patch.set_facecolor("#ffe0e0")
        fig.suptitle("⚠  JAMMING DETECTED", color="red", fontsize=14, fontweight="bold")
    else:
        fig.patch.set_facecolor("white")
        fig.suptitle("LoRa RSSI Monitor", fontsize=13, fontweight="bold", color="black")


ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.show()

log_file.close()
ser.close()
print(f"\nDone. Log saved to: {log_filename}")
