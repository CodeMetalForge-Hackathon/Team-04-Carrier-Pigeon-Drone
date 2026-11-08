"""
LoRa Chat Client — Laptop 2
Reads incoming messages from LilyGo over serial.
Type a message here and press Enter to send it over LoRa.

Usage:
    python lora_chat.py

Requirements:
    pip install pyserial
"""

import serial
import threading
from datetime import datetime

PORT = "COM11"   # change to your RX LilyGo port
BAUD = 115200

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Connected to {PORT} — LoRa Chat ready")
    print("Type a message and press Enter to send. Ctrl+C to quit.\n")
    print("-" * 50)
except serial.SerialException as e:
    print(f"\nERROR: Could not open {PORT}")
    print("Close Arduino Serial Monitor / Arduino IDE first.\n")
    print(f"Details: {e}")
    exit(1)

def read_serial():
    """Background thread — continuously reads from LilyGo."""
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Incoming LoRa message
            if line.startswith("CHAT_RX:"):
                # Format: CHAT_RX:BOARD_1: hello|RSSI:-67|SNR:8.2
                parts = line[len("CHAT_RX:"):].split("|")
                msg  = parts[0] if len(parts) > 0 else ""
                rssi = parts[1].replace("RSSI:", "") if len(parts) > 1 else "?"
                snr  = parts[2].replace("SNR:", "")  if len(parts) > 2 else "?"
                print(f"\n[{timestamp()}] 📡 {msg}  ({rssi} dBm, SNR {snr} dB)")
                print("> ", end="", flush=True)   # reprint prompt

            # Debug lines from Arduino
            elif line.startswith("Sent:") or line.startswith("Received:"):
                pass   # suppress duplicate prints

        except Exception:
            continue

# Start background reader
t = threading.Thread(target=read_serial, daemon=True)
t.start()

# Main loop — send typed messages
try:
    while True:
        print("> ", end="", flush=True)
        msg = input()
        if msg.strip():
            ser.write((msg + "\n").encode("utf-8"))
            print(f"[{timestamp()}] ✉  You: {msg}")
except KeyboardInterrupt:
    print("\nExiting.")
    ser.close()
