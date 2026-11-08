# Crazyflie 2.1 — laptop control over Bluetooth

Keyboard flight + IMU logging for a Bitcraze Crazyflie 2.1, **without a Crazyradio dongle**.
`cflib` has no Bluetooth driver, so `cf_ble.py` speaks the Crazyflie's CRTP protocol directly
over BLE (via `bleak`).

| File | What it does |
|---|---|
| `cf_keyboard.py` | Keyboard teleop + CSV logging (BLE with `--ble`, or Crazyradio/USB via cflib) |
| `cf_ble.py` | Minimal BLE link: setpoints, arming, logging, reboot. `python cf_ble.py` = scan + ping |
| `cf_control.py` | cflib example: connect (radio/USB), stream battery + attitude, optional hover |
| `cache/` | Log TOC (variable list) from cflib — lets BLE logging skip a ~1 min TOC download |
| `flights/` | Recorded flights: acc (g), gyro (deg/s), roll/pitch/yaw, thrust, battery, commands |

## Setup (Windows)

```bash
pip install cflib bleak pynput
```

* **Bluetooth:** nothing else needed. Unplug the USB cable (otherwise replies go to USB).
* **USB cable (cflib `usb://0`):** install the **libusb-win32** driver on "Crazyflie 2.X" with
  [Zadig](https://zadig.akeo.ie) (Options → List All Devices).

## Fly

```bash
python cf_ble.py                                   # find your drone's BLE address
python cf_keyboard.py --ble <ADDRESS> --dry-run    # test keys, motors never commanded
python cf_keyboard.py --ble <ADDRESS>              # fly
```

| Key | Action |
|---|---|
| **T** | take off: spool up to 50 % in ~1 s (just below lift-off) |
| ↑ / ↓ | tap ±2 % thrust, hold = smooth ramp |
| W A S D | tilt forward / left / back / right (max 4°, eases in) |
| Q / E | yaw left / right |
| SPACE | land (ramp thrust to 0) |
| **ESC** | **emergency stop** |

Without a Flow deck it flies in **manual mode**: nothing holds height or position. Take off
decisively (T, then tap ↑ to ~57-60 %) — sitting on the ground at half thrust winds up the
attitude controller and flips it on lift-off. Every run writes `flights/flight_<date>_<time>.csv`.

## Gotchas we hit

* **Not thrusting, no error** → after a flip the firmware (2023.11) sets `supervisor` **LOCKED**
  and ignores thrust until reboot. `cf_keyboard.py` detects this and reboots the drone over BLE.
* **One BLE connection at a time** — the drone stops advertising while connected. Close other
  scripts / the phone app; power-cycle if it stays invisible.
* The drone only sends a BLE packet when it receives one → `cf_ble.py` polls with null packets.
* Firmware bug: BLE packets > 19 bytes are split and **byte 19 is lost**, so log blocks are kept
  to one notification (≤ 7 fp16 or 3 float values).
* Battery sags hard under load (3.8 V → ~3.3 V at hover thrust). Charge before flying.
