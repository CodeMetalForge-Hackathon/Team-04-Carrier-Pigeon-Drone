"""Crazyflie 2.1 starter: scan, connect, read battery/attitude, optional short hover.

    python cf_control.py            # scan + connect + stream telemetry (no flying)
    python cf_control.py --fly      # take off to 0.3 m, hover 3 s, land  (needs Flow deck)
    python cf_control.py --uri radio://0/80/2M/E7E7E7E7E7

Requires: pip install cflib   + Crazyradio PA with the libusb driver (Zadig) on Windows.
"""
import argparse
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"  # factory default address


def find_uri(preferred):
    found = cflib.crtp.scan_interfaces(address=int(preferred.split("/")[-1], 16))
    if not found:
        found = cflib.crtp.scan_interfaces()
    print("Found:", [f[0] for f in found] or "nothing")
    return found[0][0] if found else preferred


def stream_telemetry(scf, seconds=5):
    lg = LogConfig(name="telemetry", period_in_ms=100)
    lg.add_variable("pm.vbat", "float")
    lg.add_variable("stabilizer.roll", "float")
    lg.add_variable("stabilizer.pitch", "float")
    lg.add_variable("stabilizer.yaw", "float")
    end = time.time() + seconds
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            print(f"bat {data['pm.vbat']:.2f} V | roll {data['stabilizer.roll']:6.1f} "
                  f"pitch {data['stabilizer.pitch']:6.1f} yaw {data['stabilizer.yaw']:6.1f}")
            if time.time() > end:
                break


def hover(scf):
    scf.cf.platform.send_arming_request(True)  # firmware 2023+ requires arming
    time.sleep(1.0)
    with MotionCommander(scf, default_height=0.10) as mc:  # takes off on enter, lands on exit
        time.sleep(3)
        # mc.forward(0.3); mc.turn_left(90); mc.up(0.2) ... see MotionCommander docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=uri_helper.uri_from_env(default=DEFAULT_URI))
    ap.add_argument("--fly", action="store_true")
    args = ap.parse_args()

    cflib.crtp.init_drivers()
    uri = find_uri(args.uri)
    print("Connecting to", uri)
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        print("Connected!")
        stream_telemetry(scf)
        if args.fly:
            hover(scf)


if __name__ == "__main__":
    main()
