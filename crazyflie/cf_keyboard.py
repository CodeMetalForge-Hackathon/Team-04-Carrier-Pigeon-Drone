"""Keyboard teleop for the Crazyflie 2.1, with IMU logging to CSV. (Raspberry-Pi ready.)

    T           TAKE OFF: spool up to just below lift-off (~1 s), then fine-tune with arrows
    UP / DOWN   tap = thrust +/-2%, hold = smooth ramp   (hover mode: height +/-0.1 m per tap)
    W / S       forward / back   (gentle 4 deg tilt, eases in over 1 s, levels on release)
    A / D       left / right
    Q / E       yaw left / right
    SPACE       land: ramp thrust down to zero
    ESC         EMERGENCY STOP - motors off immediately

Take off decisively: press T, then tap UP until it lifts. Sitting on the ground with
the motors half-on lets the attitude controller wind up, and it flips on lift-off.

Two modes, picked automatically:
  * HOVER  (Flow deck attached): arrows change target height, WASD = velocity.
           The drone holds position by itself when no key is pressed.
  * MANUAL (no positioning deck): arrows change thrust, WASD = tilt angle.
           Nothing holds it in place - it WILL drift; you are the stabiliser.

Keyboard backend (auto-selected, override with --keys):
  * pynput   : used on a desktop session (X11/Wayland) with pynput installed. True
               key-release, so "hold to ramp / release to level" works best.
  * terminal : used headless / over SSH, or when pynput is missing. Reads this
               terminal directly (no X needed). Taps work fully; "hold" is emulated
               from the OS key-repeat and tilt/yaw auto-level shortly after the last
               keypress. Run it in the terminal that has keyboard focus.

Every run records accelerometer, gyro, attitude, thrust, battery and your commands to
flights/flight_<date>_<time>.csv  (use --no-log to turn off).

    python3 cf_keyboard.py --ble F9:C5:66:38:6D:AA --dry-run   # Bluetooth, motors never commanded
    python3 cf_keyboard.py --ble F9:C5:66:38:6D:AA             # fly over Bluetooth
    python3 cf_keyboard.py                                     # fly over Crazyradio (default URI)
    python3 cf_keyboard.py --ble AA:BB.. --keys terminal       # force the SSH/headless backend
"""
import argparse
import csv
import os
import select
import sys
import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

RATE_HZ = 20
DT = 1.0 / RATE_HZ
HERE = os.path.dirname(os.path.abspath(__file__))

# gradual controls
STEP_PCT = 2           # thrust change per arrow tap, % of full thrust
HOLD_DELAY = 0.3       # s an arrow must be held before it starts ramping
HOLD_RATE_PCT = 8      # %/s while an arrow is held
THRUST_SLEW_PCT = 15   # actual thrust follows the target at most this many %/s (no jumps)
SPOOL_PCT = 50         # T jumps the target here: just below lift-off (this drone lifted ~57-60%)
SPOOL_SLEW_PCT = 50    # %/s while spooling up with T (gets through the ground phase fast)
THRUST_MAX_PCT = 80    # cap
LAND_PCT_PER_S = 15    # SPACE ramps down at this rate
RAMP = 0.05            # tilt/yaw/velocity move 5% of their max per tick (full in 1 s)
TILT_DEG = 4           # max roll/pitch angle for WASD
YAW_RATE = 45          # max deg/s for Q/E
LOW_BATTERY_V = 3.2    # status line warns below this

# hover mode (Flow deck)
HOVER_SPEED = 0.4      # max m/s for WASD
Z_STEP = 0.1           # m per arrow tap
TAKEOFF_Z = 0.4        # m
Z_MIN, Z_MAX = 0.1, 1.5
CLIMB_SPEED = 0.1      # m/s used for takeoff / landing ramps

LOG_BLOCKS = {         # name: (period ms, variables, fetch type) - over BLE max 7 fp16 / 3 float
    # acc in g, gyro in deg/s - one CSV row per imu sample
    "imu": (20, ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"], "fp16"),
    # attitude in deg, thrust in raw units (0-65535), battery in V, supervisor bitfield
    "state": (100, ["stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
                    "stabilizer.thrust", "pm.vbat", "supervisor.info"], "fp16"),
}

# supervisor.info bits (firmware 2023.11). After a tumble the drone sets LOCKED and ignores
# all thrust until it is rebooted - no error is reported, the motors just stay off.
SUP_ARMED, SUP_TUMBLED, SUP_LOCKED, SUP_CRASHED = 1, 5, 6, 7


def sup_bits(latest):
    v = latest.get("supervisor.info")
    return None if v is None else int(v)


def sup_problem(latest):
    s = sup_bits(latest)
    if s is None:
        return None
    if s >> SUP_LOCKED & 1:
        return "LOCKED (it tumbled) - motors disabled until reboot"
    if s >> SUP_TUMBLED & 1:
        return "TUMBLED - drone is upside down / tipped over"
    if s >> SUP_CRASHED & 1:
        return "CRASHED"
    return None


class _Rebooted(Exception):
    pass


class Keys:
    """Held-key state + control queries. Backends feed it via _down()/_up()."""

    def __init__(self):
        self.held = set()
        self.taps = 0          # +1 per UP tap, -1 per DOWN tap, consumed by the control loop
        self.pressed_at = {}   # key -> time it went down
        self.spool = False     # T pressed
        self.estop = False
        self.land = False
        self.lock = threading.Lock()

    def _down(self, n):
        if n is None:
            return
        if n == "esc":
            self.estop = True
            return
        if n == "space":
            self.land = True
        if n == "t":
            self.spool = True
        with self.lock:
            if n not in self.held:
                self.pressed_at[n] = time.time()
                if n in ("up", "down"):
                    self.taps += 1 if n == "up" else -1
            self.held.add(n)

    def _up(self, n):
        if n is None:
            return
        with self.lock:
            self.held.discard(n)

    def start(self):
        pass

    def stop(self):
        pass

    def take_taps(self):
        with self.lock:
            t, self.taps = self.taps, 0
            return t

    def axis(self, pos, neg):
        with self.lock:
            return (pos in self.held) - (neg in self.held)

    def held_long(self, pos, neg, delay):
        """+1/-1 if pos/neg has been held longer than delay seconds, else 0."""
        now = time.time()
        with self.lock:
            p = pos in self.held and now - self.pressed_at.get(pos, now) > delay
            n = neg in self.held and now - self.pressed_at.get(neg, now) > delay
            return p - n


class PynputKeys(Keys):
    """Desktop backend: real press/release events via pynput (needs X11/Wayland)."""

    def start(self):
        from pynput import keyboard
        self._kb = keyboard
        self._listener = keyboard.Listener(on_press=self._p, on_release=self._r)
        self._listener.start()

    def _name(self, k):
        if isinstance(k, self._kb.KeyCode) and k.char:
            return k.char.lower()
        return {self._kb.Key.up: "up", self._kb.Key.down: "down",
                self._kb.Key.space: "space", self._kb.Key.esc: "esc"}.get(k)

    def _p(self, k):
        n = self._name(k)
        if n == "esc":
            self._down("esc")
            return False  # stop the listener
        self._down(n)

    def _r(self, k):
        self._up(self._name(k))

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass


class TerminalKeys(Keys):
    """Headless/SSH backend: reads this terminal in raw mode. No X server needed.

    stdin gives key *presses* only (no release), so a key is treated as 'held' while
    the OS key-repeat keeps re-sending it, and auto-released HOLD_TIMEOUT after the
    last repeat. Thrust taps are exact; tilt/yaw ease back to level on key-up."""

    HOLD_TIMEOUT = 0.22

    def start(self):
        import termios
        import tty
        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)          # no echo, chars available immediately, Ctrl-C still works
        self._seen = {}
        self._stop = False
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._sweep_loop, daemon=True).start()

    def _tap(self, n):
        self._down(n)
        with self.lock:
            self._seen[n] = time.time()

    def _read_loop(self):
        while not self._stop:
            r, _, _ = select.select([self._fd], [], [], 0.1)
            if not r:
                continue
            try:
                data = os.read(self._fd, 16)
            except OSError:
                break
            self._parse(data)

    def _parse(self, data):
        i = 0
        while i < len(data):
            c = data[i]
            if c == 0x1b:  # ESC or an escape sequence (arrow keys)
                if data[i + 1:i + 2] == b"[" and i + 2 < len(data):
                    n = {65: "up", 66: "down", 67: None, 68: None}.get(data[i + 2])
                    if n:
                        self._tap(n)
                    i += 3
                    continue
                self._down("esc")        # bare ESC = emergency stop
                return
            ch = chr(c).lower()
            if c in (10, 13):
                pass
            elif ch == " ":
                self._down("space")
            elif ch == "t":
                self._down("t")
            elif ch in "wasdqe":
                self._tap(ch)
            i += 1

    def _sweep_loop(self):
        while not self._stop:
            now = time.time()
            with self.lock:
                for n in list(self.held):
                    if now - self._seen.get(n, 0) > self.HOLD_TIMEOUT:
                        self.held.discard(n)
            time.sleep(0.03)

    def stop(self):
        self._stop = True
        try:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)
        except Exception:
            pass


def _display_available():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def make_keys(backend):
    """Return a started Keys backend. 'auto' uses pynput on a desktop, else terminal."""
    want_pynput = backend == "pynput" or (backend == "auto" and _display_available())
    if want_pynput:
        try:
            import pynput  # noqa: F401
            k = PynputKeys()
            k.start()
            print("Keyboard: pynput (desktop)")
            return k
        except Exception as e:
            if backend == "pynput":
                raise SystemExit(f"pynput backend requested but unavailable: {e}")
            print(f"pynput unavailable ({e}); falling back to terminal keys")
    if not sys.stdin.isatty():
        raise SystemExit("terminal keyboard backend needs an interactive terminal (a TTY). "
                         "Run it directly in a terminal, not piped/detached.")
    k = TerminalKeys()
    k.start()
    print("Keyboard: terminal (raw stdin) - keep this terminal focused")
    return k


def ramp(current, target, step):
    """Move current toward target by at most step."""
    if current < target:
        return min(target, current + step)
    return max(target, current - step)


def has_flow_deck(cf):
    for deck in ("bcFlow2", "bcFlow"):
        try:
            if int(cf.param.get_value(f"deck.{deck}")):
                return True
        except Exception:
            pass
    return False


def fly_hover(cf, keys, send):
    z, airborne = 0.0, False
    vx = vy = yaw = 0.0
    print(f"HOVER mode - taking off to {TAKEOFF_Z} m")
    while not keys.estop:
        if keys.land:
            z -= CLIMB_SPEED * DT
            if z <= 0.05:
                break
            send("hover", 0, 0, 0, z)
        else:
            if not airborne:                        # takeoff ramp
                z = min(TAKEOFF_Z, z + CLIMB_SPEED * DT)
                airborne = z >= TAKEOFF_Z
                keys.take_taps()
            else:
                z = max(Z_MIN, min(Z_MAX, z + keys.take_taps() * Z_STEP))
            vx = ramp(vx, keys.axis("w", "s") * HOVER_SPEED, RAMP * HOVER_SPEED)
            vy = ramp(vy, keys.axis("a", "d") * HOVER_SPEED, RAMP * HOVER_SPEED)
            yaw = ramp(yaw, keys.axis("e", "q") * YAW_RATE, RAMP * YAW_RATE)
            send("hover", vx, vy, yaw, z)
        time.sleep(DT)


def fly_manual(cf, keys, send):
    target = thrust = 0.0      # percent of full thrust
    roll = pitch = yaw = 0.0
    print(f"MANUAL mode - no Flow deck. Press T to spool up to {SPOOL_PCT}%, "
          f"then tap UP (+{STEP_PCT}%) until it lifts (~57-60%).")
    send("manual", 0, 0, 0, 0)  # zero-thrust packet unlocks the thrust lock
    spooling = False
    while not keys.estop:
        if keys.land:
            target = 0.0
            thrust = max(0.0, thrust - LAND_PCT_PER_S * DT)
            if thrust <= 0:
                break
        else:
            if keys.spool:                       # T: go straight to just-below-lift-off
                keys.spool = False
                if target < SPOOL_PCT:
                    target, spooling = SPOOL_PCT, True
            target += keys.take_taps() * STEP_PCT
            target += keys.held_long("up", "down", HOLD_DELAY) * HOLD_RATE_PCT * DT
            target = max(0, min(THRUST_MAX_PCT, target))
            spooling = spooling and thrust < target
            thrust = ramp(thrust, target, (SPOOL_SLEW_PCT if spooling else THRUST_SLEW_PCT) * DT)
        pitch = ramp(pitch, keys.axis("w", "s") * TILT_DEG, RAMP * TILT_DEG)
        roll = ramp(roll, keys.axis("d", "a") * TILT_DEG, RAMP * TILT_DEG)
        yaw = ramp(yaw, keys.axis("e", "q") * YAW_RATE, RAMP * YAW_RATE)
        send("manual", roll, pitch, yaw, int(thrust / 100 * 65535), target)
        time.sleep(DT)


class FlightLog:
    """Writes one CSV row per IMU sample, with the latest attitude/power/command."""

    COLS = (["t_pc", "t_drone_ms"] + [v for b in LOG_BLOCKS.values() for v in b[1]]
            + ["cmd_kind", "cmd_roll_or_vx", "cmd_pitch_or_vy", "cmd_yawrate", "cmd_thrust_or_z"])

    def __init__(self):
        os.makedirs(os.path.join(HERE, "flights"), exist_ok=True)
        self.path = os.path.join(HERE, "flights", time.strftime("flight_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.COLS)
        self.latest = {}
        self.cmd = ("", 0, 0, 0, 0)
        self.rows = 0
        self.lock = threading.Lock()

    def on_log(self, block, ts, vals):
        with self.lock:
            self.latest.update(vals)
            if block != "imu":
                return
            row = [f"{time.time():.3f}", ts]
            row += [f"{self.latest[c]:.4f}" if c in self.latest else "" for c in self.COLS[2:-5]]
            row += [self.cmd[0]] + [f"{x:.3f}" for x in self.cmd[1:]]
            self.w.writerow(row)
            self.rows += 1

    def close(self):
        with self.lock:
            self.f.close()


def start_cflib_logging(cf, on_log):
    for name, (period, vars_, _) in LOG_BLOCKS.items():
        conf = LogConfig(name=name, period_in_ms=period)
        for v in vars_:
            conf.add_variable(v, "float")   # radio/USB have no size issue: full floats
        cf.log.add_config(conf)
        conf.data_received_cb.add_callback(lambda ts, data, c: on_log(c.name, ts, data))
        conf.start()


class _BleSession:
    """Makes the BLE link look like SyncCrazyflie (`with ... as scf: scf.cf`)."""

    def __init__(self, link):
        self.cf = link

    def __enter__(self):
        self.cf.__enter__()
        return self

    def __exit__(self, *exc):
        self.cf.__exit__(*exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7")
    ap.add_argument("--dry-run", action="store_true", help="connect but never send motor commands")
    ap.add_argument("--mode", choices=["auto", "hover", "manual"], default="auto")
    ap.add_argument("--ble", nargs="?", const="scan", metavar="ADDRESS",
                    help="use Bluetooth instead of Crazyradio (optionally give the drone's address)")
    ap.add_argument("--keys", choices=["auto", "pynput", "terminal"], default="auto",
                    help="keyboard backend (default auto: pynput on a desktop, terminal over SSH)")
    ap.add_argument("--no-log", action="store_true", help="don't record IMU data to CSV")
    args = ap.parse_args()

    if args.uri.startswith("usb") and not args.dry_run and not args.ble:
        print("WARNING: flying while tethered by USB is unsafe. Ctrl+C now to abort (3 s)...")
        time.sleep(3)

    keys = make_keys(args.keys)

    if args.ble:
        from cf_ble import BleCrazyflie, pick_address
        address = pick_address(None if args.ble == "scan" else args.ble)
        # BLE link can't read deck params; assume no Flow deck unless --mode hover
        args.mode = "manual" if args.mode == "auto" else args.mode
    else:
        cflib.crtp.init_drivers()

    try:
        for attempt in range(2):          # 2nd pass only if we had to reboot a locked drone
            if args.ble:
                session = _BleSession(BleCrazyflie(address))
            else:
                session = SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache=os.path.join(HERE, "cache")))
            try:
                fly_session(args, keys, session, allow_reboot=(attempt == 0))
                break
            except _Rebooted:
                print("Waiting for the drone to reboot...")
                time.sleep(6)
    finally:
        keys.stop()


def fly_session(args, keys, session, allow_reboot):
    flog = None
    latest = {}                       # newest value of every logged variable

    def on_log(block, ts, vals):
        latest.update(vals)
        if flog:
            flog.on_log(block, ts, vals)

    with session as scf:
        cf = scf.cf
        print("Connected." + (f" Firmware {cf.ping()}" if args.ble else ""))
        mode = args.mode
        if mode == "auto":
            mode = "hover" if has_flow_deck(cf) else "manual"

        if not args.no_log:
            flog = FlightLog()
        if args.ble:
            cf.on_log = on_log
            cf.start_logging(LOG_BLOCKS)
        else:
            start_cflib_logging(cf, on_log)
        if flog:
            print("Logging IMU to", flog.path)

        # ---- supervisor check: a drone that tumbled last flight silently ignores thrust
        end = time.time() + 2.5
        while sup_bits(latest) is None and time.time() < end:
            time.sleep(0.05)
        problem = sup_problem(latest)
        if problem and args.ble and allow_reboot:
            print(f"\nDrone is {problem}. Rebooting it over Bluetooth...")
            cf.stop_logging()
            if flog:
                flog.close()
                os.remove(flog.path)
            cf.reboot()
            raise _Rebooted()
        if problem:
            print(f"\nDrone is {problem}. Power-cycle it (power button) and run again.")
            return
        if sup_bits(latest) is None:
            print("(no supervisor status received - can't check for a lock)")

        last_print = [0.0]
        stop_reason = []

        def send(kind, a, b, c, d, target=None):
            if flog:
                flog.cmd = (kind, a, b, c, d)
            problem = sup_problem(latest)
            if problem and not args.dry_run and not stop_reason:
                stop_reason.append(problem)
                keys.estop = True                     # ends the flight loop
            now = time.time()
            if now - last_print[0] > 0.25:
                if kind == "hover":
                    msg = f"vx {a:5.2f}  vy {b:5.2f}  yaw {c:6.1f}  z {d:4.2f} m"
                else:
                    msg = (f"thrust {d / 655.35:5.1f}% (target {target or 0:3.0f}%)  "
                           f"roll {a:5.1f}  pitch {b:5.1f}  yaw {c:6.1f}")
                rows = f"  | {flog.rows} rows" if flog else ""
                vbat = latest.get("pm.vbat")
                if vbat:
                    rows += f"  | {vbat:.2f} V" + ("  LOW BATTERY - LAND" if vbat < LOW_BATTERY_V else "")
                if problem:
                    rows += f"  | {problem}"
                print(f"\r[{kind}] {msg}{rows}   ", end="", flush=True)
                last_print[0] = now
            if args.dry_run:
                return
            if kind == "hover":
                cf.commander.send_hover_setpoint(a, b, c, d)
            else:
                cf.commander.send_setpoint(a, b, c, d)

        if not args.dry_run:
            cf.platform.send_arming_request(True)
            time.sleep(1.0)
        print(__doc__.split("Two modes")[0])
        try:
            (fly_hover if mode == "hover" else fly_manual)(cf, keys, send)
        except KeyboardInterrupt:
            pass
        finally:
            if not args.dry_run:
                cf.commander.send_stop_setpoint()
                cf.commander.send_notify_setpoint_stop()
                cf.platform.send_arming_request(False)
            if stop_reason:
                print(f"\nFlight stopped: drone is {stop_reason[0]}.\n"
                      "Just run the script again - it will reboot the drone and reconnect.")
            else:
                print("\nMotors stopped." + (" (ESC)" if keys.estop else ""))
            if args.ble:
                cf.stop_logging()
            if flog:
                flog.close()
                print(f"Saved {flog.rows} IMU rows to {flog.path}")


if __name__ == "__main__":
    main()
