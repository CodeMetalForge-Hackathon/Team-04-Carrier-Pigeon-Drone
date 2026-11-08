"""Minimal Bluetooth LE link to a Crazyflie 2.x (no Crazyradio needed).

cflib has no BLE driver, so this talks CRTP straight over the Crazyflie GATT service
(same protocol the phone app uses) with `bleak`. It exposes the few cflib-style calls
the controller needs: commander.send_setpoint / send_hover_setpoint / send_stop_setpoint /
send_notify_setpoint_stop and platform.send_arming_request.

    python cf_ble.py                 # scan, list Crazyflies, ping the nearest one
    python cf_ble.py F9:C5:66:38:6D:AA

Wire format (crazyflie2-nrf-firmware src/ble/ble_crazyflies.c):
  write 0x0202  raw CRTP packet (header + data, <= 20 bytes)
  notify 0x0204 replies: byte0 = 0x80 | pid<<5 | len, then the CRTP packet
"""
import asyncio
import json
import os
import struct
import sys
import threading
import time

from bleak import BleakClient, BleakScanner

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
PORT_LOG = 5
LOG_TOC, LOG_SETTINGS, LOG_DATA = 0, 1, 2
FLOAT, FP16 = 0x07, 0x08

_BASE = "-1c7f-4f9e-947b-43b7c00a9a08"
CRTP_CHAR = "00000202" + _BASE
CRTPDOWN_CHAR = "00000204" + _BASE

PORT_COMMANDER, PORT_PLATFORM, PORT_GENERIC, PORT_LINK = 3, 13, 7, 15


def header(port, channel=0):
    return ((port & 0x0F) << 4) | (3 << 2) | (channel & 0x03)


async def _scan(timeout=6.0):
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    cfs = [(d.address, a.rssi) for d, a in found.values() if (d.name or "").startswith("Crazyflie")]
    return sorted(cfs, key=lambda x: -x[1])


def scan(timeout=6.0):
    return asyncio.run(_scan(timeout))


def pick_address(address=None):
    """Return the BLE address to use. Refuses to guess if several drones are in range."""
    if address:
        return address
    cfs = scan()
    for addr, rssi in cfs:
        print(f"  Crazyflie {addr}  RSSI {rssi} dBm")
    if not cfs:
        raise SystemExit("No Crazyflie found over Bluetooth. Is it switched on (and USB unplugged)?")
    if len(cfs) > 1 and cfs[0][1] - cfs[1][1] < 15:
        raise SystemExit("Several Crazyflies with similar signal - pass yours explicitly, e.g. --ble "
                         + cfs[0][0])
    print("Using nearest:", cfs[0][0])
    return cfs[0][0]


class _Commander:
    def __init__(self, link):
        self._l = link

    def send_setpoint(self, roll, pitch, yawrate, thrust):
        # same packing as cflib (pitch sign flipped for the legacy RPYT commander)
        self._l.send(header(PORT_COMMANDER), struct.pack("<fffH", roll, -pitch, yawrate, int(thrust)))

    def send_hover_setpoint(self, vx, vy, yawrate, zdistance):
        if self._l.protocol_version <= 8:   # legacy hover type, yaw sign inverted
            data = struct.pack("<Bffff", 5, vx, vy, -yawrate, zdistance)
        else:
            data = struct.pack("<Bffff", 10, vx, vy, yawrate, zdistance)
        self._l.send(header(PORT_GENERIC, 0), data)

    def send_stop_setpoint(self):
        self._l.send(header(PORT_GENERIC, 0), b"\x00", wait=True)

    def send_notify_setpoint_stop(self, remain_valid_milliseconds=0):
        self._l.send(header(PORT_GENERIC, 1), struct.pack("<BI", 0, remain_valid_milliseconds), wait=True)


class _Platform:
    def __init__(self, link):
        self._l = link

    def send_arming_request(self, do_arm):
        self._l.send(header(PORT_PLATFORM, 0), bytes([1, 1 if do_arm else 0]), wait=True)


class BleCrazyflie:
    """Context manager: `with BleCrazyflie(addr) as cf: cf.commander.send_setpoint(...)`."""

    def __init__(self, address, protocol_version=7):
        self.address = address
        self.protocol_version = protocol_version
        self.rx = []                       # raw CRTP packets received (header + data)
        self.on_log = None                 # callback(block_name, timestamp_ms, {var: value})
        self._blocks = {}                  # block id -> (name, [var names])
        self._frag = b""
        self._frag_len = 0
        self.commander = _Commander(self)
        self.platform = _Platform(self)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._client = None

    # ---- plumbing -------------------------------------------------------
    def _run(self, coro, timeout=20):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _on_down(self, _h, data):
        data = bytes(data)
        if not data:
            return
        # byte0 bits 0-4 = total packet length. Packets > 19 bytes arrive in two notifications,
        # and on this firmware the 2nd one is [ctrl, pk[20:], junk] - byte 19 is LOST (we put
        # 0 there). Log blocks are therefore sized to never split; see start_logging().
        incomplete = self._frag and len(self._frag) < self._frag_len
        if incomplete:
            self._frag = (self._frag + b"\x00" + data[1:])[:self._frag_len]
        elif data[0] & 0x80:
            self._frag_len = data[0] & 0x1F
            self._frag = data[1:]
        else:
            return
        if len(self._frag) >= self._frag_len:
            pk, self._frag = self._frag, b""
            self._handle(pk)

    def _handle(self, pk):
        if not pk:
            return
        pk = bytes([pk[0] | 0x0C]) + pk[1:]   # replies come with the link bits cleared
        if pk[0] == header(PORT_LOG, LOG_DATA) and len(pk) >= 5:
            blk = self._blocks.get(pk[1])
            if blk and self.on_log:
                ts = pk[2] | pk[3] << 8 | pk[4] << 16
                vals = struct.unpack(blk[2], pk[5:5 + struct.calcsize(blk[2])])
                self.on_log(blk[0], ts, dict(zip(blk[1], vals)))
            return
        self.rx.append(pk)
        if len(self.rx) > 200:
            del self.rx[:100]

    def _request(self, hdr, data, match, tries=4, wait=0.5):
        """Send a request and wait for a reply whose bytes start with `match`."""
        for _ in range(tries):
            self.send(hdr, data, wait=True)
            end = time.time() + wait
            while time.time() < end:
                for pk in list(self.rx):
                    if pk.startswith(match):
                        self.rx.remove(pk)
                        return pk
                time.sleep(0.02)
        return None

    # ---- logging --------------------------------------------------------
    def _log_toc(self, names):
        """Map 'group.name' -> TOC id. Uses cflib's cache if the CRC matches, else asks the drone."""
        hdr = header(PORT_LOG, LOG_TOC)
        info = self._request(hdr, b"\x03", bytes([hdr, 3]))
        if not info:
            raise RuntimeError("drone did not answer the log TOC request")
        n_items, crc = struct.unpack("<HI", info[2:8])
        path = os.path.join(CACHE_DIR, "%08X.json" % crc)
        ids = {}
        if os.path.exists(path):
            toc = json.load(open(path))
            for full in names:
                g, v = full.split(".", 1)
                ids[full] = toc[g][v]["ident"]
            return ids
        print(f"Log TOC {crc:08X} not cached - reading {n_items} entries over BLE (one-off, ~1 min)...")
        toc = {}
        for i in range(n_items):
            r = self._request(hdr, struct.pack("<BH", 2, i), bytes([hdr, 2]) + struct.pack("<H", i))
            if r:
                group, name = r[5:].split(b"\x00")[:2]
                toc.setdefault(group.decode(), {})[name.decode()] = {"ident": i, "ctype": r[4]}
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(toc, open(path, "w"))
        for full in names:
            g, v = full.split(".", 1)
            ids[full] = toc[g][v]["ident"]
        return ids

    def start_logging(self, blocks):
        """blocks: {name: (period_ms, [var names], "fp16" | "float")}.
        Values arrive via self.on_log(name, drone_ms, {var: value}).

        Each log packet must fit ONE BLE notification (19 bytes = 5 header + 14 data):
        this firmware drops byte 19 when it splits a longer packet in two. So a block
        holds at most 3 floats or 7 fp16 values (fp16 ~ 3-4 significant digits)."""
        all_vars = [v for b in blocks.values() for v in b[1]]
        ids = self._log_toc(all_vars)
        hdr = header(PORT_LOG, LOG_SETTINGS)
        self._request(hdr, b"\x05", bytes([hdr, 5]))                 # reset: drop old blocks
        for bid, (name, spec) in enumerate(blocks.items(), start=1):
            period, vars_ = spec[0], spec[1]
            kind = spec[2] if len(spec) > 2 else "float"
            fetch, size = (FP16, 2) if kind == "fp16" else (FLOAT, 4)
            if 5 + size * len(vars_) > 19:
                raise ValueError(f"log block {name}: max {14 // size} {kind} values over BLE")
            # TOC variables: firmware uses its own storage type; we only choose the fetch type
            # a BLE write holds 20 bytes -> 5 vars per command: CREATE_V2 (6), then APPEND_V2 (7)
            for i in range(0, len(vars_), 5):
                cmd = 6 if i == 0 else 7
                data = bytes([cmd, bid])
                for v in vars_[i:i + 5]:
                    data += bytes([(FLOAT << 4) | fetch]) + struct.pack("<H", ids[v])
                r = self._request(hdr, data, bytes([hdr, cmd, bid]))
                if not r or r[3] != 0:
                    raise RuntimeError(f"creating log block {name} failed: {r}")
            self._blocks[bid] = (name, vars_, "<%d%s" % (len(vars_), "e" if fetch == FP16 else "f"))
            r = self._request(hdr, bytes([3, bid, max(1, period // 10)]), bytes([hdr, 3, bid]))
            if not r or r[3] != 0:
                raise RuntimeError(f"starting log block {name} failed: {r}")

    def reboot(self):
        """Reboot the whole drone (like the power button). Clears the supervisor lock that
        firmware 2023.11 sets after a tumble. The BLE link drops; reconnect after ~5 s."""
        for data in (b"\xfe\xff", b"\xfe\xf0\x01"):     # bootloader RESET_INIT, RESET to firmware
            try:
                self.send(0xF3, data, wait=True)
            except Exception:
                pass
            time.sleep(0.3)

    def stop_logging(self):
        hdr = header(PORT_LOG, LOG_SETTINGS)
        self._request(hdr, b"\x05", bytes([hdr, 5]), tries=2)
        self._blocks.clear()

    async def _pump(self):
        """The drone only sends a packet back when it receives one, so keep polling with
        empty (0xFF) packets; anything else we send also counts as a poll."""
        while self._client and self._client.is_connected:
            if time.time() - self._last_tx > 0.008:
                try:
                    await self._client.write_gatt_char(CRTP_CHAR, b"\xff", response=False)
                except Exception:
                    pass
                self._last_tx = time.time()
            await asyncio.sleep(0.01)

    async def _connect(self, attempts=3):
        for i in range(1, attempts + 1):
            print(f"Connecting to {self.address} over BLE (attempt {i}/{attempts})...", flush=True)
            self._client = BleakClient(self.address, timeout=10)
            try:
                await self._client.connect()
                await self._client.start_notify(CRTPDOWN_CHAR, self._on_down)
                self._last_tx = 0.0
                self._pump_task = self._loop.create_task(self._pump())
                return
            except Exception as e:
                print(f"  failed: {type(e).__name__}: {e}", flush=True)
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
                await asyncio.sleep(1.5)
        raise ConnectionError(
            "Could not connect over BLE. The Crazyflie allows ONE connection at a time:\n"
            "  - close any other cf_*.py window / the Crazyflie phone app\n"
            "  - power-cycle the drone (and keep the USB cable unplugged), then retry")

    def send(self, hdr, data=b"", wait=False):
        pk = bytes([hdr]) + bytes(data)
        if len(pk) > 20:
            raise ValueError("BLE CRTP packet > 20 bytes")
        self._last_tx = time.time()
        fut = asyncio.run_coroutine_threadsafe(
            self._client.write_gatt_char(CRTP_CHAR, pk, response=wait), self._loop)
        if wait:
            fut.result(5)
        else:
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # don't spam on drop

    def ping(self):
        """Ask for the firmware version string; returns it (or None)."""
        self.rx.clear()
        for _ in range(3):
            self.send(header(PORT_PLATFORM, 1), b"\x01", wait=True)
            self._run(asyncio.sleep(0.4))
            for pk in self.rx:
                if pk[:2] == bytes([header(PORT_PLATFORM, 1), 1]):
                    return pk[2:].decode(errors="replace")
        return None

    # ---- context manager -----------------------------------------------
    def __enter__(self):
        self._thread.start()
        try:
            self._run(self._connect(), timeout=60)
        except ConnectionError as e:
            self._loop.call_soon_threadsafe(self._loop.stop)
            raise SystemExit(str(e))
        except BaseException:
            self._loop.call_soon_threadsafe(self._loop.stop)
            raise
        return self

    def __exit__(self, *exc):
        try:
            self.commander.send_stop_setpoint()
        except Exception:
            pass
        try:
            task = getattr(self, "_pump_task", None)
            if task:
                self._loop.call_soon_threadsafe(task.cancel)
            self._run(self._client.disconnect(), timeout=5)
        except Exception:
            pass                                   # link may already be gone (e.g. after reboot)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


if __name__ == "__main__":
    addr = pick_address(sys.argv[1] if len(sys.argv) > 1 else None)
    with BleCrazyflie(addr) as cf:
        print("Connected over BLE. Firmware:", cf.ping())
        counts = {}
        cf.on_log = lambda name, ts, vals: (counts.__setitem__(name, counts.get(name, 0) + 1),
                                            print(f"{ts:9d} ms {name:6s} " + "  ".join(
                                                f"{k}={v:8.3f}" for k, v in vals.items())))
        cf.start_logging({"acc": (20, ["acc.x", "acc.y", "acc.z"]),
                          "gyro": (20, ["gyro.x", "gyro.y", "gyro.z"])})
        time.sleep(3)
        cf.stop_logging()
        print("packets in 3 s:", counts)
