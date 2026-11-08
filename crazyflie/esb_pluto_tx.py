#!/usr/bin/env python3
"""
Transmit nRF24L01+ Enhanced ShockBurst (ESB) packets with an ADALM-Pluto.

Packet format (dynamic-payload ESB, as used by RF24 with
setAutoAck(true) + enableDynamicPayloads()):

  preamble(8) | address(3-5 bytes) | PCF(9 bits: len6, PID2, NO_ACK1) | payload | CRC-16

Modulation: GFSK, BT=0.5, modulation index 0.32
(+/-160 kHz deviation at 1 Mbps, +/-320 kHz at 2 Mbps).

The script fills one cyclic TX buffer with 4 packets that use PIDs 0..3,
separated by silent gaps. Consecutive packets therefore have different PIDs,
so an nRF24 receiver does not discard them as duplicates.

Limitation: the Pluto cannot answer or check ESB ACKs in real time. USB
latency is milliseconds and ESB expects an answer within a few hundred µs.
The receiver still gets the packets, but your side sees no "ACK received".

Usage:
  pip install pyadi-iio numpy
  python3 esb_pluto_tx.py --addr NODE1 --channel 76 --rate 1 --payload hello
  python3 esb_pluto_tx.py --selftest        # offline check, no Pluto needed

Requires: Pluto drivers/libiio installed (https://wiki.analog.com/university/tools/pluto).
Use low gain and stay within your local 2.4 GHz ISM rules.
"""
import argparse
import sys
import time

import numpy as np


def bytes_to_bits(data):
    """MSB-first bit list (nRF24 sends every byte MSB first)."""
    return [(byte >> (7 - i)) & 1 for byte in data for i in range(8)]


def int_to_bits(value, width):
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def crc16_ccitt_bits(bits, crc=0xFFFF):
    """Bitwise CRC-16-CCITT (poly 0x1021, init 0xFFFF), as in nRF24."""
    for b in bits:
        crc ^= b << 15
        crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        crc &= 0xFFFF
    return crc


def build_esb_packet(addr_onair, payload, pid, no_ack=False):
    """Return ESB packet bits. addr_onair = address bytes in over-the-air order."""
    if not 3 <= len(addr_onair) <= 5:
        raise ValueError("address must be 3-5 bytes")
    if len(payload) > 32:
        raise ValueError("payload max 32 bytes")
    body = (bytes_to_bits(addr_onair)
            + int_to_bits(len(payload), 6)
            + int_to_bits(pid & 3, 2)
            + [1 if no_ack else 0]
            + bytes_to_bits(payload))
    crc = crc16_ccitt_bits(body)
    preamble = 0xAA if addr_onair[0] & 0x80 else 0x55
    return bytes_to_bits([preamble]) + body + int_to_bits(crc, 16)


def gfsk_modulate(bits, sps, bt=0.5, h=0.32):
    """Gaussian FSK, complex baseband, unit amplitude."""
    nrz = np.repeat(2 * np.asarray(bits, dtype=float) - 1, sps)
    span = 3  # symbols
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    freq = np.convolve(nrz, g, mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps
    return np.exp(1j * phase)


def gfsk_demod_bits(iq, sps, nbits, offset=0):
    """Simple discriminator for the self-test."""
    d = np.angle(iq[1:] * np.conj(iq[:-1]))
    centers = offset + np.arange(nbits) * sps + sps // 2
    return [1 if d[c - 1] > 0 else 0 for c in centers]


def build_burst(addr_onair, payload, sps, gap_samples, ramp=64):
    """4 packets (PID 0..3) with silence between, ready for a cyclic buffer."""
    chunks = []
    for pid in range(4):
        bits = build_esb_packet(addr_onair, payload, pid)
        iq = gfsk_modulate(bits, sps)
        # short amplitude ramps (on padding, not on data bits) to limit splatter
        r = np.hanning(2 * ramp)
        pre = np.full(ramp, iq[0])
        post = np.full(ramp, iq[-1])
        pkt = np.concatenate([pre, iq, post])
        w = np.concatenate([r[:ramp], np.ones(len(iq)), r[ramp:]])
        chunks.append(pkt * w)
        chunks.append(np.zeros(gap_samples, dtype=complex))
    return np.concatenate(chunks)


def selftest():
    addr = bytes(reversed(b"NODE1"))
    payload = b"hello"
    bits = build_esb_packet(addr, payload, pid=1)
    sps = 4
    iq = gfsk_modulate(bits, sps)
    rx = gfsk_demod_bits(iq, sps, len(bits))
    errors = sum(a != b for a, b in zip(bits, rx))
    # CRC over body + CRC must be 0 (residue check)
    body_and_crc = bits[8:]
    residue = crc16_ccitt_bits(body_and_crc)
    print(f"packet bits: {len(bits)}  demod bit errors: {errors}  CRC residue: 0x{residue:04X}")
    ok = errors == 0 and residue == 0
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="NODE1",
                    help="address as you'd pass it to RF24 openWritingPipe/openReadingPipe (3-5 chars)")
    ap.add_argument("--addr-hex", help="address as hex in over-the-air order (overrides --addr), e.g. E7E7E7E7E7")
    ap.add_argument("--channel", type=int, default=76, help="nRF24 channel 0-125 (freq = 2400 + ch MHz)")
    ap.add_argument("--rate", type=int, choices=[1, 2], default=1, help="data rate in Mbps (1 or 2)")
    ap.add_argument("--payload", default="hello", help="payload text (max 32 bytes)")
    ap.add_argument("--gain", type=float, default=-20, help="Pluto TX gain in dB (-89.75..0); start low")
    ap.add_argument("--gap-ms", type=float, default=50, help="silence between packets")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="Pluto URI (ip:192.168.2.1 or usb:...)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--selftest", action="store_true", help="offline packet/modulation check")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.addr_hex:
        addr_onair = bytes.fromhex(args.addr_hex)
    else:
        # RF24 writes the array LSByte-first into TX_ADDR; the radio sends MSByte first,
        # so the last character goes out first.
        addr_onair = bytes(reversed(args.addr.encode()))
    payload = args.payload.encode()

    rate = args.rate * 1_000_000
    sps = 4
    fs = rate * sps
    freq = 2_400_000_000 + args.channel * 1_000_000

    burst = build_burst(addr_onair, payload, sps, int(fs * args.gap_ms / 1000))

    import adi  # pyadi-iio
    sdr = adi.Pluto(args.uri)
    sdr.sample_rate = int(fs)
    sdr.tx_rf_bandwidth = int(fs)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = args.gain
    sdr.tx_cyclic_buffer = True

    print(f"TX {freq/1e6:.0f} MHz (ch {args.channel}), {args.rate} Mbps, "
          f"addr on-air {addr_onair.hex().upper()}, payload {payload!r}, gain {args.gain} dB")
    print("Repeating packets (PID 0..3). Ctrl-C to stop.")
    sdr.tx((burst * 2 ** 14).astype(np.complex64))
    try:
        if args.seconds > 0:
            time.sleep(args.seconds)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sdr.tx_destroy_buffer()
        print("stopped")


if __name__ == "__main__":
    main()
