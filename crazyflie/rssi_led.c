/**
 * rssi_led.c - Onboard Nordic (ESB) signal-strength indicator.
 *
 * Crazyflie app-layer module. Runs on the STM32, reads the radio link RSSI
 * (radio.rssi) and link state (radio.isConnected) from the onboard log
 * subsystem, and shows the strength on the right-hand GREEN LED:
 *
 *   no link      -> LED off
 *   strong       -> solid on
 *   medium       -> slow blink
 *   weak         -> fast blink
 *
 * radio.rssi is the Crazyradio <-> Crazyflie ESB link strength, stored as a
 * positive number of -dBm (60 == -60 dBm; smaller magnitude = stronger). It
 * only updates while ESB packets arrive, which is why we gate on isConnected.
 *
 * The left green/red LEDs are driven by the firmware's own link sequences, so
 * this uses GREEN_R (USER_NOTF_LED), which is otherwise idle.
 */
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "log.h"
#include "led.h"

#define DEBUG_MODULE "RSSILED"
#include "debug.h"

#define SIGNAL_LED       LED_GREEN_R   /* otherwise-idle right green LED */

/* Thresholds in dBm. Tune for your range/environment. */
#define RSSI_STRONG_DBM  (-60)         /* >= this: solid on            */
#define RSSI_WEAK_DBM    (-75)         /* >= this: medium; below: weak */

#define TICK_MS          50            /* LED update period            */

void appMain(void) {
  logVarId_t idRssi = logGetVarId("radio", "rssi");
  logVarId_t idConn = logGetVarId("radio", "isConnected");

  if (!logVarIdIsValid(idRssi)) {
    DEBUG_PRINT("radio.rssi log var not found -- aborting indicator\n");
    return;
  }

  DEBUG_PRINT("RSSI LED indicator running on GREEN_R\n");

  uint32_t phase = 0;      /* blink phase, counted in ticks */
  uint32_t printDiv = 0;

  while (1) {
    vTaskDelay(M2T(TICK_MS));
    phase++;

    bool connected = logVarIdIsValid(idConn) ? (logGetUint(idConn) != 0) : true;
    int rssi_dbm = -(int)logGetUint(idRssi);

    bool ledOn;
    if (!connected) {
      ledOn = false;                         /* no ESB link */
    } else if (rssi_dbm >= RSSI_STRONG_DBM) {
      ledOn = true;                          /* strong: solid */
    } else if (rssi_dbm >= RSSI_WEAK_DBM) {
      ledOn = (phase % 20) < 10;             /* medium: ~1 Hz blink */
    } else {
      ledOn = (phase % 6) < 3;               /* weak: ~3 Hz blink */
    }

    ledSet(SIGNAL_LED, ledOn);

    if (++printDiv >= (1000 / TICK_MS)) {    /* ~1 Hz console print */
      printDiv = 0;
      DEBUG_PRINT("link=%d rssi=%d dBm\n", connected ? 1 : 0, rssi_dbm);
    }
  }
}
