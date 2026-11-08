/**
 * rssi_led.c - Onboard signal-strength indicator + button-armed jam detector.
 *
 * SAFE VERSION: the "decision" only drives the LED (a fast strobe). No motor
 * commands. Use this to verify the whole chain on the bench:
 *     button press -> armed -> signal lost/weak -> LED strobes = "decision fired".
 * Flight actions (flee / land) are added later, once a Flow deck is confirmed.
 *
 * Signal source is the Nordic ESB link (radio.rssi / radio.isConnected), e.g.
 * the Pluto beacon. radio.rssi is -dBm as a positive number (smaller = stronger).
 *
 * Button: Grove button on RX2 / PA3 (DECK_GPIO_RX2). Grove modules output HIGH
 * when pressed. Power the Grove from the 3.0V pin, NOT 5V (PA3 is not 5V-tolerant).
 *
 * Indicator LED: right green (GREEN_R), otherwise idle.
 */
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "log.h"
#include "led.h"
#include "deck_digital.h"
#include "deck_constants.h"

#define DEBUG_MODULE "RSSILED"
#include "debug.h"

#define SIGNAL_LED       LED_GREEN_R
#define BUTTON_PIN       DECK_GPIO_RX2     /* PA3, Grove signal (pressed = HIGH) */

/* RSSI thresholds in dBm (positive stored value = -dBm). */
#define RSSI_STRONG_DBM  (-60)
#define RSSI_WEAK_DBM    (-75)

#define TICK_MS          50                /* main loop period */
#define TRIG_HOLD_MS     500               /* weak/lost must persist this long to fire */
#define DEBOUNCE_TICKS   2                 /* button stable for N ticks */

void appMain(void) {
  logVarId_t idRssi = logGetVarId("radio", "rssi");
  logVarId_t idConn = logGetVarId("radio", "isConnected");

  if (!logVarIdIsValid(idRssi)) {
    DEBUG_PRINT("radio.rssi not found -- aborting\n");
    return;
  }

  pinMode(BUTTON_PIN, INPUT);              /* Grove module has its own pull-down */

  DEBUG_PRINT("RSSI monitor + button(RX2) running\n");

  bool armed = false;
  bool triggered = false;
  uint32_t phase = 0;
  uint32_t weakTicks = 0;                  /* how long signal has been weak/lost */
  uint32_t btnStable = 0;
  bool btnPrev = false;
  uint32_t printDiv = 0;

  while (1) {
    vTaskDelay(M2T(TICK_MS));
    phase++;

    /* ---- read signal ---- */
    bool connected = logVarIdIsValid(idConn) ? (logGetUint(idConn) != 0) : true;
    int rssi_dbm = -(int)logGetUint(idRssi);
    bool weak = (!connected) || (rssi_dbm < RSSI_WEAK_DBM);

    /* ---- read button (pressed = HIGH), with simple debounce ---- */
    bool btnNow = (digitalRead(BUTTON_PIN) == 1);
    if (btnNow == btnPrev) {
      if (btnStable < 1000) btnStable++;
    } else {
      btnStable = 0;
      btnPrev = btnNow;
    }
    /* rising edge, debounced -> toggle armed, clear any trigger */
    static bool btnLatched = false;
    if (btnNow && btnStable == DEBOUNCE_TICKS && !btnLatched) {
      armed = !armed;
      triggered = false;
      weakTicks = 0;
      btnLatched = true;
      DEBUG_PRINT("%s\n", armed ? "ARMED" : "disarmed");
    }
    if (!btnNow) btnLatched = false;

    /* ---- decision ---- */
    if (armed && !triggered) {
      if (weak) {
        weakTicks += TICK_MS;
        if (weakTicks >= TRIG_HOLD_MS) {
          triggered = true;
          DEBUG_PRINT("JAM/LOSS DETECTED -> decision fired (rssi=%d conn=%d)\n",
                      rssi_dbm, connected ? 1 : 0);
          /* TODO: flight action goes here once Flow deck confirmed */
        }
      } else {
        weakTicks = 0;                     /* signal recovered, reset timer */
      }
    }

    /* ---- LED ---- */
    bool ledOn;
    if (triggered) {
      ledOn = (phase % 2) == 0;            /* triggered: very fast strobe */
    } else if (!connected) {
      ledOn = armed ? ((phase % 10) < 1) : false;  /* armed+nolink: brief wink */
    } else if (rssi_dbm >= RSSI_STRONG_DBM) {
      ledOn = true;                        /* strong: solid */
    } else if (rssi_dbm >= RSSI_WEAK_DBM) {
      ledOn = (phase % 20) < 10;           /* medium: slow blink */
    } else {
      ledOn = (phase % 6) < 3;             /* weak: fast blink */
    }
    ledSet(SIGNAL_LED, ledOn);

    if (++printDiv >= (1000 / TICK_MS)) {
      printDiv = 0;
      DEBUG_PRINT("armed=%d trig=%d link=%d rssi=%d dBm\n",
                  armed ? 1 : 0, triggered ? 1 : 0, connected ? 1 : 0, rssi_dbm);
    }
  }
}
