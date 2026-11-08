/**
 * rssi_led.c - Onboard signal monitor + button-armed jam detector + MOTOR TEST.
 *
 * ================== REMOVE THE PROPELLERS BEFORE RUNNING ==================
 * When the decision fires, the motors spin up at low power for a couple of
 * seconds so you can SEE the drone react. There is NO positioning deck, so
 * this deliberately does NOT take off - it only spins the motors in place.
 * Run it on a bench with the props OFF and the drone held/weighted down.
 * =========================================================================
 *
 * Chain: Pluto ESB beacon -> radio.rssi onboard -> button arms -> signal lost
 * for TRIG_HOLD_MS -> arm motors + low-thrust spin for SPIN_MS -> disarm.
 *
 * Button: Grove on RX2/PA3 (DECK_GPIO_RX2), pressed = HIGH, powered from 3.0V.
 * LED: right green (GREEN_R) shows strength / strobes when triggered.
 */
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "log.h"
#include "led.h"
#include "deck_digital.h"
#include "deck_constants.h"
#include "commander.h"
#include "stabilizer_types.h"
#include "supervisor.h"

#define DEBUG_MODULE "RSSILED"
#include "debug.h"

#define SIGNAL_LED       LED_GREEN_R
#define BUTTON_PIN       DECK_GPIO_RX2     /* PA3, Grove signal, pressed = HIGH */

#define RSSI_STRONG_DBM  (-60)
#define RSSI_WEAK_DBM    (-75)

#define TICK_MS          50
#define TRIG_HOLD_MS     500               /* weak/lost must persist to fire */
#define DEBOUNCE_TICKS   2

/* ---- motor test (PROPS OFF) ---- */
#define SPIN_THRUST      10000.0f          /* of 65535 (~15%). Keep low. */
#define SPIN_MS          2000              /* how long motors spin */
#define UNLOCK_MS        200               /* send thrust=0 first to clear lock */

static void sendSpin(float thrust) {
  setpoint_t sp;
  memset(&sp, 0, sizeof(sp));
  sp.mode.roll  = modeAbs;      sp.attitude.roll  = 0.0f;
  sp.mode.pitch = modeAbs;      sp.attitude.pitch = 0.0f;
  sp.mode.yaw   = modeVelocity; sp.attitudeRate.yaw = 0.0f;
  sp.mode.z     = modeDisable;
  sp.thrust     = thrust;
  commanderSetSetpoint(&sp, COMMANDER_PRIORITY_EXTRX);   /* above the pilot link */
}

void appMain(void) {
  logVarId_t idRssi = logGetVarId("radio", "rssi");
  logVarId_t idConn = logGetVarId("radio", "isConnected");
  if (!logVarIdIsValid(idRssi)) {
    DEBUG_PRINT("radio.rssi not found -- aborting\n");
    return;
  }

  pinMode(BUTTON_PIN, INPUT);              /* Grove has its own pull-down */
  DEBUG_PRINT("RSSI + button + MOTOR TEST (PROPS OFF!) running\n");

  bool monitorOn = false;                  /* set by the button */
  bool triggered = false;
  bool testActive = false;
  uint32_t testElapsed = 0;
  uint32_t weakTicks = 0;
  uint32_t phase = 0, printDiv = 0;
  bool btnPrev = false, btnLatched = false;
  uint32_t btnStable = 0;

  while (1) {
    vTaskDelay(M2T(TICK_MS));
    phase++;

    bool connected = logVarIdIsValid(idConn) ? (logGetUint(idConn) != 0) : true;
    int rssi_dbm = -(int)logGetUint(idRssi);
    bool weak = (!connected) || (rssi_dbm < RSSI_WEAK_DBM);

    /* ---- button: debounced rising edge toggles monitorOn ---- */
    bool btnNow = (digitalRead(BUTTON_PIN) == 1);
    if (btnNow == btnPrev) { if (btnStable < 1000) btnStable++; }
    else { btnStable = 0; btnPrev = btnNow; }
    if (btnNow && btnStable == DEBOUNCE_TICKS && !btnLatched) {
      monitorOn = !monitorOn;
      triggered = false; weakTicks = 0;
      if (!monitorOn && testActive) {      /* disarm if turned off mid-test */
        sendSpin(0.0f); supervisorRequestArming(false); testActive = false;
      }
      btnLatched = true;
      DEBUG_PRINT("%s\n", monitorOn ? "ARMED (monitoring)" : "disarmed");
    }
    if (!btnNow) btnLatched = false;

    /* ---- decision ---- */
    if (monitorOn && !triggered) {
      if (weak) {
        weakTicks += TICK_MS;
        if (weakTicks >= TRIG_HOLD_MS) {
          triggered = true;
          testActive = true;
          testElapsed = 0;
          supervisorRequestArming(true);   /* PROPS OFF! */
          DEBUG_PRINT("JAM/LOSS -> MOTOR TEST (props off) rssi=%d conn=%d\n",
                      rssi_dbm, connected ? 1 : 0);
        }
      } else {
        weakTicks = 0;
      }
    }

    /* ---- run the motor test ---- */
    if (testActive) {
      sendSpin(testElapsed < UNLOCK_MS ? 0.0f : SPIN_THRUST);
      testElapsed += TICK_MS;
      if (testElapsed >= SPIN_MS) {
        sendSpin(0.0f);
        supervisorRequestArming(false);
        testActive = false;
        DEBUG_PRINT("motor test done\n");
      }
    }

    /* ---- LED ---- */
    bool ledOn;
    if (triggered)            ledOn = (phase % 2) == 0;          /* strobe */
    else if (!connected)      ledOn = monitorOn ? ((phase % 10) < 1) : false;
    else if (rssi_dbm >= RSSI_STRONG_DBM) ledOn = true;
    else if (rssi_dbm >= RSSI_WEAK_DBM)   ledOn = (phase % 20) < 10;
    else                      ledOn = (phase % 6) < 3;
    ledSet(SIGNAL_LED, ledOn);

    if (++printDiv >= (1000 / TICK_MS)) {
      printDiv = 0;
      DEBUG_PRINT("mon=%d trig=%d test=%d link=%d rssi=%d dBm\n",
                  monitorOn, triggered, testActive, connected ? 1 : 0, rssi_dbm);
    }
  }
}
