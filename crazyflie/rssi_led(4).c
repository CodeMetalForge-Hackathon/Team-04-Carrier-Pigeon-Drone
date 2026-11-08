/**
 * rssi_led.c - Onboard signal monitor + jam decision + action.
 *
 * Chain: Pluto ESB beacon -> radio.rssi onboard -> button arms -> signal lost
 * for TRIG_HOLD_MS -> ACTION.
 *
 * ACTION depends on hardware, checked at runtime:
 *   - Flow deck present  : arm, take off to FLY_HEIGHT_M, hover, fly straight
 *                          FLY_FORWARD_M, land. (Controlled flight.)
 *   - No positioning deck: PROPS-OFF motor spin only. It will NOT take off.
 *
 * ================== SAFETY ==================
 * With a Flow deck this FLIES. Clear area, stand back, props seated, charged
 * battery. Without a deck it only spins motors - run with PROPS OFF.
 * The button must be pressed (armed) for any action to happen.
 * ===========================================
 *
 * Button: Grove on RX2/PA3 (DECK_GPIO_RX2), pressed = HIGH, powered from 3.0V.
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
#include "crtp_commander_high_level.h"
#include "stabilizer_types.h"
#include "supervisor.h"
#include "param_logic.h"

#define DEBUG_MODULE "RSSILED"
#include "debug.h"

#define SIGNAL_LED       LED_GREEN_R
#define BUTTON_PIN       DECK_GPIO_RX2

#define RSSI_STRONG_DBM  (-60)
#define RSSI_WEAK_DBM    (-75)

#define TICK_MS          50
#define TRIG_HOLD_MS     500
#define DEBOUNCE_TICKS   2

/* ---- flight sequence (Flow deck only) ---- */
#define FLY_HEIGHT_M     0.4f
#define FLY_FORWARD_M    1.0f      /* "straight line" distance, +x (forward) */
#define TAKEOFF_S        2.0f
#define HOVER_S          1.5f
#define CRUISE_S         3.0f
#define LAND_S           2.0f

/* ---- motor test (no deck, PROPS OFF) ---- */
#define SPIN_THRUST      10000.0f
#define SPIN_MS          2000
#define UNLOCK_MS        200

typedef enum { ACT_NONE, ACT_SPIN, ACT_TAKEOFF, ACT_HOVER, ACT_CRUISE, ACT_LAND } act_t;

static bool flowDeckPresent(void) {
  paramVarId_t v2 = paramGetVarId("deck", "bcFlow2");
  paramVarId_t v1 = paramGetVarId("deck", "bcFlow");
  if (paramVarIdIsValid(v2) && paramGetInt(v2)) return true;
  if (paramVarIdIsValid(v1) && paramGetInt(v1)) return true;
  return false;
}

static void sendSpin(float thrust) {
  setpoint_t sp;
  memset(&sp, 0, sizeof(sp));
  sp.mode.roll = modeAbs;  sp.attitude.roll = 0.0f;
  sp.mode.pitch = modeAbs; sp.attitude.pitch = 0.0f;
  sp.mode.yaw = modeVelocity; sp.attitudeRate.yaw = 0.0f;
  sp.mode.z = modeDisable;
  sp.thrust = thrust;
  commanderSetSetpoint(&sp, COMMANDER_PRIORITY_EXTRX);
}

void appMain(void) {
  logVarId_t idRssi = logGetVarId("radio", "rssi");
  logVarId_t idConn = logGetVarId("radio", "isConnected");
  if (!logVarIdIsValid(idRssi)) { DEBUG_PRINT("radio.rssi not found\n"); return; }

  pinMode(BUTTON_PIN, INPUT);
  DEBUG_PRINT("RSSI monitor + jam action running\n");

  bool monitorOn = false, triggered = false;
  act_t act = ACT_NONE;
  uint32_t actMs = 0, weakTicks = 0, phase = 0, printDiv = 0;
  bool btnPrev = false, btnLatched = false;
  uint32_t btnStable = 0;

  while (1) {
    vTaskDelay(M2T(TICK_MS));
    phase++;

    bool connected = logVarIdIsValid(idConn) ? (logGetUint(idConn) != 0) : true;
    int rssi_dbm = -(int)logGetUint(idRssi);
    bool weak = (!connected) || (rssi_dbm < RSSI_WEAK_DBM);

    /* ---- button toggles monitorOn ---- */
    bool btnNow = (digitalRead(BUTTON_PIN) == 1);
    if (btnNow == btnPrev) { if (btnStable < 1000) btnStable++; }
    else { btnStable = 0; btnPrev = btnNow; }
    if (btnNow && btnStable == DEBOUNCE_TICKS && !btnLatched) {
      monitorOn = !monitorOn;
      if (!monitorOn) {                       /* abort any action, disarm */
        crtpCommanderHighLevelStop();
        sendSpin(0.0f);
        supervisorRequestArming(false);
        act = ACT_NONE; triggered = false;
      }
      weakTicks = 0; btnLatched = true;
      DEBUG_PRINT("%s\n", monitorOn ? "ARMED (monitoring)" : "disarmed");
    }
    if (!btnNow) btnLatched = false;

    /* ---- decision ---- */
    if (monitorOn && !triggered) {
      if (weak) {
        weakTicks += TICK_MS;
        if (weakTicks >= TRIG_HOLD_MS) {
          triggered = true;
          actMs = 0;
          supervisorRequestArming(true);
          if (flowDeckPresent()) {
            crtpCommanderHighLevelTakeoff(FLY_HEIGHT_M, TAKEOFF_S);
            act = ACT_TAKEOFF;
            DEBUG_PRINT("JAM -> FLIGHT: takeoff\n");
          } else {
            act = ACT_SPIN;
            DEBUG_PRINT("JAM -> motor spin (no deck, PROPS OFF)\n");
          }
        }
      } else {
        weakTicks = 0;
      }
    }

    /* ---- run the action state machine ---- */
    actMs += (act == ACT_NONE) ? 0 : TICK_MS;
    switch (act) {
      case ACT_SPIN:
        sendSpin(actMs < UNLOCK_MS ? 0.0f : SPIN_THRUST);
        if (actMs >= SPIN_MS) { sendSpin(0.0f); supervisorRequestArming(false);
                                act = ACT_NONE; DEBUG_PRINT("spin done\n"); }
        break;
      case ACT_TAKEOFF:
        if (actMs >= (uint32_t)(TAKEOFF_S * 1000)) { act = ACT_HOVER; actMs = 0; }
        break;
      case ACT_HOVER:
        if (actMs >= (uint32_t)(HOVER_S * 1000)) {
          crtpCommanderHighLevelGoTo(FLY_FORWARD_M, 0.0f, 0.0f, 0.0f, CRUISE_S, true);
          act = ACT_CRUISE; actMs = 0; DEBUG_PRINT("flying straight\n");
        }
        break;
      case ACT_CRUISE:
        if (actMs >= (uint32_t)(CRUISE_S * 1000)) {
          crtpCommanderHighLevelLand(0.0f, LAND_S);
          act = ACT_LAND; actMs = 0; DEBUG_PRINT("landing\n");
        }
        break;
      case ACT_LAND:
        if (actMs >= (uint32_t)(LAND_S * 1000)) {
          supervisorRequestArming(false); act = ACT_NONE; DEBUG_PRINT("landed\n");
        }
        break;
      default: break;
    }

    /* ---- LED ---- */
    bool ledOn;
    if (triggered)                        ledOn = (phase % 2) == 0;
    else if (!connected)                  ledOn = monitorOn ? ((phase % 10) < 1) : false;
    else if (rssi_dbm >= RSSI_STRONG_DBM) ledOn = true;
    else if (rssi_dbm >= RSSI_WEAK_DBM)   ledOn = (phase % 20) < 10;
    else                                  ledOn = (phase % 6) < 3;
    ledSet(SIGNAL_LED, ledOn);

    if (++printDiv >= (1000 / TICK_MS)) {
      printDiv = 0;
      DEBUG_PRINT("mon=%d trig=%d act=%d link=%d rssi=%d dBm\n",
                  monitorOn, triggered, (int)act, connected ? 1 : 0, rssi_dbm);
    }
  }
}
