// ============================================================
// LAPTOP 2 — LilyGo T3 V1.6.1 — RECEIVER
// Board:   Tools > Board > esp32 > "TTGO LoRa32-OLED"
// Library: "LoRa" by Sandeep Mistry
//          "Adafruit SSD1306" by Adafruit
//
// Streams CSV to laptop over serial for rssi_plot.py
// Format: time_ms,rssi,snr,loss_pct
// ============================================================

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---- LoRa pins (T3 V1.6.1) ----
#define LORA_SCK  5
#define LORA_MISO 19
#define LORA_MOSI 27
#define LORA_CS   18
#define LORA_RST  23
#define LORA_IRQ  26

// ---- OLED ----
#define OLED_SDA  21
#define OLED_SCL  22

// ---- Radio settings (must match TX) ----
#define RF_FREQ   915E6
#define SF        7
#define BW        125E3
#define CR        5

#define PIN_LED   25

Adafruit_SSD1306 display(128, 64, &Wire, -1);

uint32_t totalPkts  = 0;
uint32_t lostPkts   = 0;
int32_t  lastPktNum = -1;
float    rssiAvg    = 0;

void setup() {
  pinMode(PIN_LED, OUTPUT);
  Serial.begin(115200);
  delay(200);

  // OLED
  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // LoRa
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);
  LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);

  if (!LoRa.begin(RF_FREQ)) {
    Serial.println("LoRa init FAILED");
    display.setTextSize(2);
    display.setCursor(0, 0); display.println("LoRa");
    display.setCursor(0, 20); display.println("FAILED");
    display.display();
    while (1) { digitalWrite(PIN_LED, !digitalRead(PIN_LED)); delay(100); }
  }

  LoRa.setSpreadingFactor(SF);
  LoRa.setSignalBandwidth(BW);
  LoRa.setCodingRate4(CR);

  // CSV header — rssi_plot.py skips this line
  Serial.println("time_ms,rssi,snr,loss_pct");
  Serial.println("RX ready — 915 MHz SF7");

  display.setTextSize(2);
  display.setCursor(0, 0); display.println("RX MODE");
  display.setTextSize(1);
  display.setCursor(0, 25); display.println("Waiting for TX...");
  display.display();
}

void loop() {
  int pktSize = LoRa.parsePacket();
  if (!pktSize) return;

  // Read packet
  String rx = "";
  while (LoRa.available()) rx += (char)LoRa.read();

  int   rssi = LoRa.packetRssi();
  float snr  = LoRa.packetSnr();
  totalPkts++;

  // Sequence-based packet loss detection
  uint32_t pktNum = 0;
  if (rx.startsWith("PKT:")) pktNum = rx.substring(4).toInt();
  if (lastPktNum >= 0 && pktNum > (uint32_t)(lastPktNum + 1))
    lostPkts += pktNum - lastPktNum - 1;
  lastPktNum = pktNum;

  float lossPct = (float)lostPkts / (totalPkts + lostPkts) * 100.0;

  // Rolling average RSSI
  rssiAvg = (rssiAvg == 0) ? rssi : (rssiAvg * 0.9 + rssi * 0.1);

  // Stream CSV to laptop
  Serial.printf("%lu,%d,%.1f,%.1f\n", millis(), rssi, snr, lossPct);

  // OLED
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);  display.printf("RSSI: %d dBm", rssi);
  display.setCursor(0, 12); display.printf("SNR:  %.1f dB", snr);
  display.setCursor(0, 24); display.printf("Avg:  %.0f dBm", rssiAvg);
  display.setCursor(0, 36); display.printf("Loss: %.1f%%", lossPct);
  display.setCursor(0, 48); display.printf("Pkts: %lu rx / %lu lost", totalPkts, lostPkts);
  display.display();

  digitalWrite(PIN_LED, HIGH); delay(20); digitalWrite(PIN_LED, LOW);
}
