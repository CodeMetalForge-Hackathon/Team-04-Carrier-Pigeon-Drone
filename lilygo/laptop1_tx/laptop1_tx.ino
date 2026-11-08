// ============================================================
// LAPTOP 1 — LilyGo T3 V1.6.1 — TRANSMITTER
// Board:   Tools > Board > esp32 > "TTGO LoRa32-OLED"
// Library: "LoRa" by Sandeep Mistry
//          "Adafruit SSD1306" by Adafruit
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

// ---- Radio settings ----
#define RF_FREQ   915E6   // 915 MHz (US)
#define TX_POWER  20      // dBm
#define SF        7       // Spreading factor (7 = fastest)
#define BW        125E3   // Bandwidth
#define CR        5       // Coding rate 4/5

#define PIN_LED   25
#define TX_DELAY  200     // ms between packets (~5 per second)

Adafruit_SSD1306 display(128, 64, &Wire, -1);
uint32_t packetNum = 0;

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

  LoRa.setTxPower(TX_POWER);
  LoRa.setSpreadingFactor(SF);
  LoRa.setSignalBandwidth(BW);
  LoRa.setCodingRate4(CR);

  Serial.println("TX ready — 915 MHz SF7");
}

void loop() {
  char msg[32];
  snprintf(msg, sizeof(msg), "PKT:%lu", packetNum++);

  LoRa.beginPacket();
  LoRa.print(msg);
  LoRa.endPacket();

  Serial.printf("TX: %s\n", msg);
  digitalWrite(PIN_LED, HIGH); delay(50); digitalWrite(PIN_LED, LOW);

  // OLED
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(0, 0);  display.println("TX MODE");
  display.setTextSize(1);
  display.setCursor(0, 25); display.printf("Freq:  915 MHz");
  display.setCursor(0, 36); display.printf("Power: %d dBm", TX_POWER);
  display.setCursor(0, 47); display.printf("Sent:  %lu pkts", packetNum);
  display.display();

  delay(TX_DELAY);
}
