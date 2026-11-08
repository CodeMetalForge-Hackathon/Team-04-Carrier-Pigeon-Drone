// ============================================================
// LilyGo T3 V1.6.1 — Two-way LoRa Chat
// Same sketch on BOTH boards — set a different name below
// Board:   Tools > Board > esp32 > "TTGO LoRa32-OLED"
// Library: "LoRa" by Sandeep Mistry
//          "Adafruit SSD1306" by Adafruit
//
// Usage:
//   Type a message in Serial Monitor (115200 baud, Newline ending)
//   Press Enter — it sends over LoRa and shows on the other board
// ============================================================

#include <SPI.h>
#include <LoRa.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---- Set a different name on each board ----
#define MY_NAME   "BOARD_1"   // change to "BOARD_2" on the other one

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
#define RF_FREQ   915E6
#define TX_POWER  20
#define SF        7
#define BW        125E3
#define CR        5

#define PIN_LED   25
#define MAX_MSG   200   // max message length

Adafruit_SSD1306 display(128, 64, &Wire, -1);

String incomingMsg  = "";
String lastSent     = "";
String lastReceived = "";
int    lastRSSI     = 0;
float  lastSNR      = 0;

// ---- Display layout ----
void updateDisplay() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Header
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("[ %s ] 915MHz", MY_NAME);
  display.drawLine(0, 9, 127, 9, SSD1306_WHITE);

  // Last sent
  display.setCursor(0, 12);
  display.print("TX: ");
  display.println(lastSent.substring(0, 18));   // truncate to fit

  // Last received
  display.setCursor(0, 24);
  display.print("RX: ");
  display.println(lastReceived.substring(0, 18));

  // RSSI / SNR
  display.setCursor(0, 36);
  display.printf("RSSI: %d dBm  SNR: %.1f", lastRSSI, lastSNR);

  // Prompt
  display.drawLine(0, 46, 127, 46, SSD1306_WHITE);
  display.setCursor(0, 50);
  display.print("> ");
  display.print(incomingMsg.substring(0, 18));

  display.display();
}

void sendMessage(String msg) {
  if (msg.length() == 0) return;

  String packet = String(MY_NAME) + ": " + msg;

  LoRa.beginPacket();
  LoRa.print(packet);
  LoRa.endPacket();

  lastSent = msg;
  Serial.println("Sent: " + packet);

  digitalWrite(PIN_LED, HIGH); delay(80); digitalWrite(PIN_LED, LOW);
  updateDisplay();
}

void setup() {
  pinMode(PIN_LED, OUTPUT);
  Serial.begin(115200);
  delay(200);

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
  }

  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);
  LoRa.setPins(LORA_CS, LORA_RST, LORA_IRQ);

  if (!LoRa.begin(RF_FREQ)) {
    Serial.println("LoRa init FAILED");
    display.clearDisplay();
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

  Serial.printf("\n[ %s ] ready — type a message and press Enter\n", MY_NAME);
  updateDisplay();
}

void loop() {
  // ---- Check for incoming LoRa packet ----
  int pktSize = LoRa.parsePacket();
  if (pktSize) {
    String rx = "";
    while (LoRa.available()) rx += (char)LoRa.read();

    lastRSSI     = LoRa.packetRssi();
    lastSNR      = LoRa.packetSnr();
    lastReceived = rx;

    Serial.printf("Received: \"%s\"  RSSI: %d dBm  SNR: %.1f dB\n",
                  rx.c_str(), lastRSSI, lastSNR);

    // Print in a format Python can parse
    Serial.printf("CHAT_RX:%s|RSSI:%d|SNR:%.1f\n", rx.c_str(), lastRSSI, lastSNR);

    digitalWrite(PIN_LED, HIGH); delay(40); digitalWrite(PIN_LED, LOW);
    delay(40);
    digitalWrite(PIN_LED, HIGH); delay(40); digitalWrite(PIN_LED, LOW);

    updateDisplay();
  }

  // ---- Check for typed message from Serial / Python ----
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (incomingMsg.length() > 0) {
        sendMessage(incomingMsg);
        incomingMsg = "";
      }
    } else {
      if (incomingMsg.length() < MAX_MSG) {
        incomingMsg += c;
        updateDisplay();   // show typed characters live on OLED
      }
    }
  }
}
