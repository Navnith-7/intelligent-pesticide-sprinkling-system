/*
 * Intelligent Pesticide Sprinkling System - Arduino Uno Relay Controller
 * Serial Protocol: 9600 Baud, 8N1
 * Commands:
 *   '1' -> Activate Relay (Water Pump ON)
 *   '0' -> Deactivate Relay (Water Pump OFF - Fail-Safe Default)
 */

const int RELAY_PIN = 7;       // Digital pin connected to Relay IN1
const bool ACTIVE_LOW = true;  // Most 2-channel opto-isolated relays trigger on LOW

void setRelay(bool turnOn) {
  if (ACTIVE_LOW) {
    digitalWrite(RELAY_PIN, turnOn ? LOW : HIGH);
  } else {
    digitalWrite(RELAY_PIN, turnOn ? HIGH : LOW);
  }
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  setRelay(false); // Fail-safe: force pump OFF immediately on boot/reset
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      setRelay(true);
    } else if (cmd == '0') {
      setRelay(false);
    }
  }
}
