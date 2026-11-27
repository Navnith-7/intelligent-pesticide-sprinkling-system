/*
 * Intelligent Pesticide Sprinkling System - Arduino Uno Relay Controller
 * Serial Protocol: 9600 Baud, 8N1
 * Commands:
 *   '1' -> Activate Relay (Water Pump ON)
 *   '0' -> Deactivate Relay (Water Pump OFF - Fail-Safe Default)
 *
 * Hardware watchdog: if the relay stays ON longer than MAX_ON_MS without a '0'
 * (host crash, USB disconnect mid-pulse), the sketch switches it OFF by itself.
 * A repeated '1' while the relay is already ON does NOT extend the timer.
 */

const int RELAY_PIN = 7;       // Digital pin connected to Relay IN1
const bool ACTIVE_LOW = true;  // Most 2-channel opto-isolated relays trigger on LOW

const unsigned long MAX_ON_MS = 3000UL;  // Maximum continuous ON time (watchdog)

bool relayOn = false;
unsigned long relayOnSince = 0;

void setRelay(bool turnOn) {
  if (ACTIVE_LOW) {
    digitalWrite(RELAY_PIN, turnOn ? LOW : HIGH);
  } else {
    digitalWrite(RELAY_PIN, turnOn ? HIGH : LOW);
  }

  if (turnOn && !relayOn) {
    relayOnSince = millis();  // start of a new ON period
  }
  relayOn = turnOn;
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

  // Non-blocking watchdog (unsigned subtraction is safe across millis() rollover).
  if (relayOn && (millis() - relayOnSince >= MAX_ON_MS)) {
    setRelay(false);
  }
}
