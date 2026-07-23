// Pulse generator for hardware-clocked SLM sequence playback.
//
// Drives the SLM board's external trigger input with a fixed-period pulse train,
// so a preloaded sequence advances on the Arduino's crystal rather than on host
// software timing. Host side is pytweezer/arduino.py (ArduinoPulser).
//
// Protocol, newline-terminated in both directions:
//
//   "<n>"               fire n pulses at the current period
//   "<n>,<period_us>"   set the period, then fire n pulses
//   "q"                 halt; ignores everything until reset


#include <TimerOne.h>

const int triggerPin = 9;
const int ledPin = LED_BUILTIN;

const unsigned long DEFAULT_PERIOD_US = 1000;
const unsigned long MIN_PERIOD_US = 700;
const unsigned long MAX_PERIOD_US = 1000000;
const unsigned int PULSE_WIDTH_US = 15;

const long blinkInterval = 500;

unsigned long periodUs = DEFAULT_PERIOD_US;
unsigned long previousMillis = 0;
long trainRequested = 0;

volatile long pulsesRemaining = 0;
volatile bool isPulsing = false;
volatile bool reportPending = false;
volatile bool haveFirstPulse = false;
volatile unsigned long firstPulseUs = 0;
volatile unsigned long lastPulseUs = 0;

// Runs off Timer1 at periodUs. Kept short: two digitalWrites and the pulse
// width, ~25 us total against a >=700 us period.
void isrFirePulse() {
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(PULSE_WIDTH_US);
  digitalWrite(triggerPin, LOW);

  unsigned long now = micros();
  if (!haveFirstPulse) {
    firstPulseUs = now;
    haveFirstPulse = true;
  }
  lastPulseUs = now;

  if (--pulsesRemaining <= 0) {
    Timer1.stop();
    isPulsing = false;
    reportPending = true;
  }
}

void startTrain(long n) {
  trainRequested = n;
  pulsesRemaining = n;
  haveFirstPulse = false;
  isPulsing = true;
  Timer1.initialize(periodUs);
  Timer1.attachInterrupt(isrFirePulse);
}

void handleCommand(const String &input) {
  if (input == "q") {
    Timer1.stop();
    digitalWrite(triggerPin, LOW);
    digitalWrite(ledPin, LOW);
    isPulsing = false;
    Serial.println("OK halted");
    Serial.flush();
    while (true) {}
  }

  if (isPulsing) {
    Serial.println("ERR busy");
    return;
  }

  long n;
  unsigned long p = periodUs;
  int comma = input.indexOf(',');
  if (comma >= 0) {
    n = input.substring(0, comma).toInt();
    p = (unsigned long)input.substring(comma + 1).toInt();
  } else {
    n = input.toInt();
  }

  if (n <= 0) {
    Serial.println("ERR bad count");
  } else if (p < MIN_PERIOD_US || p > MAX_PERIOD_US) {
    Serial.print("ERR period out of range ");
    Serial.print(MIN_PERIOD_US);
    Serial.print('-');
    Serial.println(MAX_PERIOD_US);
  } else {
    periodUs = p;
    startTrain(n);   // nothing is printed until the train finishes
  }
}

void setup() {
  pinMode(triggerPin, OUTPUT);
  digitalWrite(triggerPin, LOW);
  pinMode(ledPin, OUTPUT);
  Serial.begin(250000);
  Serial.setTimeout(20);   
  Serial.println("Listening for Python...");
}

void loop() {
  // Slow blink while idle, solid while a train is in flight.
  if (isPulsing) {
    digitalWrite(ledPin, HIGH);
  } else {
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= blinkInterval) {
      previousMillis = currentMillis;
      digitalWrite(ledPin, !digitalRead(ledPin));
    }
  }

  // The train has ended; safe to read the ISR's timestamps.
  if (reportPending) {
    reportPending = false;
    Serial.print("OK ");
    Serial.print(trainRequested);
    Serial.print(' ');
    Serial.println(lastPulseUs - firstPulseUs);
  }

  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      handleCommand(input);
    }
  }
}
