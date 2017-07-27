

#define ECHOPIN 9

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
  pinMode(ECHOPIN, INPUT);
  

}

void loop() {
  // put your main code here, to run repeatedly:
  int d = digitalRead(ECHOPIN);
  Serial.println(d);
  delay(1000);

}
