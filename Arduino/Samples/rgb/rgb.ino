/*彩光led*/
#define PIN_R  9 
#define PIN_G  10 
#define PIN_B  11

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin 13 as an output.
  pinMode(PIN_R, OUTPUT);
  pinMode(PIN_G, OUTPUT);
  pinMode(PIN_B, OUTPUT);
}

// the loop function runs over and over again forever
void loop() {
  int i=0;
  int j=0;
  int k=0;
  
  for(int i=0;i<255;i++){
    analogWrite(PIN_R,i);
    analogWrite(PIN_G,i);
    analogWrite(PIN_B,255-i); 
  
    delay(10);   // wait for a second
  } 
  
  delay(200);              // wait for a second
}
