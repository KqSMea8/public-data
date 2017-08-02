/*
 * us-016 
*/

unsigned int ADCValue;

void setup(){
    Serial.begin(9600);
}

void loop(){

    ADCValue = analogRead(0);
    ADCValue *= 3; //range为空或者高电平   

    Serial.print(ADCValue, DEC);
    Serial.println("mm");
    delay(1000);//delay 1S
}
