/*
 * 霍尔测速传感器测试
 * 传感器： https://item.taobao.com/item.htm?id=43734867714
 * 主要芯片：LM393、3144霍尔传感器
 * 问题： 原理还不清楚，不知道具体该如何测速
*/

#define PIN_LED 12
#define PIN_D0 2
#define PIN_A0 0

void setup()
{
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_D0, INPUT);
  pinMode(PIN_A0, INPUT); //A0无效？
  Serial.begin(9600);
}

void loop()
{
  int d = digitalRead(PIN_D0); 

  if(d==LOW){
    digitalWrite(PIN_LED,HIGH);
  }else{
    digitalWrite(PIN_LED,LOW);
  }

  // int a = analogRead(PIN_A0);

  // Serial.print("a:");
  // Serial.println(a);

  // Serial.print("d:");
  // Serial.println(d);
  // Serial.println("");
}
