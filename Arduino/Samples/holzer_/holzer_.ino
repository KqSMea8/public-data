/*
 * 霍尔测速传感器测试
 * 传感器： https://item.taobao.com/item.htm?id=43734867714
 * 主要芯片：LM393、3144霍尔传感器
 
 * 参考文章：
 * http://ardui.co/archives/380
 * http://www.chuangkoo.com/project/69
 
 * 问题： 原理还不清楚，不知道具体该如何测速
*/

#define PIN_D0 2
#define PIN_A0 0

void setup()
{
  // put your setup code here, to run once:
  pinMode(PIN_D0, INPUT);
  Serial.begin(9600);
}

void loop()
{
  int d = digitalRead(PIN_D0);
  int a = analogRead(PIN_A0);

  Serial.print("a:");
  Serial.println(a);

  Serial.print("d:");
  Serial.println(d);
  Serial.println("");
}
