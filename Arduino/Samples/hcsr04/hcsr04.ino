/*
* 声波测距模块
* HC-SR04,US-015 适用
* https://item.taobao.com/item.htm?id=43808564495 #HC-SR04
* https://item.taobao.com/item.htm?id=43795685137 #US-015

* https://item.taobao.com/item.htm?id=43736207670 #US-100
http://bbs.elecfans.com/forum.php?mod=viewthread&tid=1099658&extra=page=1
* US-100 后面有个跳线帽，带上跳线帽TX-RX - UART模式,不带跳线帽和US-015等相同
* 
*/

#define ECHOPIN 3
#define TRIGPIN 2

#define ALARMPIN 4
#define LEDPIN 13

//警报器
float sinVal;
int toneVal;

void setup()
{
  Serial.begin(9600);
  pinMode(ECHOPIN, INPUT); 
  pinMode(TRIGPIN, OUTPUT);  
  pinMode(ALARMPIN, OUTPUT);  
  pinMode(LEDPIN, OUTPUT);  
}

float getDistance(){
  digitalWrite(TRIGPIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGPIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGPIN, LOW);

  float distance = pulseIn(ECHOPIN, HIGH);
  distance = distance / 58;
   
  Serial.println(distance);
  
  return distance;
}

// alram 
void alarm()
{
    float sinVal;            
    int toneVal;

    for (int x = 0; x < 180; x++)
    {
        //将sin函数角度转化为弧度
        sinVal = (sin(x * (3.1412 / 180)));
        //用sin函数值产生声音的频率
        toneVal = 2000 + (int(sinVal * 1000));
        //给输出引脚一个
        tone(ALARMPIN, toneVal);
        delay(2);
    }
}
  
void loop()
{ 
  float distance = getDistance();
  
  delay(200);
  if (distance > 50) {
    digitalWrite(LEDPIN, LOW );  
    delay(100);
  } 
  else if(distance != 0) {
    digitalWrite(LEDPIN, HIGH);
    alarm();  
    delay(100);
  }else{
    //
  }
}
