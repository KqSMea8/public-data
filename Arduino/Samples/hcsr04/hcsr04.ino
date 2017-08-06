/*
  声波测距模块
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

void alarm(){
  for(int x=0; x<180; x++){
    //将sin函数角度转化为弧度
    sinVal = (sin(x*(3.1412/180)));
    //用sin函数值产生声音的频率
    toneVal = 2000+(int(sinVal*1600));
    
    //蜂鸣器;'
    //]OU MHadfbn
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
