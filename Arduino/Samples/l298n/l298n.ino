#include <IRremote.h>
//#include <IRremoteInt.h>

#define IN1 3
#define IN2 4
#define IN3 5
#define IN4 6
#define PWMA 9
#define PWMB 10

#define RECV_PIN 11

#define TRIGPIN 12
#define ECHOPIN 13

int pma = 80;
int pmb = 80;

IRrecv irrecv(RECV_PIN); //设置RECV_PIN（也就是11引脚）为红外接收端
decode_results results;  //定义results变量为红外结果存放位置

void setup()
{
  Serial.begin(9600);  //串口波特率设为9600
  irrecv.enableIRIn(); //启动红外解码

  //超声波测距 hc-sr04
  pinMode(ECHOPIN, INPUT);
  pinMode(TRIGPIN, OUTPUT);

  //电极驱动 l298n
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(PWMB, OUTPUT);
}

void stop()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void go(int pwa, int pwb)
{
  Serial.println(pwa);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  analogWrite(PWMA, pwa);
  analogWrite(PWMB, pwb);
}

void back()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(PWMA, 168);
  analogWrite(PWMB, 200);
}

void left()
{
  analogWrite(PWMA, 90);
  analogWrite(PWMB, 0);
  delay(60);
}

void right()
{
  analogWrite(PWMA, 0);
}

float getDistance()
{
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

int cmd_status = 0;

bool started = false; //状态

void loop()
{
  if (irrecv.decode(&results))
  {
    Serial.println(results.value); //
    if (results.value == 16580863)
    {
      started = !started;
    }

    Serial.println(started); //
    irrecv.resume();         // 继续等待接收下一组信号
  }

  if (!started)
  {
    stop();
    return;
  }

  float d = getDistance();
  if (d > 254.0)
  {
    pma = pmb = 120;
    go(pma, pmb);
    Serial.println(">254");
  }
  else if (d > 80)
  {
    pma = pmb = 80;
    go(pma, pmb);
    Serial.println(">80");
  }
  else if (d > 10)
  {
    pma = pmb = 60;
    go(pma, pmb);
    Serial.println(">10");
  }
  else
  {
    Serial.println("<10");

    back();
    delay(100);
    left();
    delay(100);
  }
}
