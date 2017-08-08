/*
  硬件: https://detail.tmall.com/item.htm?id=520019625635
  doc : http://www.dfrobot.com.cn/community/thread-2535-1-1.html
*/
int buttonPin = 2;         // button连接到数字2
int buttonState;           // 记录button当前状态值
int lastButtonState = LOW; // 记录button前一个状态值
long lastDebounceTime = 0;
long debounceDelay = 50; //去除抖动时间

int relayPin = 3;     // 继电器连接到数字3
int relayState = LOW; // 继电器初始状态为HIGH

int i = 30;

void setup() {
  pinMode(buttonPin, INPUT);
  pinMode(relayPin, OUTPUT);

  digitalWrite(relayPin, relayState); // 设置继电器的初始状态
  Serial.begin(9600);
}

void loop()
{
  int reading = digitalRead(buttonPin); //reading用来存储buttonPin的数据

  // 一旦检测到数据发生变化，记录当前时间
  if (reading != lastButtonState)
  {
    lastDebounceTime = millis();
  }

  // 等待50ms，再进行一次判断，是否和当前button状态相同
  // 如果和当前状态不相同，改变button状态
  // 同时，如果button状态为高（也就是被按下），那么就改变继电器的状态
  if ((millis() - lastDebounceTime) > debounceDelay)
  {
    if (reading != buttonState)
    {
      buttonState = reading;

      if (buttonState == HIGH)
      {
        relayState = !relayState;
      }

      //test for analogWrite
      i = i + 10;
      if (i >= 255)
      {
        i = 10;
      }
      Serial.println(i);
      //120以上，继电器会发出鸣叫声
    }
  }

  //  analogWrite(relayPin,i);
  digitalWrite(relayPin, relayState);

  // 改变button前一个状态值
  lastButtonState = reading;
}
