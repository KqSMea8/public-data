/* 光敏二极管, 光照越强，电压越低
* doc : http://www.dfrobot.com.cn/community/thread-2505-1-1.html
上拉电阻的概念
*/

int LED = 8;                     //设置LED灯为数字引脚13
int val = 0;                      //设置模拟引脚0读取光敏二极管的电压值

void setup(){
      pinMode(LED,OUTPUT);         // LED为输出模式
      pinMode(0,INPUT);         // LED为输出模式
      Serial.begin(9600);        // 串口波特率设置为9600
}

void loop(){
      val = analogRead(0);         // 读取电压值0~1023
      Serial.println(val);         // 串口查看电压值的变化
      if(val<1000){                // 一旦小于设定的值，LED灯关闭
              digitalWrite(LED,LOW);
      }else{                        // 否则LED亮起
              digitalWrite(LED,HIGH);
      }
      delay(10);                   // 延时10ms
}