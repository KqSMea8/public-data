/* 振动传感器：sw200d
* taobao: https://detail.tmall.com/item.htm?id=552701970373
硬件：一个pin是金色的，一个是普通色的，滚珠默认在白色pin位置？
* doc:  http://www.dfrobot.com.cn/community/thread-2504-1-1.html
* 主要： attachInterrupt 函数
https://www.arduino.cc/en/Reference/AttachInterrupt
interrupt：中断号0或者1。如果选择0的话，连接到数字引脚2上，选择1的话，连接到数字引脚3上。
*/

#define LED_PIN 8 //定义LED为数字引脚8
#define SW_PIN 3  //连接震动开关到中断1，也就是数字引脚3  

unsigned char state = 0;

void setup() { 
  pinMode(LED_PIN, OUTPUT);         //LED为输出模式
  pinMode(SW_PIN, INPUT);        //震动开关为输入模式

  //低电平变高电平的过程中，触发中断1，调用blink函数
  attachInterrupt(1, blink, RISING);   
}

void loop(){
      if(state!=0){              // 如果state不是0时
        state = 0;               // state值赋为0
        digitalWrite(LED_PIN,HIGH);   // 亮灯
        delay(500);          //延时500ms
      }  
      else 
        digitalWrite(LED_PIN,LOW);     // 否则，关灯
} 

void blink(){                //中断函数blink()
    state++;             //一旦中断触发，state就不断自加
}