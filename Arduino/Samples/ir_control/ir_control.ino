/*
*/
#include <IRremote.h>
#include <IRremoteInt.h> 

//这段代码来自IRremote库中examples中的 IRrecvDemo
//#include <IRremote.h>          //调用IRremote.h库
int  RECV_PIN = 11;              //定义RECV_PIN变量为11
IRrecv irrecv(RECV_PIN);       //设置RECV_PIN（也就是11引脚）为红外接收端
decode_results results;        //定义results变量为红外结果存放位置

void setup(){
     Serial.begin(9600);          //串口波特率设为9600
     irrecv.enableIRIn();         //启动红外解码
}

void loop() {
     //是否接收到解码数据,把接收到的数据存储在变量results中
     if (irrecv.decode(&results)) {  
            //接收到的数据以16进制的方式在串口输出
            Serial.println(results.value, HEX); // 
            irrecv.resume();  // 继续等待接收下一组信号
      }
}
