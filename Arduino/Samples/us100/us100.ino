/*
* 声波测距模块 US-100
* https://item.taobao.com/item.htm?id=43736207670 #
http://bbs.elecfans.com/forum.php?mod=viewthread&tid=1099658&extra=page=1
* US-100 后面有个跳线帽，带上跳线帽TX-RX - UART模式；不带跳线帽和US-015等相同,GPIO电平触发模式
* UART模式
* error ： avrdude: stk500_getsync() attempt 1 of 10: not in sync: resp=0x00
* upload时，因为复用了串口，要先将超声波模块与Arduino板断开，否则upload会失败。
* http://www.cnblogs.com/journeyonmyway/archive/2012/01/15/2323115.html
* 这篇文档跟上篇不太一样,low,好像有些问题
*/

#include <SoftwareSerial.h>

// 串口
#define _baudrate   9600
#define _rxpin      3  // 
#define _txpin      2 //
#define DBG_UART    dbgSerial   //调试打印串口

SoftwareSerial dbgSerial( _rxpin, _txpin ); // 软串口，调试打印

void setup() 
{ 
  DBG_UART.begin( _baudrate );
  Serial.begin( _baudrate );
} 
void loop() 
{ 
  unsigned int lenHigh = 0; // 高位
  unsigned int lenLow = 0;  // 低位
  unsigned int dist_mm = 0; // 距离
  unsigned int tdata = 0;  
  unsigned int temp = 0;  // 温度

  DBG_UART.flush();    // 清空串口接收缓冲区
  DBG_UART.write(0x55);   // 发送0x55，触发US-100 开始测距
  delay(500);         // 延时500 毫秒
  
  // 当串口接收缓冲区中数据大于2字节
  if(DBG_UART.available() >= 2)
  { 
    lenHigh = DBG_UART.read();        // 距离的高字节
    lenLow = DBG_UART.read();         // 距离的低字节
    dist_mm = lenHigh*256 + lenLow; // 计算距离值
  }

  DBG_UART.flush();    // 清空串口接收缓冲区
  DBG_UART.write(0x50);   // 发送0x50，触发US-100 开始测温
  delay(500);         // 延时500 毫秒

  // 当串口接收缓冲区中数据大于1字节
  if(DBG_UART.available() >= 1)
  { 
    tdata = DBG_UART.read();        // 温度字节
    temp = tdata-45; // 计算温度值
  }
  
  // 有效的测距的结果在1mm 到 10m 之间
//  if((dist_mm > 1) && (dist_mm < 10000)) 
//  {
    Serial.print("Distance is: ");// 输出结果至串口监视器
    Serial.print(dist_mm, DEC);   
    Serial.print("mm  ");
    Serial.print("Temperature is: ");// 输出结果至串口监视器
    Serial.print(temp, DEC);   
    Serial.println(".C");           
//  }
  
  delay(500); // 等待500ms
}
