/* ----------------------------------------
* http://www.cnblogs.com/journeyonmyway/archive/2012/01/15/2323115.html
* 有些问题
* 超声波测距模块US-100串口方式测距
* 选择串口方式需要插上模块背面的跳线
* US-100的探头面向自己时，从左到右Pin脚依次为：
* VCC / Trig(Tx) / Echo(Rx) / GND / GND
* 两个GND只需要一个接地即可
* Trig 接1脚，Echo接0脚
* ----------------------------------------- */

void setup(){ 
  // 将Arduino 的RX 与TX（Digital IO 0 和1）分别于US-100 的Echo/Rx 和Trig/Tx相连
  // 确保连接前已经插上跳线，使US-100 处于串口模式
  Serial.begin(9600); // 设置波特率为 9600bps.
}

void loop(){
  unsigned int lenHigh = 0; // 高位
  unsigned int lenLow = 0;  // 低位
  unsigned int dist_mm = 0; // 距离
  
  Serial.flush();     // 清空串口接收缓冲区
  Serial.write(0x55); // 发送0x55，触发US-100 开始测距
  delay(500);         // 延时500 毫秒
  
  // 当串口接收缓冲区中数据大于2字节
  if(Serial.available() >= 2){ 
    lenHigh = Serial.read();        // 距离的高字节
    lenLow = Serial.read();         // 距离的低字节
    dist_mm = lenHigh*256 + lenLow; // 计算距离值
    
    // 有效的测距的结果在1mm 到 10m 之间
    if((dist_mm > 1) && (dist_mm < 10000)) 
    {
      Serial.print("Distance is: ");// 输出结果至串口监视器
      Serial.print(dist_mm, DEC);   
      Serial.println("mm");         
    }

    Serial.flush();    // 清空串口接收缓冲区
    Serial.write(0x50);   // 发送0x50，触发US-100 开始测温
    delay(500);

    if(Serial.available() >= 1) {  
      unsigned int tdata = Serial.read();        // 温度字节
      unsigned int temp = tdata-45; // 计算温度值
    
      Serial.print("Temperature is: ");// 输出结果至串口监视器
      Serial.print(temp, DEC);   
      Serial.println(".C"); 
    }

  }
  
  delay(500); // 等待500ms
}