

/*
硬件： https://item.taobao.com/item.htm?id=43808540465
名称： HMC5883L 模块(三轴磁场模块)
型号： GY-271
通信方式： IIC通信协议
测量范围： ±1.3-8 高斯
连接方式： 
HMC5883L | UNO | Mega2560
SCL      | A5  |  21
SDA      | A4  |  20
DRDY为中断引脚

HMC5883L Lib:
https://github.com/jarzebski/Arduino-HMC5883L


常见几种arduino型号的IIC引脚定义
Uno, Ethernet |	A4 (SDA), A5 (SCL)
Mega2560	| 20 (SDA), 21 (SCL)
Leonardo	| 2 (SDA), 3 (SCL)
Due	20 (SDA), | 21  (SCL), SDA1, SCL1

*/

#include <Wire.h> //I2C Arduino Library
#include <HMC5883L.h>

#define address 0x1E //0011110b, I2C 7bit address of HMC5883

void setup(){
  //Initialize Serial and I2C communications
  Serial.begin(9600);
  Wire.begin();
  
  //Put the HMC5883 IC into the correct operating mode
  Wire.beginTransmission(address); //open communication with HMC5883
  Wire.write(0x02); //select mode register
  Wire.write(0x00); //continuous measurement mode
  Wire.endTransmission();
}

void loop(){  
  int x,y,z; //triple axis data

  //Tell the HMC5883 where to begin reading data
  Wire.beginTransmission(address);
  Wire.write(0x03); //select register 3, X MSB register
  Wire.endTransmission();
  
 
 //Read data from each axis, 2 registers per axis
  Wire.requestFrom(address, 6);
  if(6<=Wire.available()){
    x = Wire.read()<<8; //X msb
    x |= Wire.read(); //X lsb
    z = Wire.read()<<8; //Z msb
    z |= Wire.read(); //Z lsb
    y = Wire.read()<<8; //Y msb
    y |= Wire.read(); //Y lsb
  }
  
  //Print out values of each axis
  Serial.print("x: ");
  Serial.print(x);
  Serial.print("  y: ");
  Serial.print(y);
  Serial.print("  z: ");
  Serial.println(z);
  
  delay(250);
}

