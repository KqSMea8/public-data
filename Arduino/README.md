todo:
1. 如何写Arduino的类库？
2. 如何将Serial的输出信息写入日志中？
3. 测量温度数值，过去n次数据数值采样，过滤掉异常值(大于2个标准差)，再取平均值？
 在ardunio中进行这样的计算是否何时？ ardunio的队列数据结构实现和数学函数？
4.  超声波，不是直接除以声速，而是58？
5. H桥电路，L298N，模块比单独的零件多了哪些东西？
6. 场效应管的作用？
7. IIC通信
8. UART通信
9. 传感器
    
    DHT11
    * lib可安装 
    
    TCS3200 颜色识别
    * lib可安装
    
    NRF24L01 SI24R1 D1B2  #2.4G
    * lib可安装 比较多选哪个？
    
    A7105 CC2500/NRF24L01 #2.4G
    * https://github.com/cassm/a7105
    
    RTC
    * RTCLib 可安装
        PCF8563T
        DS1302 CR2032
        DS323 AT24C32
        24C32 DS1307
    
    HB100 C4B2 doppler radar 
    * https://github.com/dlpoole/HB100-Doppler-Vehicle-Speed-Measurement

    GY-271 HMC5883L 三轴磁场传感器 （还不行，怀疑是硬件问题）
    * lib可安装
    * adafruit sensor unified #需安装该lib，否则会提示找不到 Adafruit_Sensor.h


    GY-521 MPU-6050 三轴加速度陀螺仪 6DOF #实验成功
    * https://playground.arduino.cc/Main/MPU-6050
    * https://github.com/jarzebski/Arduino-MPU6050
    * http://www.instructables.com/id/MPU6050-Arduino-6-Axis-Accelerometer-Gyro-GY-521-B/
    * SCL-A5 SDA-A4 ADO-GND INT-D2 (used for interrupt) 

    GY-91 MPU9250+BMP280 10DOF
    * lib可安装 
    VIN: Voltage Supply Pin
    3V3: 3.3v Regulator output
    GND: 0V Power Supply
    SCL: I2C Clock / SPI Clock
    SDA: I2C Data or SPI Data Input
    SDO/SAO: SPI Data output / I2C Slave Address configuration pin
    NCS: Chip Select for SPI mode only for MPU-9250
    CSB: Chip Select for BMP280





