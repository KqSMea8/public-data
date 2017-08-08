#include <Servo.h>           // 声明调用Servo.h库
Servo myservo;               // 创建一个舵机对象
int pos = 0;          // 变量pos用来存储舵机位置


int potpin = 0;              // 连接到模拟口0               
int val;                     //变量val用来存储从模拟口0读到的值

void setup() {
  // put your setup code here, to run once:
  myservo.attach(9);          // 将引脚9上的舵机与声明的舵机对象连接起来
  Serial.begin(9600);

}

void loop() {
//  for(pos = 0; pos < 180; pos += 1){    // 舵机从0°转到180°，每次增加1°          
//      myservo.write(pos);           // 给舵机写入角度   
//      delay(15);                    // 延时15ms让舵机转到指定位置
//   }
//    for(pos = 180; pos>=1; pos-=1) {    // 舵机从180°转回到0°，每次减小1°                               
//       myservo.write(pos);        // 写角度到舵机     
//       delay(15);                 // 延时15ms让舵机转到指定位置
//    } 
    
  // put your main code here, to run repeatedly:
  val = analogRead(potpin);         //从模拟口0读值，并通过val记录 
  val = map(val, 0, 1023, 0, 179);  //通过map函数进行数值转换   
  Serial.println(val) ; 
  
  myservo.write(val);               // 给舵机写入角度  
  delay(15);                        // 延时15ms让舵机转到指定位置  

}