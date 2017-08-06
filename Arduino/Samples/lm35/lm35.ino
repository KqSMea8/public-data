/* 温度传感器：LM35
* taobao: 
* doc: http://www.dfrobot.com.cn/community/thread-2485-1-1.html
* warn: 正负极接反后，LM35温度迅速上升烫手  
*/
#define PIN 0  //LM35连到模拟口，并从模拟口读值

unsigned long tepTimer ;

void setup(){
    Serial.begin(9600);        //设置波特率为9600 bps
}

void loop(){ 
    int val=analogRead(PIN);   
    double data = (double) val * (5/10.24);  // 得到电压值，通过公式换成温度
    
    if(millis() - tepTimer > 500){     // 每500ms，串口输出一次温度值
        tepTimer = millis(); 
        Serial.print(data);         // 串口输出温度值
        Serial.println("C");         // 串口输出温度单位
    } 
}