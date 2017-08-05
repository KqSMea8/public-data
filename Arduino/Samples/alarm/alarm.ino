/* 蜂鸣器 https://item.taobao.com/item.htm?id=521788890629
*  sample: https://www.arduino.cc/en/Tutorial/Melody
*/

#define ALARM_PIN 4

int length = 15;                  // the number of notes
char notes[] = "ccggaagffeeddc "; // a space represents a rest
int beats[] = {1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 4};
int tempo = 300;

void playTone(int tone, int duration)
{
    for (long i = 0; i < duration * 1000L; i += tone * 2)
    {
        digitalWrite(ALARM_PIN, HIGH);
        delayMicroseconds(tone);
        digitalWrite(ALARM_PIN, LOW);
        delayMicroseconds(tone);
    }
}

void playNote(char note, int duration)
{
    char names[] = {'c', 'd', 'e', 'f', 'g', 'a', 'b', 'C'};
    int tones[] = {1915, 1700, 1519, 1432, 1275, 1136, 1014, 956};

    // play the tone corresponding to the note name
    for (int i = 0; i < 8; i++)
    {
        if (names[i] == note)
        {
            playTone(tones[i], duration);
        }
    }
}

// alram 
void alarm()
{
    float sinVal;            
    int toneVal;

    for (int x = 0; x < 180; x++)
    {
        //将sin函数角度转化为弧度
        sinVal = (sin(x * (3.1412 / 180)));
        //用sin函数值产生声音的频率
        toneVal = 2000 + (int(sinVal * 1000));
        //给输出引脚一个
        tone(ALARM_PIN, toneVal);
        delay(2);
    }
}

void setup()
{
    pinMode(ALARM_PIN, OUTPUT);
}

void loop()
{
    // alarm();

    for (int i = 0; i < length; i++)
    {
        if (notes[i] == ' ')
        {
            delay(beats[i] * tempo); // rest
        }
        else
        {
            playNote(notes[i], beats[i] * tempo);
        }

        // pause between notes
        delay(tempo / 2);
    }
}