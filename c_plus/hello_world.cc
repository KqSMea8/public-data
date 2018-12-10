#include<iostream>  //<>与""的区别?
using namespace std ;
int main(){
    cout << "hello,world!" << endl;
    return 0;
}

// 编译&运行
// g++ hello_world.cc -o output/hello_world
// gcc hello_world.cc  -lstdc++ -o output/hello_world  #编译
// ./output/hello_world  #运行