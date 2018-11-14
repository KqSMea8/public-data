

def oushu():
    for i in range(0,100):
        # print(i) 
        if i%2==0:
            print(i)

def a(n):
    y = 0
    for i in range(1,n+1):
        y = y + i 
    print(y)

def a1(n=100):
    print( n*(n+1)/2 )



def Fibonacci():
    l1 = 0
    l2 = 1  
    print(l2)
    for i in range(1,100):
        l3 = l1 + l2
        print(i,l3,l2/l3) 
        l1 = l2 
        l2 = l3 

def is_sushu(li,n):
    for i in li :
        if n%i ==0 :
            return False 
    return True    

def get_sushu():
    li = [2]
    for i in range(3,100):
        if is_sushu(li,i):
            print(i)
            li.append(i)

def get_e(n):
    return (1+1/n)**n  

def get_all_e():
    for i in range(1,50000): 
        e = get_e(i)
        if i%100:
            print(e)

from decimal import Decimal

def get_pi(n=500000):
    x = Decimal(1)
    for i in range(1,n):
        if i%2==0:
            x = x + Decimal(1)/(i*2+1)
        else :
            x = x - Decimal(1)/(i*2+1)

        if i%5000:
            print(i,(i*2+1),4*x)

    

# a1(100)
# a(100)
# oushu()
# Fibonacci()
# get_sushu()
# get_all_e()
get_pi()