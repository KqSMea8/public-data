#!/usr/bin/python
# -*- coding: utf-8 -*-
"""统计numpy,tensorflow的使用情况"""
import os  
import sys 
import re 
import hashlib

def md5sum(str_txt):
    m = hashlib.md5()
    m.update(str_txt)
    return m.hexdigest()


def process(file,prex='np'):
    if not file.endswith('.ipynb') and not file.endswith('.py'):
        return 
    with open(file) as f:
        for line in f: 
            pattern = re.compile(r'('+prex+'\.[^(^)^`^"^,]+)')
            match = pattern.search(line)
            if match: 
                val = match.groups()[0] 
                # val = val.replace('`','')
                val = val.replace("\\n",'')
                # val = val.replace('"','')
                # val = val.replace(',','')
                print(md5sum(line.encode('utf-8')) , val)   

def run(mypath,prex='np'):
    for x in os.listdir(mypath):
        full_path = os.path.join(mypath,x)
        if os.path.isfile(full_path):
            process(full_path,prex)
        else :
            run(full_path,prex)

def execute(prex):
    run("./tensorflow_cookbook",prex)
    run("./TensorFlow_for_Machine_Intelligence",prex) 

if __name__ == "__main__":
    prex = sys.argv[1]
    execute(prex)
    # execute('tf') 

# python3 np_static.py | awk '{print $2}' | sort | uniq -c | sort -r | awk '{print $2,"|",$1,"| ."}' 
# python3 np_static.py np | awk '{print $2}' | sort | uniq -c | sort -r | awk '{print $2,$1}' 
# python3 np_static.py tf | awk '{print $2}' | sort | uniq -c | sort -r | awk '{print $2,$1}' 