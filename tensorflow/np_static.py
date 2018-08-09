#!/usr/bin/python
# -*- coding: utf-8 -*-
import os  
import re 
import hashlib

def md5sum(str_txt):
    m = hashlib.md5()
    m.update(str_txt)
    return m.hexdigest()


def process(file):
    if not file.endswith('.ipynb') and not file.endswith('.py'):
        return 
    with open(file) as f:
        for line in f: 
            pattern = re.compile(r'(np\.[^(]+)')
            match = pattern.search(line)
            if match: 
                print(md5sum(line.encode('utf-8')), match.groups()[0])   

def run(mypath):
    for x in os.listdir(mypath):
        full_path = os.path.join(mypath,x)
        if os.path.isfile(full_path):
            process(full_path)
        else :
            run(full_path)

if __name__ == "__main__":
    run("./tensorflow_cookbook")

#python3 np_static.py | awk '{print $2}' | sort | uniq -c | sort -r

# python3 np_static.py | awk '{print $2}' | sort | uniq -c