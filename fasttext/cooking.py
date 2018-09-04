#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fasttext python sample
"""
import fasttext 
from pyfasttext import FastText

print(fasttext.__VERSION__)
# print(pyfasttext.__version__)

def fasttext_sample():
    """https://pypi.org/project/fasttext/
    Traceback (most recent call last):
    File "fasttext/fasttext.pyx", line 152, in fasttext.fasttext.load_model
    RuntimeError: vector

    During handling of the above exception, another exception occurred:
    Traceback (most recent call last):
    File "cooking.py", line 10, in <module>
        model = fasttext.load_model('output/model_cooking_6.bin')
    File "fasttext/fasttext.pyx", line 154, in fasttext.fasttext.load_model
    Exception: fastText: Cannot load output/model_cooking_6.bin due to C++ extension failed to allocate the memory

    """
    model = fasttext.load_model('output/model_cooking_5.ftz')
    result = model.test('test.txt')
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)

def pyfasttext_sample():
    """https://pypi.org/project/pyfasttext/
    """
    model = FastText()
    # model.load_model('output/model_cooking_6.bin')
    model.load_model('output/model_cooking_5.ftz')
    result = model.predict_file('data/cooking/pre_cooking.valid',2)
    for i,r in enumerate(result):
        print(i,r)

if __name__ == "__main__":
    # fasttext_sample()
    pyfasttext_sample()
    
     

