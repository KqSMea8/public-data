#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fasttext python sample
"""
import fasttext


if __name__ == "__main__":
    model = fasttext.load_model('output/model_cooking_6.bin')
    result = classifier.test('data/cooking/pre_cooking.valid')
    # print('P@1:', result.precision)
    # print('R@1:', result.recall)
    # print('Number of examples:', result.nexamples)
     