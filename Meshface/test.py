#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import logging
from logging.handlers import TimedRotatingFileHandler

home_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0]) #,os.path.pardir
sys.path.append(home_dir + "/lib")
sys.path.append(home_dir + "/conf")
import GetConfigDictClass
import gcms_handler_batch


def init_log(log_config):
    """初始化日志"""  
    # 日志天级分割
    log_file_handler = TimedRotatingFileHandler(
        filename=log_config['log_file'], when="midnight", interval=1, backupCount=20)

    log_file_handler.suffix = "%Y%m%d.log"
    log_file_handler.extMatch = re.compile(r"^\d{8}.log$")
    log_format = '%(filename)s:%(module)s:%(funcName)s:\
%(lineno)d:%(levelname)s:%(asctime)s:%(message)s'
    formatter = logging.Formatter(log_format)
    log_file_handler.setFormatter(formatter)
    log_obj = logging.getLogger('')
    log_obj.addHandler(log_file_handler)

    LOG_LEVELS = {'NOTSET': 0, 'DEBUG': 10, 'INFO': 20,
                  'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    log_obj.setLevel(LOG_LEVELS.get(log_config['log_level'], 0))

    return log_obj

if __name__ == "__main__":
    config_obj = GetConfigDictClass.GetConfigDictClass(
            home_dir + '/conf/main.cfg')
    config_obj.load()
    config_dict = config_obj.getDict()

    # 日志初始化
    log_obj = init_log(config_dict.get("LOG", {}))

    gcms_config = config_dict.get("GCMS", {})