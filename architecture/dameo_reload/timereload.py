#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/qq_21439971/article/details/79356248
"""
import time 
import threading


mutex = threading.Lock()  # 声明一个锁

# Lock 对象 ?


class ReloadThread(threading.Thread):
    """
    ReloadThread
    """

    def __init__(self, log_obj):
        """init"""
        threading.Thread.__init__(self)
        self.__thread_stop = False
        self.__log_obj = log_obj

        self.__interval = 5  # 间隔5s

    def stop(self):
        """ 
        stop
        """
        self.__log_obj.info("ReloadThread stop")
        self.__thread_stop = True

    def setDaeTrue(self):
        """ 
        主线程A中，创建了子线程B，并且在主线程A中调用了B.setDaemon(),
        把主线程A设置为守护线程，若主线程A执行结束了，就不管子线程B是否完成,一并和主线程A退出。
        基本和join()是相反的
        """
        self.__log_obj.info("ReloadThread setDaeTrue")
        self.setDaemon(True)

    def run(self):
        """run"""
        while not self.__thread_stop:
            self.__log_obj.info("ReloadThread run")
            # do something ?
            # acquire(blocking=True, timeout=-1)
            if mutex.acquire(10):  # 获取锁
                # do something
                
                self.__log_obj.info("ReloadThread acquire")
                mutex.release()  # 释放锁
            else:
                self.__log_obj.error("get mutex error")

            time.sleep(self.__interval)
