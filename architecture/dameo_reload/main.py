#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import daemon #

import logging
import time 
import timereload

logging.basicConfig(level=logging.DEBUG)


def main():
    logging.info("time_reload_thread init")
    time_reload_thread = \
        timereload.ReloadThread(logging)

    logging.info("setDaeTrue")
    time_reload_thread.setDaeTrue()

    try:
        logging.info("start")
        time_reload_thread.start()
    except Exception as e:
        sys.exit(1)

    while True:
        logging.info("while ...")
        time.sleep(10)

    if not time_reload_thread.isAlive():
        logging.info("not isAlive")
        sys.exit(1)

    logging.info("time_reload_thread.stop")
    time_reload_thread.stop()


if __name__ == '__main__':
    main()
