#!/usr/bin/env python

import logging
import threading
import time

from util import Util

logger = Util.getLogger(__name__)

class TimeoutHandler:
    def __init__(self, tasks={}, interval=1, timeout=0, monitorId=None):
        self.__tasks = tasks
        self.__interval = interval
        self.__timeout = timeout
        
        if monitorId is not None:
            self.__monitorId = monitorId
        else:
            self.__monitorId = Util.getUUID()
        
        def timerTask():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('timerTask[%s] echo .....' % (self.__monitorId))
            if self.__tasks is None or len(self.__tasks) == 0: return
            current = Util.getCurrentTime()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('timerTask[%s] is invoked, current time: %s' % (self.__monitorId, current))
            for taskId in list(self.__tasks):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('timerTask[%s] examine task: %s' % (self.__monitorId, taskId))
                task = self.__tasks[taskId]
                timeout = task.timeout
                if timeout <= 0:
                    timeout = self.__timeout
                if timeout > 0:
                    timediff = current - task.timestamp
                    if (timediff > timeout):
                        del self.__tasks[taskId]
                        task.raiseTimeout()
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('timerTask[%s] run, task[%s] timeout, will be removed' % (self.__monitorId, taskId))
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('timerTask[%s] run, task[%s] in good state, keep running' % (self.__monitorId, taskId))
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('timerTask[%s] run, timeout: %s, skipped' % (self.__monitorId, timeout))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('timerTask[%s] @time[%s] has been done' % (self.__monitorId, current))
            return

        self.__timer = RepeatedTimer(interval=self.__interval, target=timerTask, name='TimeoutInspector')

    def start(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('timeoutHandler[%s] is started' % (self.__monitorId))
        self.__timer.start()

    def stop(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('timeoutHandler[%s] has been stopped' % (self.__monitorId))
        self.__timer.cancel()

    @property
    def timer(self):
        return self.__timer

class RepeatedTimer(threading.Thread):
    """Repeat `target` every `interval` seconds."""
    def __init__(self, interval, target, args=(), kwargs={}, name=None):
        super(RepeatedTimer, self).__init__(target=self.__target, name=name)
        self.__interval = interval
        self.__target = target
        self.__args = args
        self.__kwargs = kwargs
        self.__current = time.time()
        self.__event = threading.Event()

    def __target(self):
        while not self.__event.is_set():
            self.__event.wait(self.__time)
            self.__target(*self.__args, **self.__kwargs)

    @property
    def __time(self):
        return self.__interval - ((time.time() - self.__current) % self.__interval)

    def cancel(self):
        self.__event.set()
