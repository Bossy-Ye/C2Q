#!/usr/bin/env python

import logging
import pika
import Queue
import time
import threading

from engine import Engine
from executor import Executor
from task import TimeoutHandler
from util import Util

logger = Util.getLogger(__name__)

class RpcMaster:
    def __init__(self, uri, exchangeName=None, routingKey=None, applicationId=None,
            responseName=None, verbose=False,
            monitorEnabled=True, monitorId=None, monitorInterval=None, monitorTimeout=None):
        if logger.isEnabledFor(logging.DEBUG): logger.debug('Constructor begin ...')
        self.__lock = threading.RLock()
        self.__idle = threading.Condition(self.__lock)
        self.__engine = Engine(**{
            'uri': uri, 
            'exchangeName': exchangeName,
            'exchangeType': 'direct',
            'routingKey': routingKey,
            'applicationId': applicationId,
            'verbose': verbose
        })
        self.__executor = Executor(engine=self.__engine)
        self.__tasks = {}
        self.__timeoutHandler = None
        self.__responseConsumer = None

        if responseName is not None and type(responseName) is str:
            self.__responseName = responseName
            self.__executor.assertQueue(self.__responseName)
        else:
            self.__responseName = None

        if monitorEnabled is not None and type(monitorEnabled) is bool:
            self.__monitorEnabled = monitorEnabled
        else:
            self.__monitorEnabled = True

        if monitorId is not None and type(monitorId) is str:
            self.__monitorId = monitorId
        else:
            self.__monitorId = Util.getUUID()

        if monitorInterval is not None and type(monitorInterval) is int:
            self.__monitorInterval = monitorInterval
        else:
            self.__monitorInterval = 1

        if monitorTimeout is not None and type(monitorTimeout) is int:
            self.__monitorTimeout = monitorTimeout
        else:
            self.__monitorTimeout = 0


    def request(self, routineId, content, options=None):
        if logger.isEnabledFor(logging.DEBUG): logger.debug('request() is invoked')

        if (options is None): options = {}

        forked = ('mode' in options and options['mode'] == 'forked')

        if self.__monitorEnabled and self.__timeoutHandler is None:
            self.__timeoutHandler = TimeoutHandler(tasks=self.__tasks,
                monitorId=self.__monitorId,
                interval=self.__monitorInterval,
                timeout=self.__monitorTimeout)
            self.__timeoutHandler.start()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('timeoutHandler[%s] has been started' % (self.__monitorId))

        consumerInfo = None
        if forked:
            consumerInfo = self.__initResponseConsumer(True)
        else:
            if (self.__responseConsumer is None):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('request() create new ResponseConsumer')
                self.__responseConsumer = self.__initResponseConsumer(False)
            consumerInfo = self.__responseConsumer

        taskId = Util.getUUID()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('request() - new taskId: %s' % (taskId))

        def completeListener():
            self.__lock.acquire()
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('completeListener will be invoked')
                del self.__tasks[taskId]
                if len(self.__tasks) == 0:
                    if forked:
                        self.__engine.cancelConsumer(consumerInfo)
                    self.__idle.notify()
                    if logger.isEnabledFor(logging.DEBUG): logger.debug('tasks is empty')
            finally:
                self.__lock.release()

        if (routineId is not None):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('request() - routineId: %s' % (routineId))
            options['routineId'] = routineId

        task = RpcRequest(options, completeListener)
        self.__tasks[taskId] = task

        headers = { 'routineId': task.routineId, 'requestId': task.requestId }
        properties = { 'headers': headers, 'correlation_id': taskId }
        
        if not consumerInfo['fixedQueue']:
            properties['reply_to'] = consumerInfo['queueName']

        self.__engine.produce(message=content, properties=properties)

        return task

    def __initResponseConsumer(self, forked=False):
        if logger.isEnabledFor(logging.DEBUG):
                logger.debug('__initResponseConsumer() - invoked with forked: %s' % forked)
        def callback(channel, method, properties, body, replyToName):
            taskId = properties.correlation_id
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('responseConsumer - task[%s] received data' % (taskId))

            if (taskId not in self.__tasks):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('responseConsumer - task[%s] not found, skipped' % (taskId))
                return
            task = self.__tasks[taskId]

            task.push({ 'content': body, 'headers': properties.headers })
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('responseConsumer - task[%s] message enqueued' % (taskId))

            return True
        
        options = { 'binding': False, 'prefetch': 1 }
        if (not forked):
            options['queueName'] = self.__responseName
            options['consumerLimit'] = 1
            options['forceNewChannel'] = False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('__initResponseConsumer() - options: %s' % options)
        
        return self.__engine.consume(callback, options)

    def close(self):
        self.__lock.acquire()
        try:
            while len(self.__tasks) > 0: self.__idle.wait()
            if self.__responseConsumer is not None:
                self.__engine.cancelConsumer(self.__responseConsumer)
            if self.__timeoutHandler is not None:
                self.__timeoutHandler.stop()
            self.__engine.close()
        finally:
            self.__lock.release()

    def retain(self):
        if self.__timeoutHandler is not None:
            while self.__timeoutHandler.timer is not None and self.__timeoutHandler.timer.is_alive():
                self.__timeoutHandler.timer.join(1)
        if self.__engine is not None:
            while self.__engine.consumingLoop is not None and self.__engine.consumingLoop.is_alive():
                self.__engine.consumingLoop.join(1)

    @property
    def executor(self):
        return self.__executor

class RpcRequest:
    EMPTY = { 'status': 'EMPTY' }
    ERROR = { 'status': 'ERROR' }

    def __init__(self, options, callback):
        if logger.isEnabledFor(logging.DEBUG): logger.debug('RpcRequest constructor begin')
        self.__requestId = Util.getRequestId(options, True)
        self.__routineId = Util.getRoutineId(options, True)
        self.__timeout = 0
        if 'timeout' in options:
            timeout = options['timeout']
            if (type(timeout) is int or type(timeout) is long) and timeout > 0:
                self.__timeout = timeout
        self.__timestamp = Util.getCurrentTime()
        self.__completeListener = callback
        self.__list = Queue.Queue()

    @property
    def requestId(self):
        return self.__requestId

    @property
    def routineId(self):
        return self.__routineId

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def timeout(self):
        return self.__timeout

    def raiseTimeout(self):
        self.__list.put(self.ERROR, True)
        self.__list.join()

    def hasNext(self):
        self.__current = self.__list.get()
        self.__list.task_done()
        if (self.__current == self.EMPTY): return False
        if (self.__current == self.ERROR): return False
        return True

    def next(self):
        _result = self.__current
        self.__current = None
        return _result

    def push(self, message):
        self.__list.put(item=message, block=True)
        if (self.__isDone(message)):
            self.__list.put(self.EMPTY, True)
            if (callable(self.__completeListener)): self.__completeListener()
        self.__list.join()

    def __isDone(self, message):
        status = None
        if ('headers' in message and message['headers'] is not None):
            headers = message['headers']
            if ('status' in headers): status = headers['status']
        return (status in ['failed', 'completed'])