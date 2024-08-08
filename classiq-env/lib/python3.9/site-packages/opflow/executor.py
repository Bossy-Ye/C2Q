#!/usr/bin/env python

import logging
import json
import pika
import threading
import time

from engine import Engine
from exception import ConstructorError
from exception import OperationError
from util import Util

logger = Util.getLogger(__name__)

class Executor:
    def __init__(self, engine):
        if engine is not None and isinstance(engine, Engine):
            self.__engine = engine
        else:
            raise ConstructorError('"engine" not found or not an Engine object')

    def assertExchange(self, exchangeName, exchangeType='direct'):
        try:
            return self.__declareExchange(exchangeName=exchangeName, exchangeType=exchangeType)
        except OperationError:
            raise
        except:
            raise ConstructorError('assertExchange() has been failed')

    def declareExchange(self, exchangeName, exchangeType='direct', durable=True):
        try:
            return self.__declareExchange(exchangeName=exchangeName, exchangeType=exchangeType, durable=durable)
        except OperationError:
            raise
        except:
            raise OperationError('declareExchange() has been failed')

    def deleteExchange(self, exchangeName):
        try:
            def exchangeDelete(channel):
                channel.exchange_delete(exchange=exchangeName, if_unused=True)
            self.__engine.acquireChannel(exchangeDelete)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('exchange(%s) has been deleted successfully' % (exchangeName))
        except:
            raise OperationError('deleteExchange() has been failed')

    def assertQueue(self, queueName):
        try:
            return self.__declareQueue(queueName=queueName)
        except OperationError:
            raise
        except:
            raise ConstructorError('assertQueue() has been failed')

    def declareQueue(self, queueName, durable=True, exclusive=False, auto_delete=False):
        try:
            return self.__declareQueue(queueName=queueName, durable=durable, exclusive=exclusive, auto_delete=auto_delete)
        except OperationError:
            raise
        except:
            raise OperationError('declareQueue() has been failed')

    def deleteQueue(self, queueName, if_unused=True, if_empty=False):
        try:
            def queueDelete(channel):
                channel.queue_delete(queue=queueName, if_unused=if_unused, if_empty=if_empty)
            self.__engine.acquireChannel(queueDelete)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('queue(%s) has been deleted successfully' % (queueName))
        except:
            raise OperationError('deleteQueue() has been failed')

    def __declareExchange(self, exchangeName, exchangeType='direct', durable=True):
        try:
            def exchangeDetect(channel):
                return channel.exchange_declare(exchange=exchangeName, passive=True)
            result = self.__engine.acquireChannel(exchangeDetect)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('exchange(%s) already exist, skipped' % (exchangeName))
            return result
        except pika.exceptions.ChannelClosed as exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('exchange(%s) not found, exception: %s' % (exchangeName, exception.__class__.__name__))
            def exchangeCreate(channel):
                return channel.exchange_declare(exchange=exchangeName, exchange_type=exchangeType, durable=durable)
            result = self.__engine.acquireChannel(exchangeCreate)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('exchange(%s) has been created successfully' % (exchangeName))
            return result

    def __declareQueue(self, queueName, durable=True, exclusive=False, auto_delete=False):
        try:
            def queueDetect(channel):
                return channel.queue_declare(queue=queueName, passive=True)
            result = self.__engine.acquireChannel(queueDetect)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('queue(%s) already exist' % (queueName))
            return result
        except pika.exceptions.ChannelClosed as exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('queue(%s) not found, exception: %s' % (queueName, exception.__class__.__name__))
            def queueCreate(channel):
                return channel.queue_declare(queue=queueName, durable=durable, exclusive=exclusive, auto_delete=auto_delete)
            result = self.__engine.acquireChannel(queueCreate)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('queue(%s) has been created successfully' % (queueName))
            return result
