#!/usr/bin/env python

import logging
import json
import pika
import threading
import time

from exception import ConstructorError
from exception import OperationError
from exception import NotcallableError
from util import Util

logger = Util.getLogger(__name__)

class Engine:
    def __init__(self, **kwargs):
        if 'uri' in kwargs and type(kwargs['uri']) is str and kwargs['uri'] is not None:
            self.__uri = kwargs['uri']
        else:
            raise ConstructorError('"uri" not found or not a string')

        try:
            self.__connection = pika.BlockingConnection(pika.URLParameters(self.__uri))
            self.__channel = None
            self.__thread = None

            if ('exchangeName' in kwargs):
                self.__exchangeName = kwargs['exchangeName']
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeName value: %s' % self.__exchangeName)
            else:
                self.__exchangeName = None
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeName is empty')

            if ('exchangeType' in kwargs):
                self.__exchangeType = kwargs['exchangeType']
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeType value: %s' % self.__exchangeType)
            else:
                self.__exchangeType = 'direct'
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeType is empty, use "direct" as default')

            if ('exchangeDurable' in kwargs) and type(kwargs['exchangeDurable']) is bool:
                self.__exchangeDurable = kwargs['exchangeDurable']
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeDurable value: %s' % self.__exchangeDurable)
            else:
                self.__exchangeDurable = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('exchangeDurable is empty, use True as default')

            if (self.__exchangeName != None and self.__exchangeType != None):
                channel = self.__getChannel()
                channel.exchange_declare(exchange=self.__exchangeName,
                                         type=self.__exchangeType,
                                         durable=self.__exchangeDurable)
            
            if 'routingKey' in kwargs:
                self.__routingKey = kwargs['routingKey']
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('routingKey value: %s' % self.__routingKey)
            else:
                self.__routingKey = None

            if 'otherKeys' in kwargs:
                if type(kwargs['otherKeys']) is str:
                    self.__otherKeys = kwargs['otherKeys'].split(',')
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('otherKeys value: %s' % self.__otherKeys)
                if type(kwargs['otherKeys']) is list:
                    self.__otherKeys = kwargs['otherKeys']
            else:
                self.__otherKeys = []
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('otherKeys is empty, use [] as default')

            if 'applicationId' in kwargs and type(kwargs['applicationId']) is str:
                self.__applicationId = kwargs['applicationId']
            else:
                self.__applicationId = None
            
            if ('verbose' in kwargs) and type(kwargs['verbose']) is bool:
                self.__verbose = kwargs['verbose']
            else:
                self.__verbose = False

        except:
            raise ConstructorError('Error on connecting or exchange declaration')

    def produce(self, message, properties, override=None):
        if (self.__applicationId is not None):
            properties['app_id'] = self.__applicationId
        basicProperties = pika.spec.BasicProperties(**properties)
        self.__channel.basic_publish(body=message, properties=basicProperties,
                exchange=self.__exchangeName, routing_key=self.__routingKey)

    def consume(self, callback, options):
        _channel = None
        if ('forceNewChannel' in options and options['forceNewChannel']):
            _channel = self.__connection.channel()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('forceNewChannel is True, create new channel')
        else:
            _channel = self.__getChannel()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('forceNewChannel is False, use default channel')

        _queueName = None
        _fixedQueue = True
        _declareOk = None
        if ('queueName' in options and options['queueName'] != None):
            _declareOk = _channel.queue_declare(queue=options['queueName'],durable=True)
            _fixedQueue = True
        else:
            _declareOk = _channel.queue_declare()
            _fixedQueue = False
        _queueName = _declareOk.method.queue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('_queueName after run queue_declare(): %s' % _queueName)

        if (('binding' not in options or options['binding'] != False) and (self.__exchangeName != None)):
            if self.__routingKey is not None:
                self.__bindExchange(channel=self.__channel,
                        exchangeName=self.__exchangeName,
                        queueName=_queueName,
                        routingKeys=[self.__routingKey])
            if type(self.__otherKeys) is list and len(self.__otherKeys) > 0:
                self.__bindExchange(channel=self.__channel,
                        exchangeName=self.__exchangeName,
                        queueName=_queueName,
                        routingKeys=self.__otherKeys)

        _replyToName = None
        if ('replyTo' in options and options['replyTo'] is not None):
            _checkOk = _channel.queue_declare(queue=options['replyTo'],passive=True)
            _replyToName = _checkOk.method.queue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('_replyToName after check: %s' % _replyToName)

        _noAck = True
        if ('noAck' in options and type(options['noAck']) is bool):
            _noAck = options['noAck']
        if logger.isEnabledFor(logging.DEBUG): logger.debug('_noAck: %s' % _noAck)

        def rpcCallback(channel, method, properties, body):
            requestID = Util.getRequestId(properties.headers, False)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Request[%s] / DeliveryTag[%s] / ConsumerTag[%s]' % 
                    (requestID, method.delivery_tag, method.consumer_tag))
            try:
                if self.__applicationId is not None and self.__applicationId != properties.app_id:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Request[%s] received app_id:%s, but accepted app_id:%s, rejected' % 
                            (requestID, properties.app_id, self.__applicationId))
                else:
                    captured = callback(channel, method, properties, body, _replyToName)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug('Request[%s] invoke Ack(%s, False)) / ConsumerTag[%s]' % 
                            (requestID, method.delivery_tag, method.consumer_tag))
                    if captured:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('Request[%s] has finished successfully' % (requestID))
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug('Request[%s] has not matched the criteria, skipped' % (requestID))
                if not _noAck:
                    channel.basic_ack(delivery_tag=method.delivery_tag,multiple=False)
            except Exception as exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Request[%s] is broken by exception: %s' % (requestID, exception.__class__.__name__))
                    logger.debug('Request[%s] has failed. Rejected but service still alive' % (requestID))
                if not _noAck:
                    channel.basic_ack(delivery_tag=method.delivery_tag,multiple=False)

        _consumerTag = _channel.basic_consume(rpcCallback, queue=_queueName, no_ack=_noAck)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('_consumerTag after run basic_consume(): %s' % _consumerTag)

        self.__start_consuming()

        _consumerInfo = { 'channel': _channel,'queueName': _queueName,
            'fixedQueue': _fixedQueue,'consumerTag': _consumerTag }

        return _consumerInfo

    def acquireChannel(self, operator):
        if not callable(operator):
            if logger.isEnabledFor(logging.ERROR):
                logger.error('channelOperator is not callable')
            raise NotcallableError('operator should be a function')
        channel = self.__connection.channel()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('temporary channel has been created')
        output = None
        try:
            output = operator(channel=channel)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('channel_operator has been invoked successfully')
        finally:
            if channel is not None and channel.is_open:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('temporary channel will be closed')
                channel.close()
        return output

    def cancelConsumer(self, consumerInfo=None):
        if consumerInfo is None: return
        if 'channel' not in consumerInfo or 'consumerTag' not in consumerInfo: return
        consumerInfo['channel'].basic_cancel(consumerInfo['consumerTag'])

    def close(self):
        self.__stop_consuming()
        if self.__connection is not None and self.__connection.is_open:
            self.__connection.close()

    @property
    def verbose(self):
        return self.__verbose

    @property
    def consumingLoop(self):
        return self.__thread

    def __getChannel(self):
        if self.__channel is None:
            self.__channel = self.__connection.channel()
        return self.__channel

    def __bindExchange(self, channel, exchangeName, queueName, routingKeys):
        channel.exchange_declare(exchange=exchangeName, passive=True)
        channel.queue_declare(queue=queueName, passive=True)
        for routingKey in routingKeys:
            channel.queue_bind(exchange=exchangeName, queue=queueName, routing_key=routingKey)
        pass

    def __start_consuming(self):
        _LOOP_DATA_EVENTS = 0.01
        def startConsumer():
            self.__connection.process_data_events(_LOOP_DATA_EVENTS)

        if self.__thread is None:
            if self.__verbose and logger.isEnabledFor(logging.DEBUG):
                logger.debug('start thread invoke connection.process_data_events(%s)' % (_LOOP_DATA_EVENTS))
            self.__thread = StoppableThread(target=startConsumer,name='ConsumingThread')
            self.__thread.start()

    def __stop_consuming(self):
        _WAIT_THREAD_STOP = 0.5
        if self.__thread is not None:
            if self.__verbose and logger.isEnabledFor(logging.DEBUG):
                logger.debug('stop thread in (%s) seconds' % (_WAIT_THREAD_STOP))
            self.__thread.stop()
            time.sleep(_WAIT_THREAD_STOP)


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped condition."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        super(StoppableThread, self).__init__(group, target, name, args, kwargs)
        self.__target = target
        self.__args = args
        self.__kwargs = kwargs
        self.__stop_event = threading.Event()

    def run(self):
        while not self.__stop_event.is_set():
            self.__target(*self.__args, **self.__kwargs)

    def stop(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('StoppableThread enable stop_event')
        self.__stop_event.set()
