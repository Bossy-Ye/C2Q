#!/usr/bin/env python

import logging
import pika

from engine import Engine
from executor import Executor
from util import Util
from exception import OperationError
from exception import ParameterError

logger = Util.getLogger(__name__)

class PubsubHandler:
    def __init__(self, uri, exchangeName=None, exchangeType='direct',
            routingKey=None, otherKeys=[], applicationId=None, verbose=False,
            subscriberName=None, recyclebinName=None, redeliveredLimit=3):
        self.__engine = Engine(**{
            'uri': uri, 
            'exchangeName': exchangeName,
            'exchangeType': exchangeType,
            'routingKey': routingKey,
            'otherKeys': otherKeys,
            'applicationId': applicationId,
            'verbose': verbose
        })
        self.__executor = Executor(engine=self.__engine)
        self.__listener = None

        if subscriberName is not None and type(subscriberName) is str:
            self.__subscriberName = subscriberName
            self.__executor.assertQueue(self.__subscriberName)
        else:
            self.__subscriberName = None

        if recyclebinName is not None and type(recyclebinName) is str:
            self.__recyclebinName = recyclebinName
            self.__executor.assertQueue(self.__recyclebinName)
        else:
            self.__recyclebinName = None

        if redeliveredLimit is not None and type(redeliveredLimit) is int:
            self.__redeliveredLimit = 0 if (redeliveredLimit < 0) else redeliveredLimit
        else:
            self.__recyclebinName = None

    def publish(self, data, opts={}, routingKey=None):
        opts = {} if opts is None else opts
        if 'requestId' not in opts or opts['requestId'] is None:
            opts['requestId'] = Util.getUUID()

        properties = { 'headers': opts }

        override = {}
        if routingKey is not None and type(routingKey) is str:
            override['routingKey'] = routingKey

        self.__engine.produce(message=data, properties=properties, override=override)

    def subscribe(self, pubsubListener):
        self.__listener = pubsubListener if (self.__listener is None) else self.__listener
        if self.__listener is None:
            raise ParameterError('Subscriber callback should not be None')
        elif self.__listener != pubsubListener:
            raise ParameterError('PubsubHandler only supports single Subscriber callback')
        def opflowListener(channel, method, properties, body, replyToName):
            headers = properties.headers
            requestId = Util.getRequestId(headers)
            if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Request[%s] - subscribe() receive a message' % requestId)
            try:
                pubsubListener(body, headers)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Request[%s] - subscribe() has processed successfully' % requestId)
            except Exception as exception:
                redeliveredCount = 0
                if 'redeliveredCount' in headers and type(headers['redeliveredCount']) is int:
                     redeliveredCount = headers['redeliveredCount']
                redeliveredCount += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('Request[%s] - redeliveredCount: %s/%s' % 
                        (requestId, redeliveredCount, self.__redeliveredLimit))
                headers['redeliveredCount'] = redeliveredCount
                props = properties
                if redeliveredCount <= self.__redeliveredLimit:
                    self.__sendToQueue(content=body, properties=props, 
                        queueName=self.__subscriberName, channel=channel)
                else:
                    if self.__recyclebinName is not None:
                        self.__sendToQueue(content=body, properties=props, 
                            queueName=self.__recyclebinName, channel=channel)
            return True

        return self.__engine.consume(opflowListener, {
            'noAck': True,
            'queueName': self.__subscriberName
        })

    def close(self):
        if self.__engine is not None:
            if logger.isEnabledFor(logging.DEBUG): logger.debug('Engine is closing')
            self.__engine.close()

    def retain(self):
        _CONSUMING_LOOP_INTERVAL = 1
        if self.__engine is not None:
            while self.__engine.consumingLoop is not None and self.__engine.consumingLoop.is_alive():
                if self.__engine.verbose and logger.isEnabledFor(logging.DEBUG):
                    logger.debug('consumingLoop interval: %s second(s)' % _CONSUMING_LOOP_INTERVAL)
                self.__engine.consumingLoop.join(_CONSUMING_LOOP_INTERVAL)

    def __sendToQueue(self, content, properties, queueName, channel):
        try:
            channel.basic_publish(exchange='', routing_key=queueName,
                body=content, properties=properties)
        except:
            raise OperationError('sendToQueue(%s) has been failed' % queueName)

    def __copyProperties(self, properties):
        props = pika.spec.BasicProperties(**dict(properties, headers=headers))
        return props

    @property
    def executor(self):
        return self.__executor
