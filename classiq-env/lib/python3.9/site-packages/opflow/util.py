#!/usr/bin/env python

import logging
import time
import uuid

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

class Util(object):
    @classmethod
    def getRoutineId(cls, headers, uuidIfNotFound=False):
        return cls.getHeaderField(headers, 'routineId', uuidIfNotFound)

    @classmethod
    def getRequestId(cls, headers, uuidIfNotFound=True):
        return cls.getHeaderField(headers, 'requestId', uuidIfNotFound)
    
    @classmethod
    def getHeaderField(cls, headers, fieldName, uuidIfNotFound=True):
        if (headers is None or fieldName not in headers):
            if uuidIfNotFound:
                return cls.getUUID()
            else:
                return None
        return headers[fieldName]

    @classmethod
    def getUUID(cls):
        return str(uuid.uuid4())

    @classmethod
    def getCurrentTime(cls):
        return time.time()

    @staticmethod
    def getLogger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)
        return logger
