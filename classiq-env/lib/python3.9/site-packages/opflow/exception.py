#!/usr/bin/env python

class ConstructorError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super(ConstructorError, self).__init__(self,*args,**kwargs)

class OperationError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super(OperationError, self).__init__(self,*args,**kwargs)

class NotcallableError(OperationError):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(self,*args,**kwargs)

class ParameterError(OperationError):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(self,*args,**kwargs)