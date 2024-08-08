#!/usr/bin/env python

import logging
import ConfigParser

from exception import ConstructorError
from exception import OperationError
from util import Util

logger = Util.getLogger(__name__)

class Helper:
	@classmethod
	def createRpcMaster(self, cfgFile):
		

	@classmethod
	def loadConfig(self, cfgFile):
		config = ConfigParser.RawConfigParser()
		config.read(cfgFile)
