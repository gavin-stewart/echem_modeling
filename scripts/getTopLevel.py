# Code for getting the top-level directory from the scripts folder.
import os, sys, os.path
topLevel = os.path.realpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.pardir))
sys.path.append(topLevel)

def makePathFromTop(name):
	return os.path.join(topLevel, name)

def makeFilePath(name):
	return os.path.join(topLevel, "files/"+name)


