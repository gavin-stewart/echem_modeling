# Tests for dataFileIO

import unittest
import tools.dataFileIO as dfio
import numpy as np
import os.path 
import os

class IOTests(unittest.TestCase):
	
	def testReadTrue(self):
		"""Test that readTimeCurrentData does no throw an exception for correctly formatted files."""
		fileName = './files/IOTest1'
		try:
			dfio.readTimeCurrentData(fileName)
		except ValueError:
			self.fail("Method readTimeCurrentData raised",
			" a value exception for", fileName)
	
	def testReadFalse1(self):
		"""Test that readTimeCurrentData fails when data is missing in the first column"""
		fileName = './files/IOTest2'
		with self.assertRaises(ValueError):
			dfio.readTimeCurrentData(fileName)
	
	def testReadFalse2(self):
		"""Test that readTimeCurrentData fails when data is missing in the second column"""
		fileName = './files/IOTest3'
		with self.assertRaises(ValueError):
			dfio.readTimeCurrentData(fileName)

	def testReadsCorrectValues(self):
		"""Test that readTimeCurrentData functions correctly"""
		fileName = './files/IOTest1'
		timeRead, currentRead = dfio.readTimeCurrentData(fileName)
		timeCorrect = np.array([1., 0.003, 5.])
		currentCorrect = np.array([0.01, 2e3, 5.1234])
		self.assertTrue(np.array_equal(timeCorrect, timeRead))
		self.assertTrue(np.array_equal(currentCorrect, currentRead))

	def testWriteDoesNotOverwriteUnlessOrderedTo(self):
		t = []
		I = []
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest3"
		with self.assertRaises(ValueError):
			dfio.writeTimeCurrentData(fileName, t, I)

	def testWriteDoesOverwriteWhenOverwriteIsSet(self):
		t = np.random.rand(10)
		I = np.random.rand(10)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest4"
		dfio.writeTimeCurrentData(fileName, t, I, True)
		tRead, IRead = dfio.readTimeCurrentData(fileName)
		self.assertTrue(np.isclose(t, tRead).all())
		self.assertTrue(np.isclose(I, IRead).all())

	def testWriteCreatesNewFileWhenOneDoesNotExists(self):
		t = np.random.rand(10)
		I = np.random.rand(10)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest6"
		self.assertFalse(os.path.exists(fileName)) # File should not exist before we begin.
		dfio.writeTimeCurrentData(fileName, t, I)
		self.assertTrue(os.path.exists(fileName))
		os.remove(fileName)
		self.assertFalse(os.path.exists(fileName))

	def testWriteFailsWhenDataAreDifferentLengths(self):
		d1 = np.random.rand(14)
		d2 = np.random.rand(21)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest5" #NB should never be written to
		with self.assertRaises(ValueError):
			dfio.writeTimeCurrentData(fileName, d1, d2, True)
			dfio.WriteTimeCurrentData(fileName, d2, d1, True)

	def testWriteBinaryDoesNotOverwriteUnlessOrderedTo(self):
		t = np.zeros(5)
		I = np.zeros(5)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest3.npz"
		with self.assertRaises(ValueError):
			dfio.writeTimeCurrentDataBinary(fileName, t, I)

	def testWriteBinaryDoesOverwriteWhenOverwriteIsSet(self):
		t = np.linspace(0,1,10)
		I = np.random.rand(10)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest4.npz"
		dfio.writeTimeCurrentDataBinary(fileName, t, I, True)
		tRead, IRead = dfio.readTimeCurrentDataBinary(fileName)
		self.assertTrue(np.isclose(t, tRead).all())
		self.assertTrue(np.isclose(I, IRead).all())

	def testWriteBinaryCreatesNewFileWhenOneDoesNotExists(self):
		t = np.random.rand(10)
		I = np.random.rand(10)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest6.npz"
		self.assertFalse(os.path.exists(fileName)) # File should not exist before we begin.
		dfio.writeTimeCurrentData(fileName, t, I)
		self.assertTrue(os.path.exists(fileName))
		os.remove(fileName)
		self.assertFalse(os.path.exists(fileName))

	def testWriteBinaryFailsWhenDataAreDifferentLengths(self):
		d1 = np.random.rand(14)
		d2 = np.random.rand(21)
		fileName = "/users/gavart/Private/python/electrochemistry/files/IOTest5.dat" #NB should never be written to
		with self.assertRaises(ValueError):
			dfio.writeTimeCurrentData(fileName, d1, d2, True)
			dfio.WriteTimeCurrentData(fileName, d2, d1, True)

	def testJSONReadDoesNotFailOnSimpleFile(self):
		fileName="/users/gavart/Private/python/electrochemistry/tests/files/test.json"
		dataName="test"
		typeName = "test"
		params = dfio.readParametersFromJSON(fileName, dataName, typeName)
		self.assertFalse(params is None)

	def testJSONReadCanReadDimensionalData(self):
		fileName= "/users/gavart/Private/python/electrochemistry/files/simulationParameters.json"
		dataName="Martin's experiment"
		params = dfio.readDimensionalParametersFromJSON(fileName, dataName)
		self.assertFalse(params is None)
		self.assertTrue(isinstance(params, dict))
		self.assertTrue(np.isclose(params['dE'], 150e-3))
		self.assertTrue(params['reverse'])
