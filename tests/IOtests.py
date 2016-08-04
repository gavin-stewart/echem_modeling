"""Tests the file IO functions in tools/io.py
"""

import unittest
import tools.io as io
import numpy as np
import os.path
import os

class IOTests(unittest.TestCase): #pylint: disable=R0904
    """Test cases to test tools/io.py
    """

    def test_read_true(self):
        """Test that readTimeCurrentData does no throw an exception for
        correctly formatted files.
        """
        file_name = './files/IOTest1'
        try:
            io.readTimeCurrentData(file_name)
        except ValueError:
            self.fail("Method readTimeCurrentData raised" +
                      " a value exception for" + file_name)

    def test_read_false1(self):
        """Test that readTimeCurrentData fails when data is missing in the
        first column
        """
        file_name = './files/IOTest2'
        with self.assertRaises(ValueError):
            io.readTimeCurrentData(file_name)

    def test_read_false2(self):
        """Test that readTimeCurrentData fails when data is missing in the
        second column
        """
        file_name = './files/IOTest3'
        with self.assertRaises(ValueError):
            io.readTimeCurrentData(file_name)

    def test_reads_correct_values(self):
        """Test that readTimeCurrentData functions correctly
        """
        file_name = './files/IOTest1'
        time_read, current_read = io.readTimeCurrentData(file_name)
        time_correct = np.array([1., 0.003, 5.])
        current_correct = np.array([0.01, 2e3, 5.1234])
        self.assertTrue(np.array_equal(time_correct, time_read))
        self.assertTrue(np.array_equal(current_correct, current_read))

    def test_write_does_not_overwrite(self):
        """Test that writeTimeCurrentData does not overwrite file without
        user flag.
        """
        time = []
        current = []
        file_name = "./files/IOTest3"
        with self.assertRaises(ValueError):
            io.writeTimeCurrentData(file_name, time, current)

    def test_write_does_overwrite(self):
        """Test that writeTimeCurrentData does overwrite file when user sets
        overwrite flag
        """
        time = np.random.rand(10)
        current = np.random.rand(10)
        file_name = "./files/IOTest4"
        io.writeTimeCurrentData(file_name, time, current, True)
        time_read, current_read = io.readTimeCurrentData(file_name)
        self.assertTrue(np.allclose(time, time_read))
        self.assertTrue(np.allclose(current, current_read))

    def test_write_creates_new_file(self):
        """Test that writeTimeCurrentData creates a new file if one does not
        exist.
        """
        time = np.random.rand(10)
        current = np.random.rand(10)
        file_name = "./files/IOTest6"
        # File should not exist before we begin.
        if os.path.exists(file_name):
            os.remove(file_name)
        io.writeTimeCurrentData(file_name, time, current)
        self.assertTrue(os.path.exists(file_name))
        os.remove(file_name)
        self.assertFalse(os.path.exists(file_name))

    def test_write_data_len_check(self):
        """Test that an exception is raised when time and current data passed
        to writeTimeCurrentData have difference lengths.
        """
        data1 = np.random.rand(14)
        data2 = np.random.rand(21)
        file_name = "./files/IOTest5" #NB should never be written to
        with self.assertRaises(ValueError):
            io.writeTimeCurrentData(file_name, data1, data2, True)
            io.writeTimeCurrentData(file_name, data2, data1, True)

    def test_write_binary_not_overwrite(self):
        """Test that writeTimeCurrentDataBinary does not overwrite a file
        when user has not set overwrite flag.
        """
        time = np.zeros(5)
        current = np.zeros(5)
        file_name = "./files/IOTest3.npz"
        with self.assertRaises(ValueError):
            io.writeTimeCurrentDataBinary(file_name, time, current)

    def test_write_binary_overwrite(self):
        """Test that writeTimeCurrentDataBinary does overwrite a file when
        user sets the overwrite flag.
        """
        time = np.linspace(0, 1, 10)
        current = np.random.rand(10)
        file_name = "./files/IOTest4.npz"
        io.writeTimeCurrentDataBinary(file_name, time, current, True)
        time_read, current_read = io.readTimeCurrentDataBinary(file_name)
        self.assertTrue(np.allclose(time, time_read))
        self.assertTrue(np.allclose(current, current_read))

    def test_write_binary_creates_file(self):
        """Test that writeTimeCurrentDataBinary creates a new file when one
        does not exist.
        """
        time = np.random.rand(10)
        current = np.random.rand(10)
        file_name = "./files/IOTest6.npz"
        # File should not exist before we begin.
        if os.path.exists(file_name):
            os.remove(file_name)
        io.writeTimeCurrentData(file_name, time, current)
        self.assertTrue(os.path.exists(file_name))
        os.remove(file_name)
        self.assertFalse(os.path.exists(file_name))

    def test_write_binary_length_check(self):
        """Test that writeTimeCurrentDataBinary raises an exception when the
        lengths of the time and current data differ.
        """
        data1 = np.random.rand(14)
        data2 = np.random.rand(21)
        file_name = "./files/IOTest5.dat" #NB should never be written to
        with self.assertRaises(ValueError):
            io.writeTimeCurrentData(file_name, data1, data2, True)
            io.writeTimeCurrentData(file_name, data2, data1, True)

    def test_JSON_read_finds_value(self): #pylint: disable=C0103
        """Test that readParametersFromJSON can find data in a simple file.
        """
        file_name = "./tests/files/test.json"
        data_name = "test"
        type_name = "test"
        params = io.readParametersFromJSON(file_name, data_name, type_name)
        self.assertFalse(params is None)

    def test_JSON_read_work_dim_data(self):#pylint: disable=C0103
        """Test that readParametersFromJSON can load paramters correctly fron
        JSON.
        """
        file_name = "./files/simulationParameters.json"
        data_name = "Martin's experiment"
        params = io.readDimensionalParametersFromJSON(file_name, data_name)
        self.assertFalse(params is None)
        self.assertTrue(isinstance(params, dict))
        self.assertTrue(np.isclose(params['dE'], 150e-3))
        self.assertTrue(params['reverse'])
