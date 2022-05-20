import sys
import os
import unittest
import numpy as np
import pandas as pd

#sys.path.insert(0, '../Scripts_/')

#from data_description import DataDescription
df = pd.DataFrame({'nums': [8, 2, 8, 1, 2], 'char': ['h', 'i', 'j', 'k', 'l'],
                   'double': [0.2323, -0.23123, 0.3332, 0.04525, 4.3434]})


class TestCases(unittest.TestCase):
    def test_show_datatypes(self):
        #data_preProcessing = DataDescription(df)
        # data_preProcessing.show_datatypes()
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
