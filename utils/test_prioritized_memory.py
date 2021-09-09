import unittest

from utils.prioritizedMemory import prioritizedMemory


class MyTestCase(unittest.TestCase):
    def test_index_equal(self):
        self.memory = prioritizedMemory(10, 3, 0.5, 0.5)
        for i in range(5):
            self.memory.add(f'experience{i}', i)


if __name__ == '__main__':
    unittest.main()
