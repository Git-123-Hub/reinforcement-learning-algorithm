import unittest

import numpy as np

from utils import prioritizedMemory

capacity, batch_size, alpha, beta = 10, 3, 0.5, 0.5


class MyTestCase(unittest.TestCase):
    def test_add(self):
        memory = prioritizedMemory(capacity, batch_size, alpha, beta)
        # add within capacity
        for i in range(capacity):
            memory.add((f'exp{i}', i))
            self.assertEqual(len(memory), i + 1)
            self.assertEqual(memory[i], (f'exp{i}', max(i, memory.max_priority) ** alpha))

        # add exceed capacity
        for i in range(capacity):
            memory.add((f'exp{i + capacity}', i + capacity))
            self.assertEqual(len(memory), capacity)
            self.assertEqual(memory[i], (f'exp{i + capacity}', max(i + capacity, memory.max_priority) ** alpha))

    def test_update(self):
        # construct a memory
        memory = prioritizedMemory(capacity, batch_size, alpha, beta)
        for i in range(capacity):
            memory.add((f'exp{i}', i))
        # setup sample_index and the new td_errors
        memory.sample_index = [i for i in range(capacity)]
        td_errors = np.random.random(capacity)
        memory.update(td_errors)
        for i in range(capacity):
            self.assertAlmostEqual(memory.priority[i], abs(td_errors[i]) ** alpha)

    def test_reset(self):
        memory = prioritizedMemory(capacity, batch_size, alpha, beta)
        # add within capacity
        for i in range(capacity):
            memory.add((f'exp{i}', i))
        memory.reset()
        self.assertEqual(len(memory.priority), 0)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.max_priority, 1)
        self.assertEqual(memory.ready, False)


if __name__ == '__main__':
    unittest.main()
