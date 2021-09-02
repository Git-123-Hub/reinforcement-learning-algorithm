import unittest

from replayMemory import replayMemory

capacity, batch_size = 10, 4


def experience(i):
    """use string to represent experience tuple(state, action, reward, next_state, done)"""
    return f's{i}', f'a{i}', f'r{i}', f's_{i}', f'd{i}'


def experience_number(i):
    """use string to represent experience tuple(state, action, reward, next_state, done)"""
    return i, i * 1e1, i * 1e2, i * 1e2, i * 1e4


class test_replay_memory(unittest.TestCase):
    """Test the functionality of replayMemory. All the experience all sting to simplify"""

    def test_add(self):
        memory = replayMemory(capacity, batch_size)

        # the memory is not ready before the number of experience greater or equal to batch_size
        for i in range(1, batch_size):
            memory.add(experience(i))
            self.assertEqual(len(memory), i)
            self.assertEqual(memory.ready, False)

        # the memory is ready once the number of experience reaches the batch_size
        for i in range(batch_size, capacity + 1):
            memory.add(experience(i))
            self.assertEqual(len(memory), i)
            self.assertEqual(memory.ready, True)

        # if the memory is already full, add more experience will not change its length
        for i in range(capacity + 1, capacity * 2 + 1):
            memory.add(experience(i))
            self.assertEqual(len(memory), capacity)
            self.assertEqual(memory.ready, True)
            # the new experience will be add to the front of the memory
            self.assertEqual(memory[i % capacity - 1], experience(i))

    def test_sample(self):
        memory = replayMemory(capacity, batch_size)
        for i in range(capacity):
            memory.add(experience_number(i))
        for _ in range(5):
            # print(memory.sample())
            pass


if __name__ == '__main__':
    unittest.main()
