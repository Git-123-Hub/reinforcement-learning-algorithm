import random
import unittest

import numpy as np

from utils.sumTree import sumTree


def get_tree():
    """construct a tree for testing"""
    tree = sumTree(4)
    tree.add(0.1)
    tree.add(0.2)
    tree.add(0.3)
    tree.add(0.4)
    return tree


class MyTestCase(unittest.TestCase):
    """test the function of class sumTree with a simple example:
               1.0
        0.3          0.7
    0.1    0.2   0.3    0.4
    """

    def test_add_within_capacity(self):
        """test function add, len, sum when add within capacity"""
        # construct a new tree for testing add
        tree = sumTree(4)
        tree_sum = 0
        self.assertAlmostEqual(tree_sum, tree.sum)

        tree.add(0.1)
        tree_sum += 0.1
        self.assertEqual(len(tree), 1)
        self.assertAlmostEqual(tree_sum, tree.sum)
        np.testing.assert_almost_equal(tree.tree, [0.1, 0.1, 0, 0.1, 0, 0, 0])

        tree.add(0.2)
        tree_sum += 0.2
        self.assertEqual(len(tree), 2)
        self.assertAlmostEqual(tree_sum, tree.sum)
        np.testing.assert_almost_equal(tree.tree, [0.3, 0.3, 0, 0.1, 0.2, 0, 0])

        tree.add(0.3)
        tree_sum += 0.3
        self.assertEqual(len(tree), 3)
        self.assertAlmostEqual(tree_sum, tree.sum)
        np.testing.assert_almost_equal(tree.tree, [0.6, 0.3, 0.3, 0.1, 0.2, 0.3, 0])

        tree.add(0.4)
        tree_sum += 0.4
        self.assertEqual(len(tree), 4)
        self.assertAlmostEqual(tree_sum, tree.sum)
        np.testing.assert_almost_equal(tree.tree, [1, 0.3, 0.7, 0.1, 0.2, 0.3, 0.4])

    def test_add_exceed_capacity(self):
        """test function add, len, sum when add exceed capacity"""
        tree = get_tree()
        tree_sum = tree.sum

        # rewrite the first element
        tree.add(0.5)
        tree_sum += 0.5 - 0.1
        self.assertEqual(len(tree), 4)  # length will not change
        np.testing.assert_almost_equal(tree.tree, [1.4, 0.7, 0.7, 0.5, 0.2, 0.3, 0.4])

        # rewrite the second element
        tree.add(0.6)
        tree_sum += 0.6 - 0.2
        self.assertEqual(len(tree), 4)
        np.testing.assert_almost_equal(tree.tree, [1.8, 1.1, 0.7, 0.5, 0.6, 0.3, 0.4])

        # rewrite the third element
        tree.add(0.7)
        tree_sum += 0.7 - 0.3
        self.assertEqual(len(tree), 4)
        np.testing.assert_almost_equal(tree.tree, [2.2, 1.1, 1.1, 0.5, 0.6, 0.7, 0.4])

        # rewrite the fourth element
        tree.add(0.1)
        tree_sum += 0.1 - 0.4
        self.assertEqual(len(tree), 4)
        np.testing.assert_almost_equal(tree.tree, [1.9, 1.1, 0.8, 0.5, 0.6, 0.7, 0.1])

    @staticmethod
    def test_update():
        tree = get_tree()
        tree.update(3, 0.5)
        np.testing.assert_almost_equal(tree.tree, [1.4, 0.7, 0.7, 0.5, 0.2, 0.3, 0.4])
        tree.update(4, 1.3)
        np.testing.assert_almost_equal(tree.tree, [2.5, 1.8, 0.7, 0.5, 1.3, 0.3, 0.4])
        tree.update(5, 0.8)
        np.testing.assert_almost_equal(tree.tree, [3, 1.8, 1.2, 0.5, 1.3, 0.8, 0.4])
        tree.update(6, 0.1)
        np.testing.assert_almost_equal(tree.tree, [2.7, 1.8, 0.9, 0.5, 1.3, 0.8, 0.1])

    def test_find(self):
        tree = get_tree()
        leaf_nodes = tree.tree[-tree.capacity:]
        leaf_nodes_cumsum = np.cumsum(leaf_nodes)
        leaf_nodes_cumsum = list(np.around(leaf_nodes_cumsum, decimals=1))
        leaf_nodes_cumsum.insert(0, 0)
        for _ in range(10):
            self.assertEqual(tree.find(random.uniform(leaf_nodes_cumsum[0], leaf_nodes_cumsum[1])), (3, leaf_nodes[0]))
            self.assertEqual(tree.find(random.uniform(leaf_nodes_cumsum[1], leaf_nodes_cumsum[2])), (4, leaf_nodes[1]))
            self.assertEqual(tree.find(random.uniform(leaf_nodes_cumsum[2], leaf_nodes_cumsum[3])), (5, leaf_nodes[2]))
            self.assertEqual(tree.find(random.uniform(leaf_nodes_cumsum[3], leaf_nodes_cumsum[4])), (6, leaf_nodes[3]))

    def test_sum(self):
        """test tree.sum is the value of all the value of leaf nodes"""
        tree = get_tree()
        self.assertAlmostEqual(tree.sum, sum(tree.tree[-tree.capacity:]))

    def test_tree_define(self):
        """
        test whether the tree still satisfy its definition
        i.e. the value of parent node equals the sum of the value of the child nodes
        """
        # a binary tree with n leaf nodes will have 2*n-1 nodes in total
        # so we walk through all the other n-1 internal nodes to ensure that it has children
        tree = get_tree()
        for index in range(tree.capacity - 1):
            left_child_index = 2 * index + 1
            right_child_index = left_child_index + 1
            self.assertAlmostEqual(tree.tree[index], tree.tree[left_child_index] + tree.tree[right_child_index])
            print(f'index: {index}, node value: {tree.tree[index]}, '
                  f'left: {tree.tree[left_child_index]}, right: {tree.tree[right_child_index]}')


if __name__ == '__main__':
    unittest.main()
