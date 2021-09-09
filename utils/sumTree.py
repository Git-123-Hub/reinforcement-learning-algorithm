############################################
# @Author: Git-123-Hub
# @Date: 2021/9/8
# @Description: implementation of data structure sumTree
############################################

import numpy as np


class sumTree:
    """
    A binary tree data structure where the value of a parent node is
    the sum of the values of its children
    """

    def __init__(self, capacity):
        """
        initial the tree with `capacity`
        :param capacity: number of leaf nodes
        """
        self.capacity = capacity

        # an array to store all the nodes(data) of the tree
        self.tree = np.zeros(2 * capacity - 1)
        # if a binary tree has n leaf nodes, then the total number of nodes of the tree is 2*n-1

        self._length = 0
        # current number of all the nodes added, should be smaller than capacity

        self._index = 0
        # used to indicate the position of newly add node
        # `self._index` is the index of all the leaf nodes, count from 0 to capacity-1

    # todo: test reset()
    def reset(self):
        self.tree = np.zeros(2 * self.capacity - 1)
        self._length = 0
        self._index = 0

    def add(self, value):
        """
        add a new node to the tree with the value provided
        :param value: the value of the new node
        """
        index = self._index + self.capacity - 1
        # convert index of the leaf nodes to index of the whole tree
        # i.e. consider all the leaf nodes as another list: leaf_nodes,
        # then leaf_nodes[self._index] and self.tree[index] represent the same node

        # propagate value change to the whole tree
        self._propagate(index, value - self.tree[index])
        # update value of this node
        self.tree[index] = value

        self._index += 1
        if self._index >= self.capacity: self._index = 0  # add new node from start
        if self._length < self.capacity: self._length += 1  # length should be smaller than capacity

    def _propagate(self, index, delta):
        """
        from the node in `index`, update all its parents' value
        :param index: the index of the child node(index of the tree)
        :param delta: value change of the child node, will be propagated to all its parents
        """
        parent_index = (index - 1) // 2
        self.tree[parent_index] += delta
        if parent_index != 0:  # propagate until the root node
            self._propagate(parent_index, delta)

    def update(self, index, value):
        """
        update the value of the leaf node and propagate the change to all its parents
        :param index: the index of the node to be updated(index of the leaf nodes)
        :param value: the new value of the node
        """
        index += self.capacity - 1  # convert index of the leaf nodes to index of the tree
        delta = value - self.tree[index]
        self.tree[index] = value
        self._propagate(index, delta)

    def find(self, value):
        """
        find the first node in the tree whose value is greater than `value`
        :param value: value to be compared
        :return: index of the found node(index of all the leaf nodes), value of the found node
        """
        index = self._compare(0, value)  # index of the tree
        found_value = self.tree[index]
        index -= self.capacity - 1  # convert to index of the leaf nodes
        return index, found_value

    def _compare(self, index, value):
        """
        compare the `value` with left and right child of the node `index`,
        if the `value` is less than the value of left child,
        go left and compare `value` with the children of the left child
        else subtract value of left child from `value`,
        then compare the new value with the children of the right child
        comparing will continue until `index` is the index of a leaf node
        :param index: the node's index whose children will be compared with `value`
        :param value: the value to be compared
        :return: the index of the leaf node
        """
        left_child_index = 2 * index + 1
        if left_child_index >= len(self.tree):  # left child is leaf node
            return index

        right_child_index = left_child_index + 1
        if value <= self.tree[left_child_index]:
            return self._compare(left_child_index, value)
        else:
            return self._compare(right_child_index, value - self.tree[left_child_index])

    @property
    def sum(self):
        """return the sum of values of all the leaf nodes, which is also the value of root node"""
        return self.tree[0]

    def __len__(self):
        """number of all the leaf nodes that have been added"""
        return self._length

    def __getitem__(self, item):
        return self.tree[item + self.capacity - 1]
