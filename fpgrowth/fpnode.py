#!/usr/bin/env python
# coding=utf-8

class FPNode(object):
    def __init__(self, value, count, parent):
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value):
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child

