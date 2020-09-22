import numpy as np


class IGate:
    def __init__(self, target):
        self.target = target

    def to_matrix(self):
        return np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

    def __str__(self):
        return "I({})".format(self.target)


class RXGate:
    def __init__(self, theta, target):
        self.theta = theta
        self.target = target

    def to_matrix(self):
        return np.array([
            [np.cos(self.theta/2), -1j * np.sin(self.theta/2)],
            [-1j * np.sin(self.theta/2), np.cos(self.theta/2)],
        ])

    def __str__(self):
        return "RX({}, {})".format(self.theta, self.target)


class RZGate:
    def __init__(self, theta, target):
        self.theta = theta
        self.target = target

    def to_matrix(self):
        return np.array([
            [np.exp(-1j * self.theta/2), 0],
            [0, np.exp(1j * self.theta/2)],
        ])

    def __str__(self):
        return "RZ({}, {})".format(self.theta, self.target)


class XGate:
    def __init__(self, target):
        self.target = target

    def to_matrix(self):
        return np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])

    def __str__(self):
        return "X({})".format(self.target)


class ZGate:
    def __init__(self, target):
        self.target = target

    def to_matrix(self):
        return np.array([
            [1.0, 0.0],
            [0.0, -1.0],
        ])

    def __str__(self):
        return "Z({})".format(self.target)


class CZGate:
    def __init__(self, control, target):
        self.control = control
        self.target = target

    def to_matrix(self):
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0]
        ])

    def __str__(self):
        return "CZ({}, {})".format(self.control, self.target)


class RYGate:
    def __init__(self, theta, target):
        self.theta = theta
        self.target = target

    def to_matrix(self):
        return np.array([
            [np.cos(self.theta/2), -np.sin(self.theta/2)],
            [np.sin(self.theta/2), np.cos(self.theta/2)],
        ])

    def __str__(self):
        return "RY({}, {})".format(self.theta, self.target)


class YGate:
    def __init__(self, target):
        self.target = target

    def to_matrix(self):
        return np.array([
            [0.0, -1j],
            [1j, 0.0],
        ])

    def __str__(self):
        return "Y({})".format(self.target)


class HGate:
    def __init__(self, target):
        self.target = target

    def to_matrix(self):
        return 1 / np.sqrt(2) * np.array([
            [1.0, 1.0],
            [1.0, -1.0],
        ])

    def __str__(self):
        return "H({})".format(self.target)


class CNOTGate:
    def __init__(self, control, target):
        self.control = control
        self.target = target

    def to_matrix(self):
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

    def __str__(self):
        return "CNOT({}, {})".format(self.control, self.target)
