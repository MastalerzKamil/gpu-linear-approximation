import numpy as np


class FileReader:
    def __init__(self):
        self.x_vector = []
        self.y_vector = []

    def read_from_file(self, filename):
        file = open(filename)
        for line in file:
            fields = line.split(" ")
            self.x_vector.append(float(fields[0]))
            self.y_vector.append(float(fields[1]))

    def get_x_vector(self):
        return self.x_vector

    def get_y_vector(self):
        return self.y_vector
