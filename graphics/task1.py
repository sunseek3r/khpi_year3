import cv2
import numpy as np
from dataclasses import dataclass
import cl.runtime as rt
import matplotlib.pyplot as plt
import time

S = [1]
B = [1]
C = 5

DEAD_COLOR = np.array([0.0, 255.0, 0.0])
FULL_HEALTH_COLOR = np.array([255.0, 0, 0])


def get_color(hp):
    alpha = 1.0 * hp / C

    rgb = np.abs(DEAD_COLOR + (FULL_HEALTH_COLOR - DEAD_COLOR) * alpha)
    return rgb


@dataclass
class Map:
    field: np.array = rt.class_field()

    rows: int = rt.class_field()

    cols: int = rt.class_field()

    def __init__(self, path_to_map=None):
        if path_to_map is None:
            self.field = np.random.binomial(1, 0.25, size=(20, 20)) * C
            self.rows = 20
            self.cols = 20
        else:
            self.field = self.read(path_to_map)

        self.drawable = True

    def read(self, path):
        field = np.full((4, 4), 0)
        self.rows = field.shape[0]
        self.cols = field.shape[1]
        return field

    def cycle(self):
        field = self.field
        dx = [-1, 0, 1]
        dy = [-1, 0, 1]

        for i in range(self.rows):
            for j in range(self.cols):
                alive = 0
                for xx in dx:
                    for yy in dy:
                        if self.rows > i + xx >= 0 and self.cols > j + yy >= 0:
                            if field[i + xx][j + yy] == C:
                                alive += 1
                if self.field[i][j] == C and alive not in S:
                    self.field[i][j] -= 1
                elif 0 < self.field[i][j] < C:
                    self.field[i][j] -= 1
                elif self.field[i][j] == 0 and alive in B:
                    self.field[i][j] = C

        self.drawable = np.any(self.field > 0)

    def update_field(self):
        field = list(self.field)

        rgb_field = np.array([list(map(get_color, r)) for r in field])

        image = cv2.resize(rgb_field, (self.rows*50, self.cols*50), interpolation=cv2.INTER_NEAREST)
        print(rgb_field)
        cv2.imshow("Automat", image)
        cv2.waitKey(1)
        self.cycle()


def main():
    field = Map()

    while True:
        field.update_field()
        print(field.field)
        time.sleep(1)
        if not field.drawable:
            break


if __name__ == '__main__':
    main()

