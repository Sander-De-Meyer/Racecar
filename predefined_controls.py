import matplotlib.pyplot as plt
from Racecar.car_automatic import Car

car1 = Car()

MAX_ITERATIONS = 1000

controls = [(1*(i<10), 0.01*(i < 20)) for i in range(MAX_ITERATIONS)]

positions = []

print(car1)
for i in range(MAX_ITERATIONS):
    positions.append(car1.update(controls[i]))

plt.plot([pos[0] for pos in positions],[pos[1] for pos in positions])
plt.show()