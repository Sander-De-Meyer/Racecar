import pygame
import sys
import numpy as np

from car import Car
from track import Track

MAX_ITERATIONS = 10000

controls = [[0.05*(i < 100), 0] for i in range(MAX_ITERATIONS)]

def main():
    pygame.init()
    track = Track("track2.png")
    # screen = pygame.display.set_mode((800, 800))
    clock = pygame.time.Clock()
    car = Car(400, 80, width = 40, length = 40, theta = 0, inv_radius = 0.0, manual = False)
    # track_image = pygame.image.load("track2.png")

    for i in range(MAX_ITERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # screen.fill((0, 0, 0))
        # screen.blit(track_image, (0, 0))  # Draw the track
        track.redraw()
        car.update(controls)
        car.draw(track)
        pygame.display.flip()
        clock.tick(60)

        if (car.check_collision_efficient(track)):
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    main()
