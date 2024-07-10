import pygame
from PIL import Image
import numpy as np

class Track:
    def __init__(self, image_path):
        self.image_path = image_path

        self.width, self.height, self.boundaries, self.track = self.get_boundaries()

        self.image = pygame.image.load(image_path)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.image, (0, 0))

        print(f"boundaries = \n {self.boundaries}")

    def redraw(self):
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.image, (0, 0))

    def get_boundaries(self):
        # Open the image file
        img = Image.open(self.image_path)

        # Convert image to grayscale (if not already in grayscale)
        img_gray = img.convert('L')

        # Get image size
        width, height = img_gray.size

        boundaries = []
        track = np.zeros((width, height))
        for y in range(height):
            for x in range(width):
                if img_gray.getpixel((x, y)) == 0:
                    boundaries.append([x, y])
                    track[x, y] = 1
        return width, height, boundaries, track