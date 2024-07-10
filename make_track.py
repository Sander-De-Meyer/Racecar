import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Race Track")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
gray = (169, 169, 169)
green = (0, 255, 0)

# Track dimensions
track_width = 40
outer_margin = 50

# Draw track
def draw_track():
    screen.fill(green)
    # Outer boundary
    pygame.draw.rect(screen, black, (outer_margin, outer_margin, width - 2 * outer_margin, height - 2 * outer_margin), track_width)
    # Inner boundary
    pygame.draw.rect(screen, green, (outer_margin + track_width, outer_margin + track_width, width - 2 * (outer_margin + track_width), height - 2 * (outer_margin + track_width)))

def main():
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw_track()
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
