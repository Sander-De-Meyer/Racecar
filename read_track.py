from PIL import Image
import numpy as np


def jpg_to_grayscale_coordinates(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Convert image to grayscale (if not already in grayscale)
    img_gray = img.convert('L')

    # Get image size
    width, height = img_gray.size

    print(f"size = {width}, {height}")
    # List to store coordinates and grayscale values
    coordinates = []
    
    # Iterate through each pixel and store coordinates and grayscale value
    for y in range(height):
        for x in range(width):
            # Get grayscale value at (x, y)
            grayscale_value = img_gray.getpixel((x, y))
            # Append coordinates (x, y) and grayscale value to the list
            coordinates.append(((x, y), grayscale_value))


    coords = np.zeros((width, height))
    for y in range(height):
        for x in range(width):
            coords[x,y] = img_gray.getpixel((x, y))

    return coordinates, coords


# Example usage:
image_path = 'track2.png'
coordinates, coords = jpg_to_grayscale_coordinates(image_path)

print(coordinates[-1])

print(coords[:200,395:405])
