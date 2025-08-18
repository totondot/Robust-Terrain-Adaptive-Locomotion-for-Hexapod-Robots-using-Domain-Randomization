import numpy as np
import os
from perlin_noise import PerlinNoise

# Set a random seed for reproducibility
np.random.seed(42)

def generate_and_save_heightmap(filename, width=256, length=256, scale=1.0):
    """
    Generates a heightmap using Perlin noise and saves it as a .raw file.
    """
    noise = PerlinNoise(octaves=4, seed=np.random.randint(0, 10000))
    xpix, ypix = width, length
    heightmap = np.array([[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)])
    
    # Scale the heightmap to a reasonable range
    heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
    heightmap = (heightmap * scale).astype(np.float16)

    with open(filename, 'wb') as f:
        f.write(heightmap.tobytes())

if __name__ == "__main__":
    heightmaps_folder = os.path.join(os.path.dirname(__file__), "heightmaps")
    if not os.path.exists(heightmaps_folder):
        os.makedirs(heightmaps_folder)
        print(f"Created folder: {heightmaps_folder}")
    
    num_maps_to_generate = 5
    for i in range(num_maps_to_generate):
        filename = os.path.join(heightmaps_folder, f"heightmap_{i}.raw")
        generate_and_save_heightmap(filename, scale=1.0)
        print(f"Generated and saved: {filename}")
