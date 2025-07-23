import numpy as np
import struct

def generate_random_heightmap(size=256, max_height=0.2, filename="random_terrain.raw"):
    """
    Generate a random heightmap and save it as a raw file.
    :param size: Size of the heightmap grid (e.g., 256x256)
    :param max_height: Maximum height variation (range between -max_height and +max_height)
    :param filename: The name of the output file
    """

    # Generate random terrain values between -max_height and +max_height
    heightmap = np.random.uniform(low=-max_height, high=max_height, size=(size, size))
    
    # Flatten the heightmap to a 1D array for saving
    heightmap_flat = heightmap.flatten()

    # Write the heightmap to a raw binary file
    with open(filename, "wb") as f:
        # Write the heightmap as 32-bit float values
        for value in heightmap_flat:
            f.write(struct.pack('f', value))
    print(f"Heightmap saved to {filename}")

# Generate a random heightmap and save it
generate_random_heightmap(size=256, max_height=0.2)
