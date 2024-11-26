import os
import tifffile as tiff
import numpy as np

grid = 3 #it's a 3by3 grid


# Overlap size (in pixels)
overlap = 30

# Initialize a list to store the tiles
tiles_grid = [[None for _ in range(grid)] for _ in range(grid)]

# Loop through the grid positions and load the corresponding tiles
for i in range(grid):  # Rows
    for j in range(grid):  # Columns
        # Construct the filename for each segmented tile
        tile_file_name = f'output_{i}_{j}.tiff'
        tile_path = os.path.join('/home/arawa/Segmentation_shabaz_FV/inference_3x3tiles', tile_file_name)
        
        # Check if the tile file exists
        if not os.path.isfile(tile_path):
            print(f"Segmented tile file {tile_file_name} not found. Skipping.")
            continue
        
        # Load the segmented tile
        tile_image = tiff.imread(tile_path)  # Shape: (Z, H, W)
        
        # Store the tile in the grid
        tiles_grid[i][j] = tile_image

# Verify that all tiles have been loaded
for i in range(grid):
    for j in range(grid):
        if tiles_grid[i][j] is None:
            raise ValueError(f"Tile at position ({i}, {j}) is missing. Cannot proceed with stitching.")

# Get tile dimensions
tile_depth, tile_height, tile_width = tiles_grid[0][0].shape

# Calculate the dimensions of the stitched image
stitched_height = tile_height * grid - overlap * (grid - 1)
stitched_width = tile_width * grid - overlap * (grid - 1)

# Initialize a list to store the stitched Z-slices
stitched_slices = []

# Function to blend two overlapping regions
def blend_overlap(region1, region2, overlap_size, axis):
    alpha = np.linspace(0, 1, overlap_size)
    if axis == 0:  # Vertical blending
        alpha = np.tile(alpha[:, np.newaxis], (1, region1.shape[1]))
    else:  # Horizontal blending
        alpha = np.tile(alpha, (region1.shape[0], 1))
    blended_region = region1 * (1 - alpha) + region2 * alpha
    return blended_region

# Loop through each Z-slice
for z_index in range(tile_depth):
    stitched_slice = np.zeros((stitched_height, stitched_width), dtype=tiles_grid[0][0].dtype)
    for i in range(grid):
        for j in range(grid):
            # Extract the Z-slice from the current tile
            tile_slice = tiles_grid[i][j][z_index]
            # Calculate the position to place the tile slice in the stitched image
            start_row = i * (tile_height - overlap)
            start_col = j * (tile_width - overlap)
            end_row = start_row + tile_height
            end_col = start_col + tile_width
            
            # Place the tile slice in the stitched image with blending
            if i > 0:  # Blend with the tile above
                overlap_region = blend_overlap(stitched_slice[start_row:start_row+overlap, start_col:end_col], tile_slice[:overlap, :], overlap, axis=0)
                stitched_slice[start_row:start_row+overlap, start_col:end_col] = overlap_region
                stitched_slice[start_row+overlap:end_row, start_col:end_col] = tile_slice[overlap:, :]
            elif j > 0:  # Blend with the tile to the left
                overlap_region = blend_overlap(stitched_slice[start_row:end_row, start_col:start_col+overlap], tile_slice[:, :overlap], overlap, axis=1)
                stitched_slice[start_row:end_row, start_col:start_col+overlap] = overlap_region
                stitched_slice[start_row:end_row, start_col+overlap:end_col] = tile_slice[:, overlap:]
            else:  # No blending needed for the first tile
                stitched_slice[start_row:end_row, start_col:end_col] = tile_slice

    stitched_slices.append(stitched_slice)

# Stack all Z-slices to form the full 3D image
stitched_image = np.array(stitched_slices)

# Save the stitched image as a TIFF file
tiff.imwrite('/home/arawa/Segmentation_shabaz_FV/stitched_segmented_image_FV3_6x6.tiff', stitched_image, dtype=stitched_image.dtype)

print("Stitched segmented image saved successfully")