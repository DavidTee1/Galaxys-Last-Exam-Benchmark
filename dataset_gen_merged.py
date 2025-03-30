import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import itertools
import os
import json
import math
import datetime as dt
import copy
from collections import deque
import time

#############################
# Task: Line Intersections (LI)
#############################

def generate_mixed_line(x):
    """
    Generate a single composite line over x.
    The line is divided into a random number of segments,
    and each segment is randomly generated as linear,
    oscillatory, or piecewise.
    """
    n_segments = random.randint(2, 4)
    indices = sorted(random.sample(range(1, len(x)-1), n_segments-1))
    indices = [0] + indices + [len(x)-1]
    y = np.empty_like(x)
    y_start = random.uniform(-10, 10)
    for i in range(len(indices)-1):
        start_idx = indices[i]
        end_idx = indices[i+1]
        x_seg = x[start_idx:end_idx+1]
        y_end = random.uniform(-10, 10)
        method = random.choice(['linear', 'oscillatory', 'piecewise'])
        if method == 'linear':
            y_seg = np.linspace(y_start, y_end, len(x_seg))
        elif method == 'oscillatory':
            t = (x_seg - x_seg[0]) / (x_seg[-1] - x_seg[0]) if (x_seg[-1] - x_seg[0]) != 0 else np.zeros_like(x_seg)
            amplitude = random.uniform(0.5, 2.0)
            freq = random.uniform(1.0, 4.0)
            if np.all(t == t[0]):
                 f_t = t
            else:
                 f_t = t + amplitude * np.sin(2 * np.pi * freq * t) * t * (1 - t)

            y_seg = y_start + (y_end - y_start) * f_t
            y_seg = np.clip(y_seg, -10, 10)
        elif method == 'piecewise':
            if len(x_seg) < 3:
                y_seg = np.linspace(y_start, y_end, len(x_seg))
            else:
                n_possible = len(x_seg) - 2
                if n_possible >= 2:
                    n_control = random.randint(2, min(4, n_possible))
                else:
                    n_control = 0

                if n_control > 0:
                    if len(list(x_seg[1:-1])) >= n_control:
                         x_control = sorted(random.sample(list(x_seg[1:-1]), n_control))
                    else: 
                         x_control = sorted(list(x_seg[1:-1]))

                    x_points = [x_seg[0]] + x_control + [x_seg[-1]]
                    y_points = [y_start] + [random.uniform(-10, 10) for _ in range(n_control)] + [y_end]
                    y_seg = np.interp(x_seg, x_points, y_points)
                else:
                    y_seg = np.linspace(y_start, y_end, len(x_seg))
        y[start_idx:end_idx+1] = y_seg
        y_start = y_end
    return y

def count_intersections(x, y1, y2):
    """
    Count the number of intersections between two curves y1 and y2.
    """
    intersections = 0
    diff = y1 - y2
    sign_diff = np.sign(diff)
    for i in range(len(x)-1):
        if sign_diff[i] * sign_diff[i+1] < 0:
            intersections += 1
        elif sign_diff[i] == 0:
             # Ensure it's a crossing point, not just touching
             if i > 0 and i < len(sign_diff) - 1:
                 if sign_diff[i-1] * sign_diff[i+1] < 0:
                      intersections += 1
             # Handle start/end cases if needed (optional, based on definition)
             # elif i == 0 and len(sign_diff) > 1 and sign_diff[i+1] != 0:
             #    intersections +=1 # Count if starts at zero and then moves away
             # elif i == len(sign_diff) - 2 and sign_diff[i] != 0: # i is the index before the last element
             #    # check if the last point is zero and the previous is non-zero
             #    if sign_diff[i+1] == 0 and sign_diff[i] != 0:
             #         intersections += 1


    # Simpler approach (original): Count sign changes and explicit zeros
    # intersections = 0
    # diff = y1 - y2
    # for i in range(len(x)-1):
    #     if diff[i] == 0:
    #         # Avoid double counting if zero spans multiple points
    #         if i == 0 or diff[i-1] != 0:
    #             intersections += 1
    #     elif diff[i] * diff[i+1] < 0:
    #         intersections += 1
    return intersections


def generate_li_data(num_instances, output_dir, start_index):
    """
    Generates images with mixed lines and computes intersections.
    Returns metadata including intersection counts.
    """
    li_dir = os.path.join(output_dir, "li")
    os.makedirs(li_dir, exist_ok=True)
    results = []
    x = np.linspace(-10, 10, 400)
    nlines = 3 # Fixed number of lines as per my original script, can modify
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (nlines - 1 if nlines > 1 else 1)) for i in range(nlines)]

    for i in range(num_instances):
        img_idx = start_index + i
        plt.figure(figsize=(6, 6))
        lines_data = []
        for line_idx in range(nlines):
            y = generate_mixed_line(x)
            lines_data.append(y)
            plt.plot(x, y, color=colors[line_idx], linewidth=2)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xticks([])
        plt.yticks([])
        image_filename = f"li_{img_idx}.png"
        image_path = os.path.join(li_dir, image_filename)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        intersections_info = {}
        for (l1_idx, y1), (l2_idx, y2) in itertools.combinations(enumerate(lines_data, start=1), 2):
            count = count_intersections(x, y1, y2)
            intersections_info[f"Line {l1_idx} & Line {l2_idx}"] = count

        results.append({
            "task_type": "LI",
            "instance_id": img_idx,
            "output_file": os.path.join("li", image_filename),
            "ground_truth": {
                "intersections": intersections_info,
                "num_lines": nlines
            }
        })
    print(f"Generated {num_instances} LI instances.")
    return results

#############################
# Task: Letter Frequency (LF)
#############################

LF_IMG_WIDTH, LF_IMG_HEIGHT = 465, 462
LF_VOWELS = "AEIOU"
LF_NUM_LETTERS_PER_IMAGE = 20 # fixed quantity of letters, feel free to modify

try:
    LF_FONT = ImageFont.truetype("arial.ttf", 40)
except IOError:
    try:
        LF_FONT = ImageFont.truetype("Arial.ttf", 40)
    except IOError:
        print("Warning: Arial font not found. Using default PIL font for LF task.")
        LF_FONT = ImageFont.load_default()


def boxes_overlap(box1, box2):
    """
    Check if two boxes (x1, y1, x2, y2) overlap.
    """
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    # need to make sure these don't overlap
    if x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1:
        return False
    return True

def generate_lf_image(image_path, num_letters):
    """
    Generate one image with randomly placed non-overlapping vowels.
    Returns the count of each vowel placed in this specific image.
    """
    img = Image.new("RGB", (LF_IMG_WIDTH, LF_IMG_HEIGHT), color="white")
    draw = ImageDraw.Draw(img)
    placed_boxes = []
    letters_placed = 0
    attempts = 0
    max_attempts = 2000 
    vowel_counter_this_image = {v: 0 for v in LF_VOWELS}

    while letters_placed < num_letters and attempts < max_attempts:
        attempts += 1
        letter = random.choice(LF_VOWELS)

        try:
            bbox = draw.textbbox((0, 0), letter, font=LF_FONT)
            letter_width = bbox[2] - bbox[0]
            letter_height = bbox[3] - bbox[1]
            if LF_IMG_WIDTH - letter_width <= 0 or LF_IMG_HEIGHT - letter_height <= 0:
                 print(f"Warning: Letter '{letter}' too large for image dimensions. Skipping.")
                 continue 
            x = random.randint(0, LF_IMG_WIDTH - letter_width -1) 
            y = random.randint(0, LF_IMG_HEIGHT - letter_height -1)
            draw_x = x - bbox[0]
            draw_y = y - bbox[1]
            new_box = (x, y, x + letter_width, y + letter_height)

        except AttributeError: # Fallback for older PIL versions or default font
             try:
                 letter_width, letter_height = draw.textsize(letter, font=LF_FONT)
                 if LF_IMG_WIDTH - letter_width <= 0 or LF_IMG_HEIGHT - letter_height <= 0:
                      print(f"Warning: Letter '{letter}' too large for image dimensions. Skipping.")
                      continue
                 x = random.randint(0, LF_IMG_WIDTH - letter_width -1)
                 y = random.randint(0, LF_IMG_HEIGHT - letter_height -1)
                 new_box = (x, y, x + letter_width, y + letter_height)
                 draw_x, draw_y = x, y 
             except AttributeError: # If textsize also fails (highly unlikely)
                  print("Error: Cannot determine text size. Skipping letter.")
                  continue

        is_overlapping = False
        for box in placed_boxes:
            if boxes_overlap(new_box, box):
                is_overlapping = True
                break

        if not is_overlapping:
            draw.text((draw_x, draw_y), letter, font=LF_FONT, fill="black")
            placed_boxes.append(new_box)
            vowel_counter_this_image[letter] += 1
            letters_placed += 1

    if letters_placed < num_letters:
        print(f"Warning: Only placed {letters_placed}/{num_letters} letters for {os.path.basename(image_path)} due to space/overlap constraints.")

    img.save(image_path)
    return vowel_counter_this_image

def generate_lf_data(num_instances, output_dir, start_index):
    """
    Generates images with randomly placed vowels.
    Returns metadata including vowel counts per image.
    """
    lf_dir = os.path.join(output_dir, "lf")
    os.makedirs(lf_dir, exist_ok=True)
    results = []

    for i in range(num_instances):
        img_idx = start_index + i
        image_filename = f"lf_{img_idx}.png"
        image_path = os.path.join(lf_dir, image_filename)

        vowel_counts = generate_lf_image(image_path, LF_NUM_LETTERS_PER_IMAGE)

        results.append({
            "task_type": "LF",
            "instance_id": img_idx,
            "output_file": os.path.join("lf", image_filename),
            "ground_truth": {
                "vowel_counts": vowel_counts,
                "total_vowels": sum(vowel_counts.values())
            }
        })
    print(f"Generated {num_instances} LF instances.")
    return results

#############################
# Task: Cube Counting (CC)
#############################

def generate_random_connected_cubes(num_cubes):
    """
    Generates a random connected cluster of `num_cubes` cubes
    in 3D, each at integer grid coordinates. Returns a list
    of (x, y, z) tuples.
    """
    directions = [(1,0,0), (-1,0,0),
                  (0,1,0), (0,-1,0),
                  (0,0,1), (0,0,-1)]
    cubes = [(0,0,0)]
    frontier = set([(dx, dy, dz) for dx, dy, dz in directions])
    occupied = set(cubes)

    while len(cubes) < num_cubes:
        if not frontier:
            # Should not happen if num_cubes > 1, but handle defensively
            print(f"Warning: Could only generate {len(cubes)} connected cubes.")
            break

        new_cube = random.choice(list(frontier))
        frontier.remove(new_cube)

        if new_cube in occupied:
             continue

        cubes.append(new_cube)
        occupied.add(new_cube)

        base_cube = new_cube
        for dx, dy, dz in directions:
            neighbor_cube = (base_cube[0] + dx,
                             base_cube[1] + dy,
                             base_cube[2] + dz)
            if neighbor_cube not in occupied:
                frontier.add(neighbor_cube)

    return cubes[:num_cubes]


def plot_cubes(cubes, filename):
    """
    Plots the cubes (x,y,z) as 3D bars and saves to `filename`.
    Axes and title are removed.
    """
    if not cubes:
        print(f"Warning: No cubes to plot for {filename}.")
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    xs = [c[0] for c in cubes]
    ys = [c[1] for c in cubes]
    zs = [c[2] for c in cubes]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(min(zs), max(zs))
    cmap = plt.get_cmap('viridis') # Or another cmap like 'cubehelix'

    for i, (x, y, z) in enumerate(cubes):
        color = cmap(norm(z))
        ax.bar3d(x - 0.5, y - 0.5, z - 0.5, 1, 1, 1, color=color, shade=True, alpha=0.8, edgecolor='k', linewidth=0.5)


    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center_x, center_y, center_z = np.mean(xs), np.mean(ys), np.mean(zs)
    max_range = np.array([max_x-min_x, max_y-min_y, max_z-min_z]).max() / 2.0 + 1.0 

    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)

    ax.set_axis_off() 
    # Optional: Adjust viewing angle
    # ax.view_init(elev=30, azim=45)

    plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_cc_data(num_instances, output_dir, start_index):
    """
    Generates 3D plots of connected cube clusters.
    Returns metadata including the number of cubes.
    """
    cc_dir = os.path.join(output_dir, "cc")
    os.makedirs(cc_dir, exist_ok=True)
    results = []

    for i in range(num_instances):
        img_idx = start_index + i
        # modify here for cube quantity
        num_cubes = random.randint(7, 11)
        cubes = generate_random_connected_cubes(num_cubes)
        actual_num_cubes = len(cubes)

        image_filename = f"cc_{img_idx}.png"
        image_path = os.path.join(cc_dir, image_filename)
        plot_cubes(cubes, image_path)

        results.append({
            "task_type": "CC",
            "instance_id": img_idx,
            "output_file": os.path.join("cc", image_filename),
            "ground_truth": {
                "cube_count": actual_num_cubes
            }
        })
    print(f"Generated {num_instances} CC instances.")
    return results

#############################
# Task: Wood Slide (WS) - Updated
#############################

WS_ROWS, WS_COLS = 5, 4
WS_COLORS = [ # 0 is empty, 1-8 are blocks
    None,      # Background/Empty
    "#FF5733", # Red
    "#33FF57", # Green
    "#3357FF", # Blue
    "#F1C40F", # Yellow
    "#9B59B6", # Purple
    "#1ABC9C", # Teal
    "#E67E22", # Orange
    "#2ECC71"  # Emerald
]

def ws_create_empty_grid():
    return [[0 for _ in range(WS_COLS)] for _ in range(WS_ROWS)]

def ws_generate_blocks():
    """
    Randomly choose one partition for 8 blocks that must cover 18 cells.
    Each configuration leaves exactly 2 empty cells on a 5x4 grid.
    Returns a list of tuples: (block_id, height, width)
    """
    partitions = [
        {'ones': 0, 'twos': 7, 'fours': 1}, # 0*1 + 7*2 + 1*4 = 18
        {'ones': 2, 'twos': 4, 'fours': 2}, # 2*1 + 4*2 + 2*4 = 18
        {'ones': 4, 'twos': 1, 'fours': 3}  # 4*1 + 1*2 + 3*4 = 18
    ]
    part = random.choice(partitions)
    blocks = []
    block_id = 1 
    for _ in range(part['ones']):
        blocks.append((block_id, 1, 1))
        block_id += 1
    for _ in range(part['twos']):
        if random.choice([True, False]):
            blocks.append((block_id, 1, 2)) # height=1, width=2
        else:
            blocks.append((block_id, 2, 1)) # height=2, width=1
        block_id += 1
    for _ in range(part['fours']):
        blocks.append((block_id, 2, 2)) # Always 2x2
        block_id += 1
    return blocks[:8]

def ws_can_place(grid, r, c, height, width):
    """Check if a block can be placed at (r, c)."""
    if r < 0 or r + height > WS_ROWS or c < 0 or c + width > WS_COLS:
        return False
    for i in range(r, r + height):
        for j in range(c, c + width):
            if grid[i][j] != 0: 
                return False
    return True

def ws_place_block(grid, r, c, height, width, block_id):
    """Place a block onto the grid."""
    for i in range(r, r + height):
        for j in range(c, c + width):
            grid[i][j] = block_id

def ws_remove_block(grid, block_id_to_remove):
     """ Sets all cells with the given block_id back to 0 (empty)."""
     for r in range(WS_ROWS):
          for c in range(WS_COLS):
               if grid[r][c] == block_id_to_remove:
                    grid[r][c] = 0

def ws_find_first_empty(grid):
    """Find the coordinates of the first empty cell (0)."""
    for r in range(WS_ROWS):
        for c in range(WS_COLS):
            if grid[r][c] == 0:
                return r, c
    return None 

def ws_backtrack_tiling_robust(grid, blocks_to_place, current_block_index):
    """ Robust backtracking: try placing the current block at any valid location. """
    if current_block_index == len(blocks_to_place):
         empty_count = sum(row.count(0) for row in grid)
         return empty_count == 2 # 20 total cells - 18 block cells = 2 empty

    block_id, height, width = blocks_to_place[current_block_index]
    for r in range(WS_ROWS):
        for c in range(WS_COLS):
            if grid[r][c] == 0:
                 if ws_can_place(grid, r, c, height, width):
                    ws_place_block(grid, r, c, height, width, block_id)
                    if ws_backtrack_tiling_robust(grid, blocks_to_place, current_block_index + 1):
                        return True
                    ws_remove_block(grid, block_id) 

    return False

def ws_generate_solved_configuration():
    """Generate a solved board configuration by tiling the grid with 8 wood blocks."""
    max_attempts = 100 
    for attempt in range(max_attempts):
        grid = ws_create_empty_grid()
        blocks = ws_generate_blocks()
        random.shuffle(blocks) 
        if ws_backtrack_tiling_robust(grid, blocks, 0):
            return grid, blocks
    raise ValueError("Failed to generate a valid WS tiling after multiple attempts.")

def ws_find_block_info(grid, block_id):
    """Find all cells and bounding box for a given block_id."""
    cells = []
    min_r, max_r = WS_ROWS, -1
    min_c, max_c = WS_COLS, -1
    found = False
    for r in range(WS_ROWS):
        for c in range(WS_COLS):
            if grid[r][c] == block_id:
                found = True
                cells.append((r, c))
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if not found:
        return None, None
    return cells, (min_r, max_r, min_c, max_c)

def ws_try_move(grid, block_id, direction):
    """
    Attempts to move the specified block in the given direction.
    Returns the new grid state if the move is valid, otherwise None.
    """
    cells, bbox = ws_find_block_info(grid, block_id)
    if not cells: return None
    min_r, max_r, min_c, max_c = bbox
    new_grid = [row[:] for row in grid]

    target_cells_valid = True
    target_coords = []

    if direction == "up":
        if min_r == 0: return None
        for r, c in cells:
            target_r, target_c = r - 1, c
            if not (0 <= target_r < WS_ROWS and 0 <= target_c < WS_COLS): return None 
            if grid[target_r][target_c] != 0 and grid[target_r][target_c] != block_id:
                target_cells_valid = False; break
            target_coords.append((target_r, target_c))
        if not target_cells_valid: return None
        for r, c in cells: new_grid[r][c] = 0
        for r, c in cells: new_grid[r - 1][c] = block_id

    elif direction == "down":
        if max_r == WS_ROWS - 1: return None
        for r, c in cells:
            target_r, target_c = r + 1, c
            if not (0 <= target_r < WS_ROWS and 0 <= target_c < WS_COLS): return None
            if grid[target_r][target_c] != 0 and grid[target_r][target_c] != block_id:
                target_cells_valid = False; break
            target_coords.append((target_r, target_c))
        if not target_cells_valid: return None
        for r, c in sorted(cells, key=lambda x: x[0], reverse=True): new_grid[r][c] = 0
        for r, c in sorted(cells, key=lambda x: x[0], reverse=True): new_grid[r + 1][c] = block_id

    elif direction == "left":
        if min_c == 0: return None
        for r, c in cells:
            target_r, target_c = r, c - 1
            if not (0 <= target_r < WS_ROWS and 0 <= target_c < WS_COLS): return None
            if grid[target_r][target_c] != 0 and grid[target_r][target_c] != block_id:
                target_cells_valid = False; break
            target_coords.append((target_r, target_c))
        if not target_cells_valid: return None

        for r, c in sorted(cells, key=lambda x: x[1], reverse=True): new_grid[r][c] = 0
        for r, c in sorted(cells, key=lambda x: x[1], reverse=True): new_grid[r][c - 1] = block_id

    elif direction == "right":
        if max_c == WS_COLS - 1: return None
        for r, c in cells:
            target_r, target_c = r, c + 1
            if not (0 <= target_r < WS_ROWS and 0 <= target_c < WS_COLS): return None
            if grid[target_r][target_c] != 0 and grid[target_r][target_c] != block_id:
                target_cells_valid = False; break
            target_coords.append((target_r, target_c))
        if not target_cells_valid: return None
        for r, c in sorted(cells, key=lambda x: x[1], reverse=True): new_grid[r][c] = 0
        for r, c in sorted(cells, key=lambda x: x[1], reverse=True): new_grid[r][c + 1] = block_id
    else:
        return None 

    return new_grid


def ws_get_valid_moves(grid):
    """Return a list of dicts {'grid': new_grid_state, 'block_id': moved_block, 'direction': move_dir}."""
    moves = []
    block_ids = sorted(list(set(cell for row in grid for cell in row if cell != 0))) 
    for bid in block_ids:
        for direction in ["up", "down", "left", "right"]:
            new_grid = ws_try_move(grid, bid, direction)
            if new_grid is not None:
                moves.append({'grid': new_grid, 'block_id': bid, 'direction': direction})
    return moves

def grid_to_tuple(grid):
    """Converts a list-of-lists grid to a hashable tuple-of-tuples."""
    return tuple(tuple(row) for row in grid)

def ws_generate_start_grid_bfs(solved_grid, min_moves, max_moves):
    """
    Generates a start grid configuration that is provably between min_moves and
    max_moves (inclusive) away from the solved_grid using BFS.

    Returns: (start_grid, actual_min_moves) or (None, -1) if no suitable grid found.
    """
    target_dist = random.randint(min_moves, max_moves)
    q = deque([(solved_grid, 0)]) 
    visited = {grid_to_tuple(solved_grid)} 
    target_distance_grids = []

    bfs_start_time = time.time()
    max_bfs_time = 30 

    while q:
        current_grid, current_dist = q.popleft()

        if time.time() - bfs_start_time > max_bfs_time:
            print(f"BFS Warning: Timeout ({max_bfs_time}s) reached finding distance {target_dist}. Returning best found so far.")
            if target_distance_grids:
                 chosen_grid = random.choice(target_distance_grids)
                 return chosen_grid, target_dist
            else: 
                 print("BFS Error: Timeout before finding any grid at target distance.")
                 return None, -1 

        if current_dist == target_dist:
            target_distance_grids.append(current_grid)
            continue 

        if current_dist > target_dist: 
            continue

        for move_info in ws_get_valid_moves(current_grid):
            neighbor_grid = move_info['grid']
            neighbor_tuple = grid_to_tuple(neighbor_grid)

            if neighbor_tuple not in visited:
                visited.add(neighbor_tuple)
                q.append((neighbor_grid, current_dist + 1))
                # If this neighbor *is* the target distance, store it immediately
                # This wasn't strictly needed because of the check at the start of the loop,
                # but doesn't hurt.
                # if current_dist + 1 == target_dist:
                #    target_distance_grids.append(neighbor_grid)


    if target_distance_grids:
        chosen_grid = random.choice(target_distance_grids)
        return chosen_grid, target_dist
    else:
        print(f"BFS Warning: Could not find any state exactly {target_dist} moves away. BFS explored {len(visited)} states.")
        return None, -1 

# --- Visualization ---

def ws_draw_board(ax, grid, x_offset):
    """Draws a single Klotski board state onto the provided axes."""
    padding = 0.05   
    edge_width = 1.5 

    processed_cells = set() 

    for r in range(WS_ROWS):
        for c in range(WS_COLS):
            if (r, c) in processed_cells:
                continue 

            block_id = grid[r][c]
            if block_id != 0: 
                block_cells = []
                q = deque([(r,c)])
                visited_bfs = set([(r,c)])

                while q:
                    curr_r, curr_c = q.popleft()
                    if grid[curr_r][curr_c] == block_id: 
                        block_cells.append((curr_r, curr_c))
                        processed_cells.add((curr_r, curr_c)) 

                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < WS_ROWS and 0 <= nc < WS_COLS and \
                               grid[nr][nc] == block_id and (nr, nc) not in visited_bfs:
                                visited_bfs.add((nr,nc))
                                q.append((nr,nc))
 
                if not block_cells: continue 

                min_r = min(br for br, bc in block_cells)
                max_r = max(br for br, bc in block_cells)
                min_c = min(bc for br, bc in block_cells)
                max_c = max(bc for br, bc in block_cells)

                rect_y = min_r + padding 
                rect_x = min_c + padding + x_offset 
                rect_height = (max_r - min_r + 1) - 2 * padding 
                rect_width = (max_c - min_c + 1) - 2 * padding  

                rect = patches.Rectangle(
                    (rect_x, rect_y),
                    rect_width, rect_height,
                    linewidth=edge_width,
                    edgecolor="white", 
                    facecolor=WS_COLORS[block_id] if 0 < block_id < len(WS_COLORS) else "#888888", 
                    zorder=5 
                )
                ax.add_patch(rect)


def ws_visualize_puzzle(ax, start_grid, solved_grid):
    """Sets up the axes and draws the start and solved boards side-by-side."""
    board_width_cells = WS_COLS
    board_height_cells = WS_ROWS
    gap_cells = 1.5 
    buffer = 0.5
    ax.set_xlim(-buffer, board_width_cells * 2 + gap_cells + buffer)
    label_space = 1.0
    ax.set_ylim(board_height_cells + buffer + label_space, -buffer) 

    ax.set_aspect("equal")
    ax.axis("off")

    bg_color = '#DDDDDD'
    edge_color = 'black'
    bg1 = patches.Rectangle((0, 0), board_width_cells, board_height_cells,
                            linewidth=1, edgecolor=edge_color, facecolor=bg_color, zorder=0)
    bg2 = patches.Rectangle((board_width_cells + gap_cells, 0), board_width_cells, board_height_cells,
                            linewidth=1, edgecolor=edge_color, facecolor=bg_color, zorder=0)
    ax.add_patch(bg1)
    ax.add_patch(bg2)

    ws_draw_board(ax, start_grid, x_offset=0)
    ws_draw_board(ax, solved_grid, x_offset=board_width_cells + gap_cells)

    label_y = board_height_cells + buffer + label_space / 2 
    ax.text(board_width_cells / 2, label_y, "Start", ha="center", va="center", fontsize=12)
    ax.text(board_width_cells + gap_cells + board_width_cells / 2, label_y, "Solved", ha="center", va="center", fontsize=12)


def generate_ws_data(num_instances, output_dir, start_index):
    """
    Generates pairs of Klotski-like puzzle boards (start and solved states).
    Start state is guaranteed to be N moves away from solved state (BFS).
    Returns metadata including configurations and minimum move count.
    """
    ws_dir = os.path.join(output_dir, "ws")
    os.makedirs(ws_dir, exist_ok=True)
    results = []
    min_shuffle_moves = 3
    max_shuffle_moves = 7 

    generated_count = 0
    max_gen_attempts = num_instances * 3 

    for attempt in range(max_gen_attempts):
        if generated_count >= num_instances:
            break 

        img_idx = start_index + generated_count 

        try:
            solved_grid, blocks_info = ws_generate_solved_configuration()

            start_grid, actual_min_moves = ws_generate_start_grid_bfs(
                solved_grid, min_shuffle_moves, max_shuffle_moves
            )

            if start_grid is None:
                print(f"WS Instance {img_idx}: BFS failed to find suitable start grid. Retrying generation...")
                continue 

            start_config_str = "\n".join("".join(map(str, row)) for row in start_grid)
            solved_config_str = "\n".join("".join(map(str, row)) for row in solved_grid)

            fig, ax = plt.subplots(figsize=(8, 6)) 
            ws_visualize_puzzle(ax, start_grid, solved_grid) 

            image_filename = f"ws_{img_idx}.png"
            image_path = os.path.join(ws_dir, image_filename)
            plt.savefig(image_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            results.append({
                "task_type": "WS",
                "instance_id": img_idx,
                "output_file": os.path.join("ws", image_filename),
                "ground_truth": {
                    "min_moves": actual_min_moves,
                    "start_configuration": start_config_str,
                    "solved_configuration": solved_config_str,
                    "blocks": [{'id': b[0], 'height': b[1], 'width': b[2]} for b in blocks_info]
                }
            })
            generated_count += 1 

        except ValueError as e:
            print(f"Error generating WS instance {img_idx} (ValueError in tiling/setup): {e}. Skipping/Retrying...")
            continue 
        except Exception as e:
            print(f"Unexpected error generating WS instance {img_idx}: {e}. Skipping/Retrying...")
            import traceback
            traceback.print_exc() 
            continue

    if generated_count < num_instances:
         print(f"WS Warning: Only generated {generated_count}/{num_instances} instances after {max_gen_attempts} attempts.")

    print(f"Generated {generated_count} WS instances.")
    return results



#############################
# Task: Analog Clock (AC)
#############################

def ac_draw_clock(fig, ax, time_str):
    """
    Draws an analog clock face on the given axes for the specified time.
    """
    try:
        hours, minutes = map(int, time_str.split(":"))
        hours = hours % 12 
        if hours == 0: hours = 12 
    except ValueError:
        print(f"Error: Invalid time format '{time_str}'. Expected HH:MM.")
        hours, minutes = 3, 0 

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    clock_face = plt.Circle((0, 0), 1.0, color="ivory", fill=True, zorder=0)
    ax.add_artist(clock_face)
    clock_border = plt.Circle((0, 0), 1.0, color="midnightblue", fill=False, linewidth=3, zorder=1)
    ax.add_artist(clock_border)

    for hour_num in range(1, 13):
        angle = math.radians(90 - hour_num * 30) 
        x = 0.85 * math.cos(angle)
        y = 0.85 * math.sin(angle)
        ax.text(x, y, str(hour_num), ha="center", va="center", fontsize=16, fontweight="bold", color="darkslategrey", zorder=2)

    for i in range(60):
        angle = math.radians(90 - i * 6)
        x_start = math.cos(angle)
        y_start = math.sin(angle)
        if i % 5 == 0: 
            x_end = 0.90 * x_start
            y_end = 0.90 * y_start
            ax.plot([x_end, x_start], [y_end, y_start], color="silver", linewidth=2.5, zorder=1, solid_capstyle="round")
        else: 
            x_end = 0.95 * x_start
            y_end = 0.95 * y_start
            ax.plot([x_end, x_start], [y_end, y_start], color="lightgrey", linewidth=1, zorder=1, solid_capstyle="round")

    hour_angle_deg = (hours + minutes / 60.0) * 30
    hour_angle_rad = math.radians(90 - hour_angle_deg)
    ax.plot([0, 0.5 * math.cos(hour_angle_rad)],
            [0, 0.5 * math.sin(hour_angle_rad)],
            color="darkred", linewidth=7, solid_capstyle="round", zorder=3)

    minute_angle_deg = minutes * 6
    minute_angle_rad = math.radians(90 - minute_angle_deg)
    ax.plot([0, 0.8 * math.cos(minute_angle_rad)],
            [0, 0.8 * math.sin(minute_angle_rad)],
            color="dodgerblue", linewidth=4, solid_capstyle="round", zorder=4)

    center_pin = plt.Circle((0, 0), 0.05, color="navy", fill=True, zorder=5)
    ax.add_artist(center_pin)


def ac_calculate_time(current_hour, current_minute, delta_hour, delta_minute):
    """Calculates time after adding/subtracting a delta."""
    start_dt = dt.datetime(2000, 1, 1, current_hour, current_minute)
    time_delta = dt.timedelta(hours=delta_hour, minutes=delta_minute)
    result_dt = start_dt + time_delta

    result_hour = result_dt.hour
    result_minute = result_dt.minute

    hour_12 = result_hour % 12
    if hour_12 == 0:
        hour_12 = 12 

    return f"{hour_12}:{result_minute:02d}"

def ac_create_question(current_time_str, mode):
    """Generates a question based on the mode (past or future time)."""
    names = [
        "Emily", "Jacob", "Hannah", "Michael", "Madison", "Matthew", "Ashley", "Joshua", "Sarah", "Chris",
        "Alexis", "Nicholas", "Samantha", "Andrew", "Jessica", "Joseph", "Elizabeth", "Daniel", "Taylor", "Tyler",
        "Olivia", "Ethan", "Sophia", "Logan", "Isabella", "Lucas", "Mia", "Jackson", "Ava", "Aiden"
    ]
    person = random.choice(names)
    delta_hour = random.randint(0, 3)
    delta_minute = random.randint(1, 59) 

    parts = []
    if delta_hour > 0:
        parts.append(f"{delta_hour} hour{'s' if delta_hour > 1 else ''}")
    if delta_minute > 0:
        parts.append(f"{delta_minute} minute{'s' if delta_minute > 1 else ''}")
    delta_str = " and ".join(parts)

    if mode == 'past':
        question = (f"The time shown on the clock is {current_time_str}. "
                    f"{person} arrived {delta_str} ago. "
                    f"What time did {person} arrive? (Format: H:MM or HH:MM)")
        current_h, current_m = map(int, current_time_str.split(':'))
        answer = ac_calculate_time(current_h, current_m, -delta_hour, -delta_minute)
        delta = {'hours': -delta_hour, 'minutes': -delta_minute}

    elif mode == 'future':
        question = (f"The time shown on the clock is {current_time_str}. "
                    f"{person}'s appointment is in {delta_str}. "
                    f"What time is the appointment? (Format: H:MM or HH:MM)")
        current_h, current_m = map(int, current_time_str.split(':'))
        answer = ac_calculate_time(current_h, current_m, delta_hour, delta_minute)
        delta = {'hours': delta_hour, 'minutes': delta_minute}
    else: 
         question = (f"What time is shown on the clock? (Format: H:MM or HH:MM)")
         answer = current_time_str 
         delta = {'hours': 0, 'minutes': 0}


    instruction = "The image displays a standard analog clock without a seconds hand. Use the hour and minute hands to answer the question."
    full_question = f"{instruction}\n\n{question}"

    return full_question, answer, delta

def generate_ac_data(num_instances, output_dir, start_index):
    """
    Generates images of analog clocks and questions about time calculations.
    Returns metadata including the question, answer, and time details.
    """
    ac_dir = os.path.join(output_dir, "ac")
    os.makedirs(ac_dir, exist_ok=True)
    results = []

    for i in range(num_instances):
        img_idx = start_index + i

        current_hour = random.randint(1, 12) 
        current_minute = random.randint(0, 59)
        current_time_str = f"{current_hour}:{current_minute:02d}"

        mode = random.choice(['past', 'future']) 

        question, answer, delta_info = ac_create_question(current_time_str, mode)

        fig, ax = plt.subplots(figsize=(6, 6)) 
        ac_draw_clock(fig, ax, current_time_str)

        image_filename = f"ac_{img_idx}.png"
        image_path = os.path.join(ac_dir, image_filename)
        plt.savefig(image_path, bbox_inches="tight", dpi=150) 
        plt.close(fig)

        results.append({
            "task_type": "AC",
            "instance_id": img_idx,
            "output_file": os.path.join("ac", image_filename),
            "ground_truth": {
                "question": question,
                "answer": answer, 
                "displayed_time": current_time_str,
                "mode": mode, 
                "delta_hours": delta_info['hours'],
                "delta_minutes": delta_info['minutes']
            }
        })
    print(f"Generated {num_instances} AC instances.")
    return results


#############################
# Main Execution Controller
#############################

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data across multiple tasks.")
    parser.add_argument('--total_generations', type=int, default=50,
                        help="Total number of instances to generate across all tasks.")
    parser.add_argument('--output_dir', type=str, default="synthetic_data_output",
                        help="Directory to save generated images and metadata.")
 
    args = parser.parse_args()

    tasks = {
        "LI": generate_li_data,
        "LF": generate_lf_data,
        "CC": generate_cc_data,
        "WS": generate_ws_data,
        "AC": generate_ac_data
    }
    task_names = list(tasks.keys())
    num_tasks = len(task_names)

    if args.total_generations < num_tasks:
        print(f"Warning: total_generations ({args.total_generations}) is less than the number of tasks ({num_tasks}). Generating 1 instance per task.")
        num_per_task = [1] * num_tasks
        total_generated = num_tasks 
    else:
        base_num_per_task = args.total_generations // num_tasks
        remainder = args.total_generations % num_tasks
        num_per_task = [base_num_per_task] * num_tasks
        for i in range(remainder):
            num_per_task[i] += 1
        total_generated = args.total_generations 

    print(f"Starting data generation. Total instances: {total_generated}")
    print(f"Distribution: {dict(zip(task_names, num_per_task))}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    current_index = 1 

    for i, task_name in enumerate(task_names):
        num_instances = num_per_task[i]
        if num_instances > 0:
            print(f"\n--- Generating Task: {task_name} ({num_instances} instances) ---")
            generate_func = tasks[task_name]
            try:
                task_results = generate_func(num_instances, args.output_dir, current_index)
                all_results.extend(task_results)
                current_index += len(task_results)
            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"Error generating data for task {task_name}: {e}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                import traceback
                traceback.print_exc()


    metadata_path = os.path.join(args.output_dir, "metadata.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSuccessfully generated {len(all_results)} total instances.")
        print(f"All data saved in: {args.output_dir}")
        print(f"Metadata saved to: {metadata_path}")
    except Exception as e:
        print(f"\nError saving metadata file: {e}")

if __name__ == "__main__":
    main()

