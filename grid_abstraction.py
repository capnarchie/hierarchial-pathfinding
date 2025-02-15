import time
import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
import matplotlib.colors as mcolors
import noise
import pickle
from scipy.ndimage import distance_transform_edt  # Requires SciPy

# Seed for reproducibility.
random.seed(4190857439)

# -----------------------------------------------
# Node and Cluster (Supernode) classes.
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        self.cost = random.randint(1, 100)  # Random cost for free cells.

class Supernode:
    def __init__(self, nodes, cluster_bounds, clearance):
        self.nodes = nodes  # Free nodes in the cluster.
        # Instead of arithmetic mean, choose the free node with maximum clearance.
        center_node = max(nodes, key=lambda node: clearance[node.x, node.y])
        self.x = center_node.x
        self.y = center_node.y
        # Average cost over the cluster (can be used in higher-level planning).
        self.cost = sum(node.cost for node in nodes) / len(nodes)
        self.bounds = cluster_bounds  # (min_x, max_x, min_y, max_y)

# -----------------------------------------------
# Grid functions.
def create_grid(rows, cols):
    grid = [[Node(x, y) for y in range(cols)] for x in range(rows)]
    return grid

def add_obstacles(grid, obstacle_probability=0.3):
    """Randomly mark nodes as obstacles using Perlin noise."""
    scale = 100.0  # Controls the frequency of noise patterns
    threshold = 0.001  # Lower = more obstacles, higher = fewer

    rows = len(grid)
    cols = len(grid[0])
    for x in range(rows):
        for y in range(cols):
            value = noise.pnoise2(x / scale, y / scale, octaves=2)
            if value < threshold:
                grid[x][y].cost = float('inf')

def compute_clearance_grid(grid):
    """
    Compute a clearance map where each free cell gets a value equal to its distance
    from the nearest obstacle. Uses SciPy's distance_transform_edt for speed.
    """
    rows, cols = len(grid), len(grid[0])
    # Build a binary grid: 1 for free cell, 0 for obstacle.
    grid_data = np.array([[0 if node.cost == float('inf') else 1 for node in row] for row in grid])
    clearance = distance_transform_edt(grid_data)
    return clearance

# -----------------------------------------------
# Partitioning: Fixed-size clusters.
def partition_grid(grid, cluster_size, clearance):
    """
    Partition the grid into fixed-size blocks. Each block that contains
    at least one free node is turned into a Supernode.
    """
    clusters = []
    rows = len(grid)
    cols = len(grid[0])
    
    for i in range(0, rows, cluster_size):
        for j in range(0, cols, cluster_size):
            nodes = []
            # Determine block boundaries.
            min_x, max_x = i, min(i + cluster_size - 1, rows - 1)
            min_y, max_y = j, min(j + cluster_size - 1, cols - 1)
            # Collect free nodes in the block.
            for x in range(i, min(i + cluster_size, rows)):
                for y in range(j, min(j + cluster_size, cols)):
                    if grid[x][y].cost != float('inf'):
                        nodes.append(grid[x][y])
            if nodes:
                cluster_bounds = (min_x, max_x, min_y, max_y)
                clusters.append(Supernode(nodes, cluster_bounds, clearance))
    return clusters

# -----------------------------------------------
# A* on the grid (low-level path planning).
def heuristic_grid(a, b):
    # Manhattan distance.
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_grid(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path.
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        x, y = current
        # 4-connected grid.
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny].cost == float('inf'):
                    continue
                tentative_g = g_score[current] + grid[nx][ny].cost
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic_grid(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return None  # No path found.

# -----------------------------------------------
# Helper: For abstract planning, compute an edge cost that penalizes obstacles.
def edge_cost_with_obstacles(sn, neighbor_sn, grid, num_samples=200):
    """
    Sample along the straight line between two cluster centers.
    For each sample, if a grid cell is an obstacle, add a penalty.
    """
    dx = neighbor_sn.x - sn.x
    dy = neighbor_sn.y - sn.y
    distance = np.sqrt(dx**2 + dy**2)
    
    obstacle_count = 0
    for i in range(num_samples + 1):
        t = i / num_samples
        x = int(round(sn.x + t * dx))
        y = int(round(sn.y + t * dy))
        # Make sure we are within bounds.
        if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
            continue
        if grid[x][y].cost == float('inf'):
            obstacle_count += 1
    ratio = obstacle_count / (num_samples + 1)
    # Increase the cost by a factor proportional to the obstacle ratio.
    penalty = 1 + ratio * 10000  # You can adjust the multiplier as needed.
    return distance * penalty

# -----------------------------------------------
# Build the abstract graph over clusters.
def build_abstract_graph(clusters, cluster_size, grid):
    """
    Returns:
      - graph: a dict mapping cluster indices (tuple) to a list of (neighbor_key, weight).
      - cluster_map: a dict mapping cluster indices to Supernode objects.
    """
    cluster_map = {}
    for sn in clusters:
        ci = sn.bounds[0] // cluster_size
        cj = sn.bounds[2] // cluster_size
        cluster_map[(ci, cj)] = sn
    
    graph = {}
    for (ci, cj), sn in cluster_map.items():
        neighbors = []
        # Consider 4-connected neighbor clusters.
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            neighbor_key = (ci+di, cj+dj)
            if neighbor_key in cluster_map:
                neighbor_sn = cluster_map[neighbor_key]
                # Use a cost that incorporates obstacles along the line between clusters.
                weight = edge_cost_with_obstacles(sn, neighbor_sn, grid)
                neighbors.append((neighbor_key, weight))
        graph[(ci, cj)] = neighbors
    return graph, cluster_map

def a_star_abstract(graph, start_key, goal_key, cluster_map):
    open_set = []
    heapq.heappush(open_set, (0, start_key))
    came_from = {}
    g_score = {start_key: 0}
    
    def h(key1, key2):
        c1 = cluster_map[key1]
        c2 = cluster_map[key2]
        return np.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal_key:
            # Reconstruct path.
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_key)
            path.reverse()
            return path
        for neighbor, weight in graph.get(current, []):
            tentative_g = g_score[current] + weight
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal_key)
                heapq.heappush(open_set, (f, neighbor))
    return None

# -----------------------------------------------
# New helper: Refine path using abstract waypoints.
def refine_path_with_abstract(grid, start, goal, abstract_path, cluster_map):
    """
    Given an abstract path (a list of cluster keys), compute a refined grid path by planning
    between waypoints that include:
      - The actual start position,
      - The centers of the intermediate clusters (from the abstract path),
      - The actual goal position.
    """
    # Build the waypoint list.
    waypoints = [start]
    if len(abstract_path) > 2:
        # Exclude the first and last clusters (they correspond to the start/goal clusters)
        for cluster_key in abstract_path[1:-1]:
            waypoint = (cluster_map[cluster_key].x, cluster_map[cluster_key].y)
            waypoints.append(waypoint)
    waypoints.append(goal)
    
    # Now compute refined path segments between consecutive waypoints.
    refined_path = []
    for i in range(len(waypoints) - 1):
        segment = a_star_grid(grid, waypoints[i], waypoints[i+1])
        if segment is None:
            return None  # If one segment fails, the overall refinement fails.
        if i > 0:
            # Avoid duplicating the waypoint.
            segment = segment[1:]
        refined_path.extend(segment)
    return refined_path

# -----------------------------------------------
# Fast plotting function.
def plot_grid_fast(grid, direct_path=None, refined_path=None, abstract_path=None, cluster_map=None):
    rows, cols = len(grid), len(grid[0])
    
    grid_data = np.array([[1 if node.cost != float('inf') else 0 for node in row] for row in grid])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    cmap = mcolors.ListedColormap(["black", "darkblue"])
    ax.imshow(grid_data, cmap=cmap, origin='lower')
    
    if direct_path:
        path_cols = [p[1] for p in direct_path]
        path_rows = [p[0] for p in direct_path]
        ax.plot(path_cols, path_rows, 'g-', linewidth=2, label='Direct A* Path')
    
    if refined_path:
        path_cols = [p[1] for p in refined_path]
        path_rows = [p[0] for p in refined_path]
        ax.plot(path_cols, path_rows, 'r-', linewidth=2, label='Abstract-Refined A* Path')

    if abstract_path and cluster_map:
        centers_cols = [cluster_map[key].y for key in abstract_path]
        centers_rows = [cluster_map[key].x for key in abstract_path]
        ax.plot(centers_cols, centers_rows, color='orange', linestyle='--',
                linewidth=2, label='Abstract Path')

    if cluster_map:
        cluster_centers_x = [sn.x for sn in cluster_map.values()]
        cluster_centers_y = [sn.y for sn in cluster_map.values()]
        ax.scatter(cluster_centers_y, cluster_centers_x, color='cyan', s=1, label='Cluster Centers')

    ax.set_title("Comparison of Direct and Abstract-Refined A* Paths")
    ax.legend()
    plt.show()

# -----------------------------------------------
# Helper: Find a free node (if a chosen start/goal is blocked).
def find_free_node(grid, default):
    if grid[default[0]][default[1]].cost != float('inf'):
        return default
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j].cost != float('inf'):
                return (i, j)
    return None

# -----------------------------------------------
# Main driver.
if __name__ == '__main__':
    # Create grid and add obstacles.
    # rows, cols = 2500, 2500
    # start_time = time.time()
    # grid = create_grid(rows, cols)
    # print("Time to create grid:", time.time() - start_time)
    # add_obstacles(grid, obstacle_probability=0.3)
    
    # # Compute clearance for the entire grid.
    # clearance = compute_clearance_grid(grid)
    
    # # Partition grid into clusters (for abstraction).
    cluster_size = 10
    # clusters = partition_grid(grid, cluster_size, clearance)
    
    # # Build abstract graph (nodes are clusters indexed by (ci, cj)).
    # abstract_graph, cluster_map = build_abstract_graph(clusters, cluster_size, grid)
    # # Save grid, clearance, clusters, cluster_map, abstract_graph into memory
    # def save_to_memory(filename, data):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(data, f)

    def load_from_memory(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # Filenames for saved data
    grid_file = 'grid.pkl'
    clearance_file = 'clearance.pkl'
    clusters_file = 'clusters.pkl'
    cluster_map_file = 'cluster_map.pkl'
    abstract_graph_file = 'abstract_graph.pkl'

    # # Save data
    # save_to_memory(grid_file, grid)
    # save_to_memory(clearance_file, clearance)
    # save_to_memory(clusters_file, clusters)
    # save_to_memory(cluster_map_file, cluster_map)
    # save_to_memory(abstract_graph_file, abstract_graph)

    # Load data using concurrent loading for speedup
    import concurrent.futures

    def load_data(filename):
        return load_from_memory(filename)

    start = time.time()
    filenames = [grid_file, clearance_file, clusters_file, cluster_map_file, abstract_graph_file]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        grid, clearance, clusters, cluster_map, abstract_graph = executor.map(load_data, filenames)
    print("Time to load data:", time.time() - start)



    # Choose start and goal.
    start = (190, 294)
    goal  = (1698, 436)
    start = find_free_node(grid, start)
    goal = find_free_node(grid, goal)
    if start is None or goal is None:
        print("No free nodes available for start/goal!")
        exit(1)
    
    print(f"Start: {start}, Goal: {goal}")

    # Identify the clusters that contain the start and goal.
    start_cluster_key = (start[0] // cluster_size, start[1] // cluster_size)
    goal_cluster_key  = (goal[0]  // cluster_size, goal[1]  // cluster_size)
    if start_cluster_key not in cluster_map or goal_cluster_key not in cluster_map:
        print("Start or goal not in any cluster!")
        exit(1)
    
    # --- Direct A* search (full grid) ---
    # t0 = time.time()
    # direct_path = a_star_grid(grid, start, goal)
    # t_direct = time.time() - t0
    # if direct_path is None:
    #     print("No path found with direct A*!")
    #     exit(1)
    # print("Direct A* search time: {:.4f} seconds".format(t_direct))
    
    # --- High-level (abstract) path planning ---
    t1 = time.time()
    abstract_path = a_star_abstract(abstract_graph, start_cluster_key, goal_cluster_key, cluster_map)
    t_abstract_planning = time.time() - t1
    if abstract_path is None:
        print("No abstract path found!")
        exit(1)
    print("Abstract A* planning time (high-level over clusters): {:.4f} seconds".format(t_abstract_planning))
    
    # --- Refinement using abstract waypoints ---
    t2 = time.time()
    refined_path = refine_path_with_abstract(grid, start, goal, abstract_path, cluster_map)
    t_refinement = time.time() - t2
    if refined_path is None:
        print("No refined path found using abstract waypoints!")
        exit(1)
    t_abstract_total = t_abstract_planning + t_refinement
    print("Refinement (using abstract waypoints) time: {:.4f} seconds".format(t_refinement))
    print("Total abstract-refined method time: {:.4f} seconds".format(t_abstract_total))
    
    # --- Speedup factors ---
    # speedup = t_direct / t_refinement if t_abstract_total > 0 else float('inf')
    # print("Speedup factor (direct A* time / abstract-refined time): {:.4f}".format(speedup))
    
    # Plot the results.
    plot_grid_fast(grid, direct_path=None, refined_path=refined_path, abstract_path=abstract_path, cluster_map=cluster_map)
