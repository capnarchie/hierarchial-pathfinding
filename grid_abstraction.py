import numpy as np
import random
import matplotlib.pyplot as plt
import heapq

# Seed for reproducibility.
random.seed(43908574390)

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
    def __init__(self, nodes, cluster_bounds):
        self.nodes = nodes  # Free nodes in the cluster.
        # Center computed as the arithmetic mean of free node positions.
        self.x = int(sum(node.x for node in nodes) / len(nodes))
        self.y = int(sum(node.y for node in nodes) / len(nodes))
        # Average cost over the cluster (can be used in higher-level planning).
        self.cost = sum(node.cost for node in nodes) / len(nodes)
        self.bounds = cluster_bounds  # (min_x, max_x, min_y, max_y)

# -----------------------------------------------
# Grid functions.
def create_grid(rows, cols):
    grid = [[Node(x, y) for y in range(cols)] for x in range(rows)]
    return grid

def add_obstacles(grid, obstacle_probability=0.3):
    """Randomly mark nodes as obstacles (cost == infinity)."""
    for row in grid:
        for node in row:
            if random.random() < obstacle_probability:
                node.cost = float('inf')  # Obstacle

# -----------------------------------------------
# Partitioning: Fixed-size clusters.
def partition_grid(grid, cluster_size):
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
                clusters.append(Supernode(nodes, cluster_bounds))
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
                # Skip obstacles.
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
# Build the abstract graph over clusters.
def build_abstract_graph(clusters, cluster_size):
    """
    Returns:
      - graph: a dict mapping cluster indices (tuple) to a list of (neighbor_key, weight).
      - cluster_map: a dict mapping cluster indices to Supernode objects.
    
    Cluster indices are computed as (ci, cj) = (min_x // cluster_size, min_y // cluster_size).
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
                # Use Euclidean distance between cluster centers as weight.
                weight = np.sqrt((sn.x - neighbor_sn.x)**2 + (sn.y - neighbor_sn.y)**2)
                neighbors.append((neighbor_key, weight))
        graph[(ci, cj)] = neighbors
    return graph, cluster_map

# A* on the abstract graph.
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
# Plotting functions.
def plot_grid(grid, path=None):
    """Plot the full grid with obstacles (black) and free nodes (blue)."""
    fig, ax = plt.subplots()
    
    for row in grid:
        for node in row:
            if node.cost == float('inf'):
                ax.plot(node.x, node.y, 'ks', markersize=4)  # Obstacle.
            else:
                ax.plot(node.x, node.y, 'bs', markersize=4)  # Free cell.
                
    if path is not None:
        # Plot refined path in red.
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'r-', linewidth=2, label="Refined Path")
        
    ax.set_xlim(-1, len(grid))
    ax.set_ylim(-1, len(grid[0]))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.title("Grid with Refined Path")
    plt.legend()
    plt.show()

def plot_paths(grid, clusters, abstract_path, refined_path, cluster_map):
    """
    Plot the grid and clusters with:
      - Cluster centers and boundaries (red),
      - Abstract path (green dashed line between cluster centers),
      - Refined path (red solid line along grid nodes).
    """
    fig, ax = plt.subplots()
    
    # Draw grid nodes.
    for row in grid:
        for node in row:
            if node.cost == float('inf'):
                ax.plot(node.x, node.y, 'ks', markersize=4)
            else:
                ax.plot(node.x, node.y, 'bs', markersize=4)
    
    # Draw clusters.
    for sn in clusters:
        ax.plot(sn.x, sn.y, 'ro', markersize=6)
        min_x, max_x, min_y, max_y = sn.bounds
        rect = plt.Rectangle((min_x - 0.5, min_y - 0.5),
                 max_x - min_x + 1, max_y - min_y + 1,
                 fill=False, edgecolor='lime', linestyle='--', linewidth=2000000)
        ax.add_patch(rect)
    
    # Draw abstract path (green dashed line).
    if abstract_path is not None:
        centers = [ (cluster_map[key].x, cluster_map[key].y) for key in abstract_path ]
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        ax.plot(xs, ys, 'orange', linestyle='--', linewidth=3, label="Abstract Path")
    
    # Draw refined path (red solid line).
    if refined_path is not None:
        xs = [p[0] for p in refined_path]
        ys = [p[1] for p in refined_path]
        ax.plot(xs, ys, 'c-', linewidth=3, label="Refined Path")
        ax.plot([], [], 'ro', markersize=4, label="Contracted Map")
        ax.plot([], [], 'ks', markersize=4, label="Obstacle")
        ax.plot([], [], 'bs', markersize=4, label="Walkable Node")
    
    ax.set_xlim(-1, len(grid))
    ax.set_ylim(-1, len(grid[0]))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Hierarchical Pathfinding")
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
    rows, cols = 100, 100
    grid = create_grid(rows, cols)
    add_obstacles(grid, obstacle_probability=0.3)
    
    # Partition grid into clusters (for abstraction).
    cluster_size = 5
    clusters = partition_grid(grid, cluster_size)
    
    # Build abstract graph (nodes are clusters indexed by (ci, cj)).
    abstract_graph, cluster_map = build_abstract_graph(clusters, cluster_size)
    
    # Choose start and goal (user-defined) and ensure they are free.
    start = (0,0)#find_free_node(grid, (0, 0))
    goal  = (81,86)#find_free_node(grid, (rows-1, cols-1))
    if start is None or goal is None:
        print("No free nodes available for start/goal!")
        exit(1)
    
    print(f"Start: {start}, Goal: {goal}")
    
    # Low-level pathfinding on the grid.
    refined_path = a_star_grid(grid, start, goal)
    if refined_path is None:
        print("No path found on the grid!")
    
    # Identify the clusters that contain the start and goal.
    start_cluster_key = (start[0] // cluster_size, start[1] // cluster_size)
    goal_cluster_key  = (goal[0]  // cluster_size, goal[1]  // cluster_size)
    if start_cluster_key not in cluster_map or goal_cluster_key not in cluster_map:
        print("Start or goal not in any cluster!")
        exit(1)
    
    # High-level (abstract) path planning.
    abstract_path = a_star_abstract(abstract_graph, start_cluster_key, goal_cluster_key, cluster_map)
    if abstract_path is None:
        print("No abstract path found!")
    
    # Plot the results.
    # Plot grid with refined (low-level) path.
    plot_grid(grid, refined_path)
    # Plot both abstract and refined paths overlayed on the grid and clusters.
    plot_paths(grid, clusters, abstract_path, refined_path, cluster_map)
