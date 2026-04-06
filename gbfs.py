import sys
import math
import heapq


# ─────────────────────────────────────────────
# File Parser
# ─────────────────────────────────────────────

def parse_problem(filename):
    """Parse the problem file and return nodes, edges, origin, destinations."""
    nodes = {}       # node_id -> (x, y)
    edges = {}       # node_id -> list of (neighbour_id, cost)
    origin = None
    destinations = []

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    section = None
    for line in lines:
        if not line:
            continue
        if line == 'Nodes:':
            section = 'nodes'
        elif line == 'Edges:':
            section = 'edges'
        elif line == 'Origin:':
            section = 'origin'
        elif line == 'Destinations:':
            section = 'destinations'
        else:
            if section == 'nodes':
                # Format: 1: (4,1)
                parts = line.split(':')
                node_id = int(parts[0].strip())
                coords = parts[1].strip().strip('()')
                x, y = map(int, coords.split(','))
                nodes[node_id] = (x, y)
            elif section == 'edges':
                # Format: (2,1): 4
                paren_part, cost_part = line.split(':')
                cost = int(cost_part.strip())
                paren_part = paren_part.strip().strip('()')
                from_node, to_node = map(int, paren_part.split(','))
                if from_node not in edges:
                    edges[from_node] = []
                edges[from_node].append((to_node, cost))
            elif section == 'origin':
                origin = int(line.strip())
            elif section == 'destinations':
                destinations = [int(d.strip()) for d in line.split(';')]

    return nodes, edges, origin, destinations


# ─────────────────────────────────────────────
# Heuristic: Euclidean distance to nearest goal
# ─────────────────────────────────────────────

def heuristic(node_id, destinations, nodes):
    """Euclidean distance from node to the nearest destination."""
    x1, y1 = nodes[node_id]
    min_dist = math.inf
    for dest in destinations:
        x2, y2 = nodes[dest]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist < min_dist:
            min_dist = dist
    return min_dist


# ─────────────────────────────────────────────
# Greedy Best-First Search (GBFS)
# ─────────────────────────────────────────────

def gbfs(nodes, edges, origin, destinations):
    """
    Greedy Best-First Search.
    Selects the node with the lowest heuristic h(n) = Euclidean distance to goal.
    Tie-breaking:
      1. Lower heuristic value first.
      2. Smaller node ID first (ascending order).
      3. Earlier-added node first (FIFO within equal priority).
    Returns: (goal_node, number_of_nodes_created, path_list)
    """
    destination_set = set(destinations)

    # Priority queue entries: (h_value, insertion_order, node_id, path)
    counter = 0  # insertion order tie-breaker
    start_h = heuristic(origin, destinations, nodes)
    frontier = [(start_h, counter, origin, [origin])]
    heapq.heapify(frontier)

    visited = set()
    nodes_created = 1  # count the start node

    while frontier:
        h, _, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        # Goal check
        if current in destination_set:
            return current, nodes_created, path

        # Expand neighbours
        neighbours = edges.get(current, [])
        # Sort by node ID ascending for tie-breaking consistency
        neighbours_sorted = sorted(neighbours, key=lambda x: x[0])

        for neighbour, cost in neighbours_sorted:
            if neighbour not in visited:
                counter += 1
                nodes_created += 1
                nh = heuristic(neighbour, destinations, nodes)
                heapq.heappush(frontier, (nh, counter, neighbour, path + [neighbour]))

    # No solution found
    return None, nodes_created, []


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method != 'GBFS':
        print(f"This script only supports GBFS. Got: {sys.argv[2]}")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_problem(filename)

    goal, num_nodes, path = gbfs(nodes, edges, origin, destinations)

    if goal is None:
        print(f"{filename} {sys.argv[2]}")
        print("No solution found.")
    else:
        path_str = ' -> '.join(map(str, path))
        print(f"{filename} {sys.argv[2]}")
        print(f"{goal} {num_nodes}")
        print(path_str)


if __name__ == '__main__':
    main()
