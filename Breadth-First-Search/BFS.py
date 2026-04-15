import sys
from collections import deque


# ─────────────────────────────────────────────
# File Parser
# ─────────────────────────────────────────────

def parse_problem(filename):
    """
    Parse the route-finding problem file.

    Returns:
        nodes: dict where key = node ID, value = (x, y)
        edges: dict where key = from_node, value = list of (to_node, cost)
        origin: starting node
        destinations: list of goal nodes
    """
    nodes = {}
    edges = {}
    origin = None
    destinations = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    section = None

    for line in lines:
        if not line:
            continue

        if line == "Nodes:":
            section = "nodes"
        elif line == "Edges:":
            section = "edges"
        elif line == "Origin:":
            section = "origin"
        elif line == "Destinations:":
            section = "destinations"
        else:
            if section == "nodes":
                # Format: 1: (4,1)
                parts = line.split(":")
                node_id = int(parts[0].strip())
                coords = parts[1].strip().strip("()")
                x, y = map(int, coords.split(","))
                nodes[node_id] = (x, y)

            elif section == "edges":
                # Format: (2,1): 4
                paren_part, cost_part = line.split(":")
                cost = int(cost_part.strip())
                paren_part = paren_part.strip().strip("()")
                from_node, to_node = map(int, paren_part.split(","))

                if from_node not in edges:
                    edges[from_node] = []

                edges[from_node].append((to_node, cost))

            elif section == "origin":
                origin = int(line)

            elif section == "destinations":
                # Format: 5; 4
                destinations = [int(d.strip()) for d in line.split(";") if d.strip()]

    if origin is None:
        raise ValueError("Origin node missing from input file.")

    if not destinations:
        raise ValueError("Destination nodes missing from input file.")

    return nodes, edges, origin, destinations


# ─────────────────────────────────────────────
# Breadth-First Search (BFS)
# ─────────────────────────────────────────────

def bfs(edges, origin, destinations):
    """
    Breadth-First Search.

    Expands nodes level by level using a FIFO queue.

    Tie-breaking:
      1. Nodes are expanded in breadth-first order.
      2. Smaller node IDs are added first.
      3. Earlier-added nodes are expanded first naturally by the queue.

    Returns:
        (goal_node, number_of_nodes_created, path_list)
    """
    destination_set = set(destinations)

    # Queue entries: (current_node, path_to_current)
    frontier = deque()
    frontier.append((origin, [origin]))

    # Count the start node as created
    nodes_created = 1

    while frontier:
        current, path = frontier.popleft()

        # Goal check
        if current in destination_set:
            return current, nodes_created, path

        # Get neighbours of current node
        neighbours = edges.get(current, [])

        # Sort by neighbour node ID ascending
        neighbours_sorted = sorted(neighbours, key=lambda x: x[0])

        for neighbour, cost in neighbours_sorted:
            # Prevent cycles in the current path
            if neighbour not in path:
                new_path = path + [neighbour]
                frontier.append((neighbour, new_path))
                nodes_created += 1

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

    if method != "BFS":
        print(f"This script only supports BFS. Got: {sys.argv[2]}")
        sys.exit(1)

    try:
        nodes, edges, origin, destinations = parse_problem(filename)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading problem file: {e}")
        sys.exit(1)

    goal, num_nodes, path = bfs(edges, origin, destinations)

    print(f"{filename} {sys.argv[2]}")

    if goal is None:
        print(f"None {num_nodes}")
        print("[]")
    else:
        print(f"{goal} {num_nodes}")
        path_output = "[" + ",".join(map(str, path)) + "]"
        print(path_output)


if __name__ == "__main__":
    main()