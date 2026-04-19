import sys


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
# Depth-Limited DFS
# ─────────────────────────────────────────────

def depth_limited_dfs(edges, current, destinations, limit, path, node_counter):
    """
    Recursive Depth-Limited DFS used by IDDFS.

    Args:
        edges: adjacency dictionary
        current: current node
        destinations: set of goal nodes
        limit: remaining depth limit
        path: current path as a list
        node_counter: single-item list storing total nodes created

    Returns:
        (goal_node, path_list) if found
        (None, []) if not found
    """
    # Goal check
    if current in destinations:
        return current, path

    # Stop if depth limit reached
    if limit == 0:
        return None, []

    # Expand neighbours in ascending node order
    neighbours = edges.get(current, [])
    neighbours_sorted = sorted(neighbours, key=lambda x: x[0])

    for neighbour, cost in neighbours_sorted:
        # Prevent cycles in current path
        if neighbour not in path:
            node_counter[0] += 1
            new_path = path + [neighbour]

            goal, result_path = depth_limited_dfs(
                edges,
                neighbour,
                destinations,
                limit - 1,
                new_path,
                node_counter
            )

            if goal is not None:
                return goal, result_path

    return None, []


# ─────────────────────────────────────────────
# Iterative Deepening Depth-First Search (IDDFS)
# ─────────────────────────────────────────────

def iddfs(nodes, edges, origin, destinations):
    """
    Iterative Deepening Depth-First Search.

    Repeatedly runs Depth-Limited DFS with increasing depth limits:
    0, 1, 2, 3, ...

    Tie-breaking:
      1. Smaller node IDs are expanded first.
      2. Earlier-added nodes are naturally explored first in DFS order.

    Returns:
        (goal_node, number_of_nodes_created, path_list)
    """
    destination_set = set(destinations)

    # Maximum useful depth with cycle prevention is len(nodes) - 1
    max_depth = len(nodes) - 1

    # Count nodes created across all iterations
    total_nodes_created = 0

    for depth_limit in range(max_depth + 1):
        # Count the root node as created for each new depth-limited search
        node_counter = [1]
        total_nodes_created += 1

        goal, path = depth_limited_dfs(
            edges,
            origin,
            destination_set,
            depth_limit,
            [origin],
            node_counter
        )

        # Add child nodes created during this iteration
        total_nodes_created += (node_counter[0] - 1)

        if goal is not None:
            return goal, total_nodes_created, path

    return None, total_nodes_created, []


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python cus1.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method != "CUS1":
        print(f"This script only supports CUS1. Got: {sys.argv[2]}")
        sys.exit(1)

    try:
        nodes, edges, origin, destinations = parse_problem(filename)
    except FileNotFoundError:
        print(f"Error: file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while reading problem file: {e}")
        sys.exit(1)

    goal, num_nodes, path = iddfs(nodes, edges, origin, destinations)

    print(f"{filename} {sys.argv[2]}")

    if goal is None:
        print(f"None {num_nodes}")
        print("[]")
    else:
        print(f"{goal} {num_nodes}")
        path_output = "[" + ", ".join(map(str, path)) + "]"
        print(path_output)


if __name__ == "__main__":
    main()