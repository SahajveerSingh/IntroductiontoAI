import sys
import math


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
# IDA* — Recursive DFS with f-cost threshold
# ─────────────────────────────────────────────

def ida_star(nodes, edges, origin, destinations):
    """
    Iterative Deepening A* (IDA*).

    Instead of a priority queue, IDA* performs repeated depth-limited DFS.
    Each iteration uses an f-cost threshold: f(n) = g(n) + h(n).
    - g(n): actual cost from origin to node n so far
    - h(n): Euclidean distance heuristic to nearest destination

    When a node's f-cost exceeds the current threshold, it is pruned and
    its f-cost is recorded. The next iteration's threshold is set to the
    minimum pruned f-cost, gradually expanding the search.

    Tie-breaking (matching assignment spec):
      - Among neighbours, expand in ascending node ID order.

    Returns: (goal_node, nodes_created, path)
    """
    destination_set = set(destinations)

    # nodes_created counts every time a node is generated (pushed onto the
    # recursive stack), including re-generations across iterations.
    nodes_created = [1]  # mutable container so inner function can modify it

    def search(path, g, threshold):
        """
        Recursive DFS bounded by f-cost threshold.

        path    : current path from origin to current node (list of node IDs)
        g       : cumulative cost to reach current node
        threshold: current f-cost cutoff

        Returns:
          ('FOUND', goal_node)  if a destination is reached within threshold
          ('PRUNE', min_f)      if all branches are pruned; min_f is the
                                 smallest f-cost that exceeded the threshold
        """
        current = path[-1]
        f = g + heuristic(current, destinations, nodes)

        # Prune this branch — f exceeds current threshold
        if f > threshold:
            return ('PRUNE', f)

        # Goal check
        if current in destination_set:
            return ('FOUND', current)

        min_pruned = math.inf

        # Expand neighbours in ascending node ID order (tie-breaking rule)
        neighbours = sorted(edges.get(current, []), key=lambda x: x[0])

        for neighbour, cost in neighbours:
            # Avoid revisiting nodes already on the current path (cycle prevention)
            if neighbour in path:
                continue

            nodes_created[0] += 1
            path.append(neighbour)
            result = search(path, g + cost, threshold)

            if result[0] == 'FOUND':
                return result  # bubble the solution up immediately

            if result[0] == 'PRUNE':
                if result[1] < min_pruned:
                    min_pruned = result[1]

            path.pop()  # backtrack

        return ('PRUNE', min_pruned)

    # ── Outer IDA* loop ──────────────────────────────────────────────────────
    # Initial threshold is the heuristic value of the start node (f = 0 + h)
    threshold = heuristic(origin, destinations, nodes)
    path = [origin]

    while True:
        result = search(path, 0, threshold)

        if result[0] == 'FOUND':
            return result[1], nodes_created[0], list(path)

        if result[1] == math.inf:
            # No solution exists
            return None, nodes_created[0], []

        # Raise threshold to the smallest f-cost that was pruned
        threshold = result[1]


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print("Usage: python search_cus2.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method != 'CUS2':
        print(f"This script only supports CUS2. Got: {sys.argv[2]}")
        sys.exit(1)

    nodes, edges, origin, destinations = parse_problem(filename)

    goal, num_nodes, path = ida_star(nodes, edges, origin, destinations)

    if goal is None:
        print(f"{filename} {sys.argv[2]}")
        print("No solution found.")
    else:
        path_str = str(path)
        print(f"{filename} {sys.argv[2]}")
        print(f"{goal} {num_nodes}")
        print(path_str)


if __name__ == '__main__':
    main()
