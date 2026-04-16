import sys
import math

# -------------------------------
# PARSE INPUT FILE
# -------------------------------
def parse_input_file(filename):
    coordinates = {}
    edges = {}
    origin = None
    destinations = set()
    section = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

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
                    node_id, coord_part = line.split(":")
                    node_id = int(node_id.strip())
                    x, y = coord_part.strip().strip("()").split(",")
                    coordinates[node_id] = (int(x), int(y))

                elif section == "edges":
                    pair, cost = line.split(":")
                    a, b = pair.strip().strip("()").split(",")
                    a, b = int(a), int(b)
                    cost = int(cost.strip())

                    if a not in edges:
                        edges[a] = []
                    edges[a].append((b, cost))

                elif section == "origin":
                    origin = int(line)

                elif section == "destinations":
                    destinations = {int(x.strip()) for x in line.split(";")}

    return coordinates, edges, origin, destinations


# -------------------------------
# DFS IMPLEMENTATION
# -------------------------------
def dfs_search(graph, start, goals):
    stack = [(start, [start])]
    nodes_created = 1  # count start node

    while stack:
        current, path = stack.pop()

        # check goal
        if current in goals:
            return current, nodes_created, path

        # get children sorted ascending
        children = sorted([child for child, cost in graph.get(current, [])])

        # push in reverse so smaller node is expanded first
        for child in reversed(children):
            if child not in path:  # avoid cycles
                stack.append((child, path + [child]))
                nodes_created += 1

    return None, nodes_created, []


# -------------------------------
# HEURISTIC FOR A*
# -------------------------------
def heuristic(node, destinations, coordinates):
    x1, y1 = coordinates[node]
    best = float("inf")

    for d in destinations:
        x2, y2 = coordinates[d]
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        best = min(best, dist)

    return best


# -------------------------------
# A* IMPLEMENTATION
# -------------------------------
def a_star_search(graph, coordinates, start, goals):
    frontier = []
    nodes_created = 1
    insertion_order = 0

    start_h = heuristic(start, goals, coordinates)
    frontier.append({
        "node": start,
        "path": [start],
        "g": 0,
        "h": start_h,
        "f": start_h,
        "order": insertion_order
    })

    insertion_order += 1

    while frontier:
        # Sort by:
        # 1. lowest f
        # 2. smaller node id
        # 3. earlier insertion order
        frontier.sort(key=lambda item: (item["f"], item["node"], item["order"]))
        current = frontier.pop(0)

        # Goal test
        if current["node"] in goals:
            return current["node"], nodes_created, current["path"]

        # Expand children in ascending node order
        children = sorted(graph.get(current["node"], []), key=lambda x: x[0])

        for child, cost in children:
            if child not in current["path"]:  # avoid cycles in tree path
                new_g = current["g"] + cost
                new_h = heuristic(child, goals, coordinates)
                new_f = new_g + new_h

                frontier.append({
                    "node": child,
                    "path": current["path"] + [child],
                    "g": new_g,
                    "h": new_h,
                    "f": new_f,
                    "order": insertion_order
                })

                nodes_created += 1
                insertion_order += 1

    return None, nodes_created, []


# -------------------------------
# MAIN PROGRAM (CLI REQUIRED)
# -------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    coordinates, graph, origin, destinations = parse_input_file(filename)

    if method == "DFS":
        goal, nodes_created, path = dfs_search(graph, origin, destinations)

        # REQUIRED OUTPUT FORMAT
        print(f"{filename} {method}")
        if goal is not None:
            print(f"{goal} {nodes_created}")
            print(" ".join(map(str, path)))
        else:
            print("No goal is reachable")

    elif method == "AS":
        goal, nodes_created, path = a_star_search(graph, coordinates, origin, destinations)

        # REQUIRED OUTPUT FORMAT
        print(f"{filename} {method}")
        if goal is not None:
            print(f"{goal} {nodes_created}")
            print(" ".join(map(str, path)))
        else:
            print("No goal is reachable")

    else:
        print("Method not supported yet")


if __name__ == "__main__":
    main()
