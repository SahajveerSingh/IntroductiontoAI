import sys

# -------------------------------
# PARSE INPUT FILE
# -------------------------------
def parse_input_file(filename):
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
                if section == "edges":
                    pair, cost = line.split(":")
                    a, b = pair.strip().strip("()").split(",")
                    a, b = int(a), int(b)

                    if a not in edges:
                        edges[a] = []
                    edges[a].append(b)

                elif section == "origin":
                    origin = int(line)

                elif section == "destinations":
                    destinations = {int(x.strip()) for x in line.split(";")}

    return edges, origin, destinations


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
        children = sorted(graph.get(current, []))

        # push in reverse so smaller node is expanded first
        for child in reversed(children):
            if child not in path:  # avoid cycles
                stack.append((child, path + [child]))
                nodes_created += 1

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

    graph, origin, destinations = parse_input_file(filename)

    if method == "DFS":
        goal, nodes_created, path = dfs_search(graph, origin, destinations)

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