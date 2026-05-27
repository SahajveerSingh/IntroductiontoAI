def dfs_top_k_routes(graph, origin, destination, k=5):
    routes = []

    def dfs(current, path, total_cost):
        if current == destination:
            routes.append((path.copy(), total_cost))
            return

        for neighbour, edge_cost in graph.get(current, []):
            if neighbour not in path:
                path.append(neighbour)
                dfs(neighbour, path, total_cost + edge_cost)
                path.pop()

    dfs(origin, [origin], 0)

    routes.sort(key=lambda route: route[1])
    return routes[:k]