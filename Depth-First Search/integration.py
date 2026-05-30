import json
from pathlib import Path

from travel_time import calculate_travel_time
from dfs_route_search import dfs_top_k_routes


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "src" / "config" / "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_flow_placeholder(scats_site, time_value):
    """
    Temporary prediction function.
    Replace this with the final LSTM/GRU prediction function later.
    """
    flow_by_time = {
        "07:00": 700,
        "08:00": 900,
        "09:00": 750,
        "12:00": 500,
        "17:00": 950,
        "18:00": 850
    }

    return flow_by_time.get(time_value, 500)


def get_demo_scats_edges():
    """
    Demo SCATS network.
    Distance values are in kilometres.
    """
    return {
        "2000": [("3002", 1.20), ("3003", 1.50)],
        "3002": [("3004", 1.10), ("2200", 1.40)],
        "3003": [("3004", 0.90), ("3977", 1.80)],
        "3004": [("3977", 1.00)],
        "2200": [("3977", 1.30)],
        "3977": []
    }


def build_time_weighted_graph(time_value):
    config = load_config()
    raw_edges = get_demo_scats_edges()

    graph = {}

    for start_site, neighbours in raw_edges.items():
        graph[start_site] = []

        for end_site, distance_km in neighbours:
            predicted_flow = predict_flow_placeholder(start_site, time_value)

            travel_time = calculate_travel_time(
                distance_km=distance_km,
                flow=predicted_flow,
                intersection_delay_seconds=config["intersection_delay_seconds"]
            )

            graph[start_site].append((end_site, travel_time))

    return graph


def get_routes(origin, destination, time_value):
    config = load_config()
    graph = build_time_weighted_graph(time_value)

    if origin not in graph:
        raise ValueError("Origin SCATS site does not exist in the current graph.")

    if destination not in graph:
        raise ValueError("Destination SCATS site does not exist in the current graph.")

    return dfs_top_k_routes(
        graph=graph,
        origin=origin,
        destination=destination,
        k=config["top_k_routes"]
    )