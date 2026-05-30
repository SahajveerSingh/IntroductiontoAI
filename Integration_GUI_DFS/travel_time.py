import math

def flow_to_speed(flow, speed_limit=60):
    try:
        flow = float(flow)
    except ValueError:
        flow = 500

    if flow <= 351:
        return speed_limit

    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return 5

    speed1 = (-b + math.sqrt(discriminant)) / (2 * a)
    speed2 = (-b - math.sqrt(discriminant)) / (2 * a)

    possible_speeds = [s for s in [speed1, speed2] if s > 0]

    if not possible_speeds:
        return 5

    speed = max(possible_speeds)
    return min(speed, speed_limit)


def calculate_travel_time(distance_km, flow, intersection_delay_seconds=30):
    speed = flow_to_speed(flow)

    if speed <= 0:
        speed = 5

    travel_time_hours = distance_km / speed
    travel_time_seconds = travel_time_hours * 3600

    return travel_time_seconds + intersection_delay_seconds