# flight_planner.py

from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse
import sys

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

MIN_LAYOVER_MINUTES = 60

# --------------------------------------------------------- 
# TIME FUNCTIONS
# ---------------------------------------------------------

def parse_time(time_str: str) -> int:
    """Parse HH:MM → minutes since midnight. Raise ValueError for bad format."""
    if ":" not in time_str:
        raise ValueError("Invalid time format")
    try:
        hours, minutes = time_str.split(":")
        hours = int(hours)
        minutes = int(minutes)
    except ValueError:
        raise ValueError("Invalid time components")

    if not (0 <= hours <= 23) or not (0 <= minutes <= 59):
        raise ValueError("Invalid hour/minute range")

    return hours * 60 + minutes


def format_time(total_minutes: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"

# ---------------------------------------------------------
# FLIGHT & ITINERARY
# ---------------------------------------------------------

@dataclass
class Flight:
    origin: str
    dest: str
    flight_number: str
    depart: int
    arrive: int
    economy: int
    business: int
    first: int

    def price_for(self, cabin: str) -> int:
        cabin = cabin.lower()
        if cabin == "economy":
            return self.economy
        if cabin == "business":
            return self.business
        if cabin == "first":
            return self.first
        raise ValueError("Unknown cabin")


class Itinerary:
    def __init__(self, flights: List[Flight]):
        self.flights = flights

    @property
    def origin(self) -> str:
        return self.flights[0].origin if self.flights else ""

    @property
    def dest(self) -> str:
        return self.flights[-1].dest if self.flights else ""

    @property
    def depart_time(self) -> int:
        return self.flights[0].depart if self.flights else 0

    @property
    def arrive_time(self) -> int:
        return self.flights[-1].arrive if self.flights else 0

    def is_empty(self):
        return len(self.flights) == 0

    def num_stops(self) -> int:
        return max(0, len(self.flights) - 1)

    def total_price(self, cabin: str) -> int:
        return sum(f.price_for(cabin) for f in self.flights)


# ---------------------------------------------------------
# PARSING SCHEDULE FILES
# ---------------------------------------------------------

def parse_flight_line_txt(line: str) -> Optional[Flight]:
    """Parse a single text schedule line. Return Flight or None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) != 8:
        raise ValueError(f"Invalid flight line: {line}")

    start, end, flight_id, dep_time, arr_time, econ_price, bus_price, first_price = parts

    takeoff = parse_time(dep_time)
    landing = parse_time(arr_time)
    if landing <= takeoff:
        raise ValueError("Arrival must be after departure")

    return Flight(
        start,
        end,
        flight_id,
        takeoff,
        landing,
        int(econ_price),
        int(bus_price),
        int(first_price),
    )


def load_flights_txt(path: str) -> List[Flight]:
    flights = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                fl = parse_flight_line_txt(line)
            except ValueError:
                raise
            if fl is not None:
                flights.append(fl)
    return flights


def load_flights_csv(file_path: str) -> List[Flight]:
    flight_list = []
    with open(file_path, encoding="utf-8") as file_handle:
        csv_reader = csv.DictReader(file_handle)
        for entry in csv_reader:
            takeoff_time = parse_time(entry["depart"])
            landing_time = parse_time(entry["arrive"])
            if landing_time <= takeoff_time:
                raise ValueError("Arrival must be after departure")

            flight_list.append(
                Flight(
                    entry["origin"],
                    entry["dest"],
                    entry["flight_number"],
                    takeoff_time,
                    landing_time,
                    int(entry["economy"]),
                    int(entry["business"]),
                    int(entry["first"]),
                )
            )
    return flight_list


def load_flights(path: str) -> List[Flight]:
    if path.endswith(".txt"):
        return load_flights_txt(path)
    elif path.endswith(".csv"):
        return load_flights_csv(path)
    else:
        raise ValueError("Unsupported file extension")


# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

def build_graph(flight_list: List[Flight]) -> Dict[str, List[Flight]]:
    flight_graph: Dict[str, List[Flight]] = {}
    for flight in flight_list:
        flight_graph.setdefault(flight.origin, []).append(flight)
    return flight_graph


# ---------------------------------------------------------
# EARLIEST ARRIVAL SEARCH (Dijkstra-like)
# ---------------------------------------------------------

import heapq

def find_earliest_itinerary(
    graph: Dict[str, List[Flight]],
    start: str,
    dest: str,
    earliest_departure: int,
) -> Optional[Itinerary]:

    priority_queue: List[Tuple[int, List[Flight]]] = []

    for flight in graph.get(start, []):
        if flight.depart >= earliest_departure:
            heapq.heappush(priority_queue, (flight.arrive, [flight]))

    explored_airports = {}

    while priority_queue:
        current_arrival, current_path = heapq.heappop(priority_queue)
        last_flight = current_path[-1]

        if last_flight.dest == dest:
            return Itinerary(current_path)

        if last_flight.dest in explored_airports and explored_airports[last_flight.dest] <= current_arrival:
            continue
        explored_airports[last_flight.dest] = current_arrival

        for next_flight in graph.get(last_flight.dest, []):
            if next_flight.depart >= last_flight.arrive + MIN_LAYOVER_MINUTES:
                new_path = current_path + [next_flight]
                heapq.heappush(priority_queue, (next_flight.arrive, new_path))

    return None


# ---------------------------------------------------------
# CHEAPEST SEARCH
# ---------------------------------------------------------

def find_cheapest_itinerary(
    graph: Dict[str, List[Flight]],
    start: str,
    dest: str,
    earliest_departure: int,
    cabin: str,
) -> Optional[Itinerary]:

    priority_queue: List[Tuple[int, int, List[Flight]]] = []

    for flight in graph.get(start, []):
        if flight.depart >= earliest_departure:
            heapq.heappush(priority_queue, (flight.price_for(cabin), flight.arrive, [flight]))

    explored_airports = {}

    while priority_queue:
        total_cost, current_arrival, current_path = heapq.heappop(priority_queue)
        last_flight = current_path[-1]

        if last_flight.dest == dest:
            return Itinerary(current_path)

        if last_flight.dest in explored_airports and explored_airports[last_flight.dest] <= total_cost:
            continue
        explored_airports[last_flight.dest] = total_cost

        for next_flight in graph.get(last_flight.dest, []):
            if next_flight.depart >= last_flight.arrive + MIN_LAYOVER_MINUTES:
                heapq.heappush(
                    priority_queue,
                    (total_cost + next_flight.price_for(cabin), next_flight.arrive, current_path + [next_flight])
                )

    return None


# ---------------------------------------------------------
# OUTPUT (COMPARISON TABLE)
# ---------------------------------------------------------

@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[str]
    itinerary: Optional[Itinerary]
    note: str = ""


def format_comparison_table(
    origin: str,
    dest: str,
    earliest_departure: int,
    rows: List[ComparisonRow],
) -> str:

    output_lines = []
    output_lines.append(f"Route: {origin} → {dest}, Earliest depart = {format_time(earliest_departure)}")
    output_lines.append("-" * 72)
    output_lines.append(f"{'Mode':20} {'Cabin':8} {'Dep':6} {'Arr':6} {'Total Price':12} {'Note'}")
    output_lines.append("-" * 72)

    for row in rows:
        if row.itinerary is None:
            output_lines.append(f"{row.mode:20} {row.cabin or '-':8} {'-':6} {'-':6} {'-':12} {row.note}")
        else:
            itinerary = row.itinerary
            dep_str = format_time(itinerary.depart_time)
            arr_str = format_time(itinerary.arrive_time)
            cost_str = "-"
            if row.cabin:
                cost_str = str(itinerary.total_price(row.cabin))
            output_lines.append(f"{row.mode:20} {row.cabin or '-':8} {dep_str:6} {arr_str:6} {cost_str:12} {row.note}")

    return "\n".join(output_lines)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    compare_cmd = subparsers.add_parser("compare")
    compare_cmd.add_argument("file")
    compare_cmd.add_argument("origin")
    compare_cmd.add_argument("dest")
    compare_cmd.add_argument("earliest")

    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        flight_list = load_flights(args.file)
        flight_graph = build_graph(flight_list)
        min_dep_time = parse_time(args.earliest)

        comparison_list = [
            ComparisonRow(
                mode="Earliest arrival",
                cabin=None,
                itinerary=find_earliest_itinerary(flight_graph, args.origin, args.dest, min_dep_time),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (Economy)",
                cabin="economy",
                itinerary=find_cheapest_itinerary(flight_graph, args.origin, args.dest, min_dep_time, "economy"),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (Business)",
                cabin="business",
                itinerary=find_cheapest_itinerary(flight_graph, args.origin, args.dest, min_dep_time, "business"),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (First)",
                cabin="first",
                itinerary=find_cheapest_itinerary(flight_graph, args.origin, args.dest, min_dep_time, "first"),
                note="",
            )
        ]

        print(format_comparison_table(args.origin, args.dest, min_dep_time, comparison_list))


if __name__ == "__main__":
    main()