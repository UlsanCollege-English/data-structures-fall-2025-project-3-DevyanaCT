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

def parse_time(s: str) -> int:
    """Parse HH:MM → minutes since midnight. Raise ValueError for bad format."""
    if ":" not in s:
        raise ValueError("Invalid time format")
    try:
        h, m = s.split(":")
        h = int(h)
        m = int(m)
    except:
        raise ValueError("Invalid time components")

    if not (0 <= h <= 23) or not (0 <= m <= 59):
        raise ValueError("Invalid hour/minute range")

    return h * 60 + m


def format_time(t: int) -> str:
    """Convert minutes since midnight to HH:MM."""
    h = t // 60
    m = t % 60
    return f"{h:02d}:{m:02d}"

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

    origin, dest, num, dep_s, arr_s, econ_s, biz_s, first_s = parts

    depart = parse_time(dep_s)
    arrive = parse_time(arr_s)
    if arrive <= depart:
        raise ValueError("Arrival must be after departure")

    return Flight(
        origin,
        dest,
        num,
        depart,
        arrive,
        int(econ_s),
        int(biz_s),
        int(first_s),
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


def load_flights_csv(path: str) -> List[Flight]:
    flights = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depart = parse_time(row["depart"])
            arrive = parse_time(row["arrive"])
            if arrive <= depart:
                raise ValueError("Arrival must be after departure")

            flights.append(
                Flight(
                    row["origin"],
                    row["dest"],
                    row["flight_number"],
                    depart,
                    arrive,
                    int(row["economy"]),
                    int(row["business"]),
                    int(row["first"]),
                )
            )
    return flights


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

def build_graph(flights: List[Flight]) -> Dict[str, List[Flight]]:
    graph: Dict[str, List[Flight]] = {}
    for fl in flights:
        graph.setdefault(fl.origin, []).append(fl)
    return graph


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

    pq: List[Tuple[int, List[Flight]]] = []

    for fl in graph.get(start, []):
        if fl.depart >= earliest_departure:
            heapq.heappush(pq, (fl.arrive, [fl]))

    visited = {}

    while pq:
        arr_time, path = heapq.heappop(pq)
        last = path[-1]

        if last.dest == dest:
            return Itinerary(path)

        if last.dest in visited and visited[last.dest] <= arr_time:
            continue
        visited[last.dest] = arr_time

        for nxt in graph.get(last.dest, []):
            if nxt.depart >= last.arrive + MIN_LAYOVER_MINUTES:
                newpath = path + [nxt]
                heapq.heappush(pq, (nxt.arrive, newpath))

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

    pq: List[Tuple[int, int, List[Flight]]] = []

    for fl in graph.get(start, []):
        if fl.depart >= earliest_departure:
            heapq.heappush(pq, (fl.price_for(cabin), fl.arrive, [fl]))

    visited = {}

    while pq:
        cost, arr, path = heapq.heappop(pq)
        last = path[-1]

        if last.dest == dest:
            return Itinerary(path)

        if last.dest in visited and visited[last.dest] <= cost:
            continue
        visited[last.dest] = cost

        for nxt in graph.get(last.dest, []):
            if nxt.depart >= last.arrive + MIN_LAYOVER_MINUTES:
                heapq.heappush(
                    pq,
                    (cost + nxt.price_for(cabin), nxt.arrive, path + [nxt])
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

    lines = []
    lines.append(f"Route: {origin} → {dest}, Earliest depart = {format_time(earliest_departure)}")
    lines.append("-" * 72)
    lines.append(f"{'Mode':20} {'Cabin':8} {'Dep':6} {'Arr':6} {'Total Price':12} {'Note'}")
    lines.append("-" * 72)

    for row in rows:
        if row.itinerary is None:
            lines.append(f"{row.mode:20} {row.cabin or '-':8} {'-':6} {'-':6} {'-':12} {row.note}")
        else:
            itin = row.itinerary
            dep = format_time(itin.depart_time)
            arr = format_time(itin.arrive_time)
            price = "-"
            if row.cabin:
                price = str(itin.total_price(row.cabin))
            lines.append(f"{row.mode:20} {row.cabin or '-':8} {dep:6} {arr:6} {price:12} {row.note}")

    return "\n".join(lines)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    cmp = sub.add_parser("compare")
    cmp.add_argument("file")
    cmp.add_argument("origin")
    cmp.add_argument("dest")
    cmp.add_argument("earliest")

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "compare":
        flights = load_flights(args.file)
        graph = build_graph(flights)
        earliest = parse_time(args.earliest)

        rows = [
            ComparisonRow(
                mode="Earliest arrival",
                cabin=None,
                itinerary=find_earliest_itinerary(graph, args.origin, args.dest, earliest),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (Economy)",
                cabin="economy",
                itinerary=find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "economy"),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (Business)",
                cabin="business",
                itinerary=find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "business"),
                note="",
            ),
            ComparisonRow(
                mode="Cheapest (First)",
                cabin="first",
                itinerary=find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "first"),
                note="",
            )
        ]

        print(format_comparison_table(args.origin, args.dest, earliest, rows))


if __name__ == "__main__":
    main()