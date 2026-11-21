"""Prepare the airport and route tables used by the dashboard.

Reads the OpenFlights files ``airports.dat`` and ``routes.dat`` and writes:

* ``airports.csv``: one row per airport with its position and number of
  destinations.
* ``routes.csv``: one row per pair of airports with at least one direct route.
  Routes that cross the 180th meridian (across the Pacific) are left out,
  because on a flat map they would be drawn around the whole world.
* ``routes_bundled_light.csv`` and ``routes_bundled_strong.csv``: the same routes after edge
  bundling with two bandwidths. Each route is a run of points and routes are
  separated by an empty row, the format Plotly expects for a line trace.

Bundling takes a few seconds, so it runs once here and not inside the app.
Run with ``python prepare_data.py``.
"""

import pandas as pd
from datashader.bundling import hammer_bundle

BANDWIDTHS = {"light": 0.03, "strong": 0.1}
KEEP_EVERY = 2  # keep one point out of every KEEP_EVERY in a bundled route


def load_airports(path):
    """Load the OpenFlights airport table.

    Args:
        path: Path to ``airports.dat``.

    Returns:
        A DataFrame with name, city, country, IATA code, latitude and
        longitude. Airports near the poles are dropped because the map
        projection cannot show them.
    """
    columns = ["name", "city", "country", "iata", "lat", "lon"]
    airports = pd.read_csv(path, header=None, usecols=[1, 2, 3, 4, 6, 7], names=columns)
    airports = airports[airports["lat"].abs() < 85].reset_index(drop=True)
    airports.loc[~airports["iata"].str.fullmatch(r"[A-Z]{3}", na=False), "iata"] = ""
    return airports


def load_routes(path, codes):
    """Load the OpenFlights routes and keep one row per pair of airports.

    Args:
        path: Path to ``routes.dat``.
        codes: Set of valid IATA codes.

    Returns:
        A DataFrame with columns ``a`` and ``b`` (the two airports, in
        alphabetical order) and ``airlines`` (number of airline routes
        between them).
    """
    routes = pd.read_csv(path, header=None, usecols=[2, 4], names=["origen", "destino"])
    routes = routes[routes["origen"].isin(codes) & routes["destino"].isin(codes)
                    & (routes["origen"] != routes["destino"])]
    pairs = pd.DataFrame({"a": routes.min(axis=1), "b": routes.max(axis=1)})
    return pairs.value_counts().reset_index(name="airlines")


def bundle(airports, routes, bandwidth):
    """Bundle the routes with kernel density estimation.

    Args:
        airports: Airport table indexed by IATA code.
        routes: Route table from :func:`load_routes`.
        bandwidth: Initial bandwidth of ``hammer_bundle``. Larger values
            pull more routes together.

    Returns:
        A DataFrame with columns ``x`` and ``y``. Routes are separated by
        an empty row.
    """
    codes = sorted(set(routes["a"]) | set(routes["b"]))
    position = {code: i for i, code in enumerate(codes)}
    nodes = airports.loc[codes, ["lon", "lat"]].rename(columns={"lon": "x", "lat": "y"})
    edges = pd.DataFrame({"source": routes["a"].map(position), "target": routes["b"].map(position)})
    paths = hammer_bundle(nodes.reset_index(drop=True), edges, initial_bandwidth=bandwidth, decay=0.6)

    # thin out each route but keep its two ends
    gap = paths["x"].isna()
    first = ~gap & (gap.shift(1, fill_value=True))
    last = ~gap & (gap.shift(-1, fill_value=True))
    step = paths.groupby(gap.cumsum()).cumcount()
    keep = gap | first | last | (step % KEEP_EVERY == 0)
    return paths[keep].round(2)


def main():
    airports = load_airports("airports.dat")
    codes = set(airports["iata"]) - {""}
    routes = load_routes("routes.dat", codes)

    with_code = airports[airports["iata"] != ""].drop_duplicates("iata").set_index("iata")

    # routes that cross the 180th meridian would be drawn the long way around the world
    span = (routes["a"].map(with_code["lon"]) - routes["b"].map(with_code["lon"])).abs()
    routes = routes[span <= 180].reset_index(drop=True)

    degree = pd.concat([routes["a"], routes["b"]]).value_counts()
    airports["destinations"] = airports["iata"].map(degree).fillna(0).astype(int)

    airports.to_csv("airports.csv", index=False)
    routes.to_csv("routes.csv", index=False)
    for name, bandwidth in BANDWIDTHS.items():
        bundle(with_code, routes, bandwidth).to_csv(f"routes_bundled_{name}.csv", index=False)
    print(f"{len(airports)} airports, {len(routes)} routes")


if __name__ == "__main__":
    main()
