"""Interactive dashboard of the world's airports and direct flight routes.

The data comes from OpenFlights and is prepared by ``prepare_data.py``. The app
has two views over an OpenStreetMap base map:

* Airports: the points are stored in a quadtree. Each time the map moves,
  the tree answers which airports are on screen without checking all of
  them. When there are too many to draw, the map shows one circle per
  tree cell, sized by how many airports it holds (level of detail).
* Routes: direct routes drawn as straight lines or bundled with kernel
  density estimation, in two strengths.

Run locally with ``python app.py`` or in production with
``gunicorn app:server``.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

LEAF_CAPACITY = 10
MAX_DEPTH = 12
WORLD = dict(x0=-180.0, y0=-85.0, x1=180.0, y1=85.0, zoom=0.8, lon=-20.0, lat=20.0)
MAP_STYLE = "carto-positron"  # OpenStreetMap data, light style; "open-street-map" also works

FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
INK = "#1F2933"
MUTED = "#7B8794"
FAINT = "#E4E7EB"
ACCENT = "#F28E2B"
BAR = "#A7BBC7"
LINK = "#3E6FA8"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

airports = pd.read_csv("airports.csv", keep_default_na=False)
routes = pd.read_csv("routes.csv", keep_default_na=False)
bundled = {"poco": pd.read_csv("routes_bundled_light.csv"), "mucho": pd.read_csv("routes_bundled_strong.csv")}

X = airports["lon"].to_numpy()
Y = airports["lat"].to_numpy()
by_code = airports[airports["iata"] != ""].drop_duplicates("iata").set_index("iata")
row_of_code = (airports[airports["iata"] != ""].reset_index().drop_duplicates("iata")
               .set_index("iata")["index"].to_dict())


def straight_lines(pairs):
    """Return the coordinates of straight routes separated by ``None``.

    Args:
        pairs: DataFrame with columns ``a`` and ``b`` (IATA codes).

    Returns:
        Two lists, longitudes and latitudes, ready for a Plotly line trace.
    """
    lon, lat = [], []
    for a, b in zip(pairs["a"], pairs["b"]):
        lon += [by_code.at[a, "lon"], by_code.at[b, "lon"], None]
        lat += [by_code.at[a, "lat"], by_code.at[b, "lat"], None]
    return lon, lat


STRAIGHT = straight_lines(routes)


# --------------------------------------------------------------------------
# Quadtree
# --------------------------------------------------------------------------

class QuadNode:
    """A square region of the map and the airports inside it.

    A node is either a leaf, which keeps the indices of its airports, or an
    internal node with four children of half its size.
    """

    def __init__(self, x0, y0, x1, y1, depth):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.depth = depth
        self.children = []
        self.indices = None
        self.count = 0
        self.mean_x = self.mean_y = 0.0

    def touches(self, x0, y0, x1, y1):
        """Return True if the node overlaps the window ``(x0, y0, x1, y1)``."""
        return not (self.x1 < x0 or self.x0 > x1 or self.y1 < y0 or self.y0 > y1)


def build(indices, x0, y0, x1, y1, depth=0):
    """Build a quadtree over the airports in ``indices``.

    A node is split into four while it holds more than ``LEAF_CAPACITY``
    airports and has not reached ``MAX_DEPTH``.

    Args:
        indices: Array of row numbers in the airport table.
        x0, y0, x1, y1: Bounds of the node in degrees.
        depth: Depth of the node, 0 for the root.

    Returns:
        The root ``QuadNode``.
    """
    node = QuadNode(x0, y0, x1, y1, depth)
    node.count = len(indices)
    if len(indices):
        node.mean_x, node.mean_y = X[indices].mean(), Y[indices].mean()
    if len(indices) <= LEAF_CAPACITY or depth == MAX_DEPTH:
        node.indices = indices
        return node
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    left, low = X[indices] < xm, Y[indices] < ym
    for mask, bounds in [(left & low, (x0, y0, xm, ym)), (~left & low, (xm, y0, x1, ym)),
                         (left & ~low, (x0, ym, xm, y1)), (~left & ~low, (xm, ym, x1, y1))]:
        node.children.append(build(indices[mask], *bounds, depth + 1))
    return node


def query(node, x0, y0, x1, y1, stats):
    """Return the airports inside a window, skipping nodes that do not touch it.

    Args:
        node: Root of the tree or subtree.
        x0, y0, x1, y1: The window in degrees.
        stats: Dict with counters ``nodes`` (nodes visited) and ``checked``
            (airports compared one by one). It is updated in place.

    Returns:
        List of row numbers of the airports inside the window.
    """
    stats["nodes"] += 1
    if not node.touches(x0, y0, x1, y1):
        return []
    if not node.children:
        stats["checked"] += len(node.indices)
        i = node.indices
        inside = (X[i] >= x0) & (X[i] <= x1) & (Y[i] >= y0) & (Y[i] <= y1)
        return list(i[inside])
    found = []
    for child in node.children:
        found += query(child, x0, y0, x1, y1, stats)
    return found


def cells(node, depth, x0, y0, x1, y1):
    """Return the non-empty nodes at ``depth`` that touch a window.

    Leaves above ``depth`` are returned as they are.
    """
    if node.count == 0 or not node.touches(x0, y0, x1, y1):
        return []
    if node.depth == depth or not node.children:
        return [node]
    return [c for child in node.children for c in cells(child, depth, x0, y0, x1, y1)]


tree = build(np.arange(len(airports)), -180.0, -90.0, 180.0, 90.0)


# --------------------------------------------------------------------------
# Map view
# --------------------------------------------------------------------------

def window_from_relayout(relayout, current):
    """Read the visible window from the map's ``relayoutData``.

    Args:
        relayout: The ``relayoutData`` of the map, or ``None``.
        current: The window stored before this event.

    Returns:
        A dict with the bounds ``x0``, ``y0``, ``x1``, ``y1``, the ``zoom``
        and the center ``lon``, ``lat``.
    """
    if not relayout or "map._derived" not in relayout:
        return current
    corners = relayout["map._derived"]["coordinates"]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    zoom = relayout.get("map.zoom", current["zoom"])
    center = relayout.get("map.center", {"lon": current["lon"], "lat": current["lat"]})
    if max(lons) - min(lons) >= 360:
        x0, x1 = -180.0, 180.0
    else:
        x0, x1 = max(min(lons), -180.0), min(max(lons), 180.0)
    return dict(x0=x0, y0=max(min(lats), -85.0), x1=x1, y1=min(max(lats), 85.0), zoom=zoom,
                lon=center["lon"], lat=center["lat"])


def depth_for_zoom(zoom):
    """Choose how deep the drawn cells go for a zoom level."""
    return int(min(MAX_DEPTH, max(2, math.floor(zoom) + 4)))


def cell_outlines(nodes):
    """Return the borders of the nodes as line coordinates separated by ``None``."""
    lon, lat = [], []
    for n in nodes:
        lon += [n.x0, n.x1, n.x1, n.x0, n.x0, None]
        lat += [n.y0, n.y0, n.y1, n.y1, n.y0, None]
    return lon, lat


def base_map(figure, window):
    """Apply the shared map style and point the camera at ``window``."""
    figure.update_layout(
        map=dict(style=MAP_STYLE, center=dict(lon=window["lon"], lat=window["lat"]), zoom=window["zoom"]),
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white", font=dict(family=FONT, color=INK, size=12),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT)))
    return figure


def airports_map(window, max_points, show_tree, selected):
    """Draw the airports in the window as points or grouped by quadtree cell.

    When there are too many airports, each cell at the depth chosen for the
    zoom is drawn as one circle at the mean position of its airports, with
    an area proportional to how many it holds.

    Args:
        window: Visible window from :func:`window_from_relayout`.
        max_points: Above this many airports the map switches to groups.
        show_tree: Whether to draw the borders of the leaves on screen.
        selected: Row number of the selected airport, or ``None``.

    Returns:
        The figure and a dict with the numbers shown next to the map.
    """
    box = (window["x0"], window["y0"], window["x1"], window["y1"])
    stats = {"nodes": 0, "checked": 0}
    found = query(tree, *box, stats)
    figure = go.Figure()

    if len(found) > max_points:
        depth = depth_for_zoom(window["zoom"])
        drawn = cells(tree, depth, *box)
        counts = [n.count for n in drawn]
        counts = np.array(counts)
        figure.add_trace(go.Scattermap(
            lon=[n.mean_x for n in drawn], lat=[n.mean_y for n in drawn], mode="markers",
            marker=dict(size=6 + 34 * np.sqrt(counts / counts.max()), color=LINK, opacity=0.6),
            hovertext=[f"{c} aeropuerto" if c == 1 else f"{c} aeropuertos" for c in counts], hoverinfo="text"))
        stats["drawn"], stats["drawn_as"] = len(drawn), f"grupos dibujados, uno por cuadro del nivel {depth}"
    else:
        rows = airports.iloc[found]
        figure.add_trace(go.Scattermap(
            lon=rows["lon"], lat=rows["lat"], mode="markers", customdata=found,
            marker=dict(size=4 + 1.1 * np.sqrt(rows["destinations"]), color=ACCENT, opacity=0.8),
            hovertext=rows["name"] + "<br>" + rows["city"] + ", " + rows["country"], hoverinfo="text"))
        stats["drawn"], stats["drawn_as"] = len(found), "puntos dibujados"

    if show_tree:
        leaves = cells(tree, MAX_DEPTH, *box)
        lon, lat = cell_outlines(leaves)
        figure.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", hoverinfo="skip",
                                       line=dict(width=0.6, color=LINK)))

    if selected is not None:
        row = airports.iloc[selected]
        figure.add_trace(go.Scattermap(lon=[row["lon"]], lat=[row["lat"]], mode="markers",
                                       hoverinfo="skip", marker=dict(size=16, color=INK)))

    stats["found"] = len(found)
    return base_map(figure, window), stats


def routes_map(window, bundling, selected):
    """Draw the direct routes, straight or bundled, or, when an airport is selected, only its destinations.

    Args:
        window: Visible window from :func:`window_from_relayout`.
        bundling: ``"ninguno"``, ``"poco"`` or ``"mucho"``.
        selected: Row number of the selected airport, or ``None``.

    Returns:
        The figure and a dict with the numbers shown next to the map.
    """
    figure = go.Figure()
    if bundling == "ninguno":
        lon, lat = STRAIGHT
    else:
        lon, lat = bundled[bundling]["x"], bundled[bundling]["y"]
    if selected is None:  # with a selection only the airports are drawn
        figure.add_trace(go.Scattermap(lon=lon, lat=lat, mode="lines", hoverinfo="skip",
                                       line=dict(width=0.6, color=LINK), opacity=0.25))

    hubs = by_code[by_code["destinations"] > 0]
    figure.add_trace(go.Scattermap(
        lon=hubs["lon"], lat=hubs["lat"], mode="markers",
        customdata=[row_of_code[c] for c in hubs.index],
        marker=dict(size=3 + 1.6 * np.sqrt(hubs["destinations"]), color=INK, opacity=0.55 if selected is None else 0.2),
        hovertext=hubs["name"] + "<br>" + hubs["destinations"].astype(str) + " destinos", hoverinfo="text"))

    if selected is not None and airports.at[selected, "iata"] in by_code.index:
        code = airports.at[selected, "iata"]
        own = routes[(routes["a"] == code) | (routes["b"] == code)]
        destinations = by_code.loc[[b if a == code else a for a, b in zip(own["a"], own["b"])]]
        figure.add_trace(go.Scattermap(
            lon=destinations["lon"], lat=destinations["lat"], mode="markers",
            customdata=[row_of_code[c] for c in destinations.index],
            marker=dict(size=9, color=ACCENT, opacity=0.95),
            hovertext=destinations["name"] + "<br>" + destinations["city"], hoverinfo="text"))
        figure.add_trace(go.Scattermap(lon=[by_code.at[code, "lon"]], lat=[by_code.at[code, "lat"]],
                                       mode="markers", hoverinfo="skip", marker=dict(size=16, color=INK)))

    box = (window["x0"], window["y0"], window["x1"], window["y1"])
    stats = {"nodes": 0, "checked": 0}
    found = query(tree, *box, stats)
    inside = set(airports.iloc[found]["iata"]) - {""}
    shown = routes[routes["a"].isin(inside) | routes["b"].isin(inside)]
    stats.update(found=len(found), drawn=len(shown), drawn_as="rutas que tocan la pantalla")
    return base_map(figure, window), stats


# --------------------------------------------------------------------------
# Side panels
# --------------------------------------------------------------------------

def number(value, label):
    """Return a figure with a small label underneath."""
    return html.Div([html.Div(value, style={"fontSize": "22px", "fontWeight": "600", "lineHeight": "1.1"}),
                     html.Div(label, style={"fontSize": "12px", "color": MUTED})])


def numbers_panel(stats):
    """Return the four figures that describe the last query to the tree."""
    total = len(airports)
    share = 100 * stats["checked"] / total
    return html.Div(className="numbers", children=[
        number(f"{stats['found']:,}".replace(",", " "), "aeropuertos en pantalla"),
        number(f"{stats['drawn']:,}".replace(",", " "), stats["drawn_as"]),
        number(f"{stats['checked']:,}".replace(",", " "), f"revisados con el árbol, {share:.0f}% del total"),
        number(f"{stats['nodes']:,}".replace(",", " "), "cuadros visitados"),
    ])


def info(text):
    """Return a small info icon that shows ``text`` on hover."""
    return html.Span(" ⓘ", title=text, style={"color": MUTED, "cursor": "help", "fontWeight": "normal"})


def heading(text, help_text=None):
    """Return a small bold section heading with an optional info icon."""
    children = [text, info(help_text)] if help_text else [text]
    return html.Div(children, style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px"})


def control(label, help_text, component, width):
    """Wrap a control with a heading and an info icon."""
    return html.Div(style={"width": width}, children=[heading(label, help_text), component])


def chips(words):
    """Return a list of words as small rounded tags."""
    return html.Div([html.Span(w, style={"background": "#F0F4F8", "borderRadius": "12px", "padding": "2px 10px",
                                         "margin": "0 6px 6px 0", "display": "inline-block", "fontSize": "12px"})
                     for w in words])


def detail_panel(selected):
    """Describe the selected airport and link to it on OpenStreetMap.

    Args:
        selected: Row number of the selected airport.
    """
    row = airports.iloc[selected]
    osm = f"https://www.openstreetmap.org/?mlat={row['lat']:.4f}&mlon={row['lon']:.4f}#map=13/{row['lat']:.4f}/{row['lon']:.4f}"
    children = [
        html.Div(row["name"], style={"fontWeight": "600", "fontSize": "15px"}),
        html.Div(f"{row['city']}, {row['country']}" + (f"   ·   {row['iata']}" if row["iata"] else ""),
                 style={"color": MUTED, "margin": "2px 0 10px 0"}),
        html.Div(f"{row['destinations']} destinos directos", style={"marginBottom": "8px"}),
    ]
    if row["iata"]:
        own = routes[(routes["a"] == row["iata"]) | (routes["b"] == row["iata"])].sort_values("airlines",
                                                                                               ascending=False)
        others = [b if a == row["iata"] else a for a, b in zip(own["a"], own["b"])][:15]
        children.append(chips([f"{c} {by_code.at[c, 'city']}" for c in others]))
    children.append(html.A("Ver en OpenStreetMap", href=osm, target="_blank",
                           style={"color": LINK, "textDecoration": "none", "display": "inline-block",
                                  "marginTop": "8px"}))
    return html.Div(children)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = Dash(__name__, title="Vuelos")
server = app.server  # used by gunicorn

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { margin: 0; }
  .page { height: 100vh; display: flex; flex-direction: column; box-sizing: border-box;
          padding: 14px 28px; overflow: hidden; background: white; }
  .header { display: flex; align-items: flex-end; gap: 24px; flex-wrap: wrap;
            border-bottom: 1px solid #E4E7EB; padding-bottom: 12px; margin-bottom: 14px; }
  .spacer { flex: 1; }
  .numbers { display: flex; gap: 48px; flex-wrap: wrap; margin-bottom: 12px; }
  .main { flex: 1; min-height: 0; position: relative; }
  .map { position: absolute; inset: 0; }
  .detail { position: absolute; top: 14px; right: 14px; width: 300px; max-height: calc(100% - 28px);
            overflow-y: auto; box-sizing: border-box; padding: 14px 16px; font-size: 13px; background: white;
            border-radius: 8px; box-shadow: 0 2px 10px rgba(31, 41, 51, 0.15); }
  @media (max-width: 700px) {
    .page { height: auto; overflow: visible; padding: 14px 16px; }
    .numbers { gap: 16px 28px; }
    .main { flex: none; }
    .map { position: relative; height: 70vh !important; }
    .detail { position: static; width: auto; max-height: none; box-shadow: none; padding: 14px 0; }
  }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

DROPDOWN = {"width": "100%", "fontSize": "14px"}
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}

app.layout = html.Div(className="page", style={"fontFamily": FONT, "color": INK}, children=[
    html.Div(className="header", children=[
        html.Div("Vuelos", style={"fontFamily": "Georgia, serif", "fontStyle": "italic",
                                  "fontWeight": "700", "fontSize": "34px", "lineHeight": "1"}),
        html.Div(f"{len(airports):,} aeropuertos · {len(routes):,} rutas directas".replace(",", " "),
                 style={"borderLeft": f"1px solid {FAINT}", "paddingLeft": "16px", "color": MUTED,
                        "fontSize": "16px"}),
        html.Div(className="spacer"),
        control("Vista", "Aeropuertos usa el quadtree. Rutas muestra las conexiones directas.",
                dcc.Dropdown(["Aeropuertos", "Rutas"], "Aeropuertos", id="view", clearable=False, style=DROPDOWN),
                "150px"),
        control("Máximo de puntos", "Si en pantalla hay más aeropuertos, se agrupan por cuadro del árbol.",
                dcc.Dropdown([250, 1000, 10000], 1000, id="max_points", clearable=False, style=DROPDOWN),
                "150px"),
        control("Árbol", "Dibuja los bordes de las hojas del quadtree que están en pantalla.",
                dcc.Dropdown(["oculto", "visible"], "oculto", id="show_tree", clearable=False, style=DROPDOWN),
                "120px"),
        control("Agrupamiento", "Edge bundling de las rutas. Más agrupado se ve más limpio pero esconde más.",
                dcc.Dropdown(["ninguno", "poco", "mucho"], "poco", id="bundling", clearable=False,
                             style=DROPDOWN),
                "140px"),
        html.Button("Reset", id="reset", style={"background": INK, "color": "white", "border": "none",
                                                "borderRadius": "6px", "padding": "9px 20px",
                                                "fontSize": "14px", "cursor": "pointer"}),
    ]),

    dcc.Store(id="window", data=WORLD),
    dcc.Store(id="selected", data=None),

    html.Div(id="numbers"),

    html.Div(className="main", children=[
        dcc.Graph(id="map", className="map", config=GRAPH_CONFIG, style={"height": "100%"}),
        html.Div(id="detail", className="detail", style={"display": "none"}),
    ]),
])


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------

@app.callback(Output("window", "data"),
              Input("map", "relayoutData"), Input("reset", "n_clicks"), State("window", "data"))
def store_window(relayout, reset_clicks, current):
    """Keep the visible part of the map up to date."""
    if ctx.triggered_id == "reset":
        return WORLD
    return window_from_relayout(relayout, current)


@app.callback(Output("selected", "data"),
              Input("map", "clickData"), Input("reset", "n_clicks"), State("selected", "data"))
def select_airport(click, reset_clicks, current):
    """Store the airport clicked on the map.

    Clicking the selected airport again clears the selection.
    """
    if ctx.triggered_id == "reset":
        return None
    if not click:
        return current
    clicked = click["points"][0].get("customdata")
    if clicked is None:  # click on a cell or a line
        return current
    clicked = int(clicked[0] if isinstance(clicked, list) else clicked)
    return None if clicked == current else clicked


@app.callback(Output("view", "value"), Output("max_points", "value"), Output("show_tree", "value"),
              Output("bundling", "value"), Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_controls(reset_clicks):
    """Restore the default controls."""
    return "Aeropuertos", 1000, "oculto", "poco"


@app.callback(Output("max_points", "disabled"), Output("show_tree", "disabled"),
              Output("bundling", "disabled"), Input("view", "value"))
def enable_controls(view):
    """Enable only the controls that change the current view."""
    airports_view = view == "Aeropuertos"
    return not airports_view, not airports_view, airports_view


@app.callback(Output("map", "figure"), Output("numbers", "children"),
              Output("detail", "children"), Output("detail", "style"),
              Input("view", "value"), Input("max_points", "value"), Input("show_tree", "value"),
              Input("bundling", "value"), Input("window", "data"), Input("selected", "data"))
def update_dashboard(view, max_points, show_tree, bundling, window, selected):
    """Redraw the map, the figures and the detail card for the current controls and window."""
    if view == "Aeropuertos":
        figure, stats = airports_map(window, max_points, show_tree == "visible", selected)
    else:
        figure, stats = routes_map(window, bundling, selected)
    if selected is None:
        return figure, numbers_panel(stats), None, {"display": "none"}
    return figure, numbers_panel(stats), detail_panel(selected), {"display": "block"}


if __name__ == "__main__":
    app.run(debug=True)
