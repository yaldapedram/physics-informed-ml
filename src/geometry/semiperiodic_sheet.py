import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# -----------------------------
# Parameters you can change
# -----------------------------
THETA_DEG = 61.0          # rotation angle between two sheets (degrees)
Z_DIST = 5.44             # vertical separation between the two sheets
RADIUS = 5.19             # lattice spacing in x-direction (also controls y-spacing)
W_MAX = 51.50             # sheet width (x extent) used to generate grid
H_MAX = 50.0              # sheet height (y extent) used to generate grid
EDGE_OFFSET = 2.595       # how far the artificial "edge columns" sit from the sheet

def plate_with_edges(radius: float, w_max: float, h_max: float, edge_offset: float = 2.595):
    """
    Build a semi-periodic 2D sheet and add "edge" particles along left/right boundaries.

    This keeps the original behavior:
      - Creates a 2D grid with spacing:
          w = radius
          h = sqrt(3)*radius/2
      - Applies an x-shift of +radius/2 to every other row (implemented via the original while-loop)
      - Adds edge points at x = -edge_offset and x = max(x)+edge_offset for specific rows
      - Shifts all x coordinates by +edge_offset at the end (so x is mostly positive)

    Parameters
    ----------
    radius : float
        Base spacing in x-direction.
    w_max, h_max : float
        Extents used to generate the grid via np.arange(0, max, step).
    edge_offset : float
        Offset used for the artificial edge columns (default matches original 2.595).

    Returns
    -------
    coordinates : (N, 2) np.ndarray
        Central sheet coordinates (x, y).
    edge_coordinates : (M, 2) np.ndarray
        Edge coordinates (x, y).
    """
    w = radius
    h = np.sqrt(3) * radius / 2

    coordinates = []
    edge_x = []
    edge_y = []

    # Build the 2D grid. 'f' ends up being the number of x-points per row.
    for y in np.arange(0, h_max, h):
        f = 0
        for x in np.arange(0, w_max, w):
            coordinates.append((x, y))
            f += 1

    coordinates = np.array(coordinates, dtype=float)

    # Stagger: shift blocks of length f by +radius/2 every other row (original logic)
    i = 0
    while i < (len(coordinates[:, 0]) + f * 2):
        coordinates[i : i + f, 0] += radius / 2
        i += f * 2

    # Add edge particles (original logic)
    for i in range(len(coordinates)):
        if i == 0:
            edge_x.append(-edge_offset)
            edge_y.append(coordinates[i, 1])

        elif i % f == 0 and i % (2 * f) == 0:
            # left edge at current row start
            edge_x.append(-edge_offset)
            edge_y.append(coordinates[i, 1])

            # right edge at previous row end (uses i-1)
            edge_x.append(np.max(coordinates[:, 0]) + edge_offset)
            edge_y.append(coordinates[i - 1, 1])

    # Final right-edge point if last row is not "full" (original logic)
    if coordinates[-1, 0] < np.max(coordinates[:, 0]):
        edge_x.append(np.max(coordinates[:, 0]) + edge_offset)
        edge_y.append(coordinates[len(coordinates) - 1, 1])

    edge_coordinates = np.column_stack((edge_x, edge_y)).astype(float)

    # Shift everything in x by +edge_offset (original behavior)
    coordinates[:, 0] += edge_offset
    edge_coordinates[:, 0] += edge_offset

    return coordinates, edge_coordinates



def build_semiperiodic_sheet(radius: float, w_max: float, h_max: float, edge_offset: float = 2.595):
    """
    Build a semi-periodic 2D sheet and add "edge" particles along left/right boundaries.
    Returns:
      - central_coordinates: (N, 2) array of (x, y) points
      - edge_coordinates:    (M, 2) array of (x, y) edge points
    """
    w = radius
    h = np.sqrt(3) * radius / 2

    central_coordinates = []
    edge_x = []
    edge_y = []

    # Build the 2D grid. 'f' ends up being the number of x-points per row.
    for y in np.arange(0, h_max, h):
        f = 0
        for x in np.arange(0, w_max, w):
            central_coordinates.append((x, y))
            f += 1

    central_coordinates = np.array(central_coordinates, dtype=float)

    # Stagger: shift blocks of length f by +radius/2 every other row (original logic)
    i = 0
    while i < (len(central_coordinates[:, 0]) + f * 2):
        central_coordinates[i : i + f, 0] += radius / 2
        i += f * 2

    # Add edge particles (original logic)
    for i in range(len(central_coordinates)):
        if i == 0:
            edge_x.append(-edge_offset)
            edge_y.append(central_coordinates[i, 1])

        elif i % f == 0 and i % (2 * f) == 0:
            # left edge at current row start
            edge_x.append(-edge_offset)
            edge_y.append(central_coordinates[i, 1])

            # right edge at previous row end (uses i-1)
            edge_x.append(np.max(central_coordinates[:, 0]) + edge_offset)
            edge_y.append(central_coordinates[i - 1, 1])

    # Final right-edge point if last row is not "full" (original logic)
    if central_coordinates[-1, 0] < np.max(central_coordinates[:, 0]):
        edge_x.append(np.max(central_coordinates[:, 0]) + edge_offset)
        edge_y.append(central_coordinates[-1, 1])

    edge_coordinates = np.column_stack((edge_x, edge_y)).astype(float)

    # Shift everything in x by +edge_offset (original behavior)
    central_coordinates[:, 0] += edge_offset
    edge_coordinates[:, 0] += edge_offset

    return central_coordinates, edge_coordinates



def plot_2d(central_coord: np.ndarray, edge_coord: np.ndarray, major_x: float, major_y: float):
    """Quick 2D visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(central_coord[:, 0], central_coord[:, 1], "o", markersize=3, label="Central")
    ax.plot(edge_coord[:, 0], edge_coord[:, 1], "o", markersize=3, label="Edge")

    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.grid(which="major", linestyle="--", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()


def plot_two_sheets_3d(flat_central, flat_edge, flat2_central, flat2_edge):
    """3D plot of two sheets separated in z and shifted in x."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(flat_central[:, 0], flat_central[:, 1], flat_central[:, 2], label="Central", s=8)
    ax.scatter(flat2_central[:, 0], flat2_central[:, 1], flat2_central[:, 2], s=8)

    ax.scatter(flat_edge[:, 0], flat_edge[:, 1], flat_edge[:, 2], label="Edge", s=8)
    ax.scatter(flat2_edge[:, 0], flat2_edge[:, 1], flat2_edge[:, 2], s=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


def main():
    theta = np.deg2rad(THETA_DEG)

    # Geometry: the second sheet is shifted in x based on theta and z separation
    x_shift = Z_DIST / np.tan(theta)

    central_coord, edge_coord = build_semiperiodic_sheet(RADIUS, W_MAX, H_MAX, EDGE_OFFSET)

    # 2D plot
    major_x = EDGE_OFFSET
    major_y = np.sqrt(3) * RADIUS / 2
    plot_2d(central_coord, edge_coord, major_x, major_y)

    # Convert to 3D (z=0)
    flat_central = np.column_stack((central_coord, np.zeros(len(central_coord))))
    flat_edge = np.column_stack((edge_coord, np.zeros(len(edge_coord))))

    # Second sheet: translated in z, shifted in x
    flat2_central = flat_central.copy()
    flat2_central[:, 2] += Z_DIST
    flat2_central[:, 0] -= x_shift

    flat2_edge = flat_edge.copy()
    flat2_edge[:, 2] += Z_DIST
    flat2_edge[:, 0] -= x_shift

    # 3D plot
    plot_two_sheets_3d(flat_central, flat_edge, flat2_central, flat2_edge)

    # Counts
    print("Central atoms:", len(flat_central) + len(flat2_central))
    print("Edge atoms:", len(flat_edge) + len(flat2_edge))
    print("Total atoms:", len(flat_central) + len(flat_edge) + len(flat2_central) + len(flat2_edge))


if __name__ == "__main__":
    main()
