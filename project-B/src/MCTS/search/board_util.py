import numpy as np

RADIUS = 5

ADJACENT_OFFSETS = ((+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1))

KILL_RELATIONS = {'r': 'p', 'p': 's', 's': 'r'}

AXIAL_TO_LINEAR = {
    (4, -4): 0,
    (4, -3): 1,
    (4, -2): 2,
    (4, -1): 3,
    (4, 0): 4,
    (3, -4): 5,
    (3, -3): 6,
    (3, -2): 7,
    (3, -1): 8,
    (3, 0): 9,
    (3, 1): 10,
    (2, -4): 11,
    (2, -3): 12,
    (2, -2): 13,
    (2, -1): 14,
    (2, 0): 15,
    (2, 1): 16,
    (2, 2): 17,
    (1, -4): 18,
    (1, -3): 19,
    (1, -2): 20,
    (1, -1): 21,
    (1, 0): 22,
    (1, 1): 23,
    (1, 2): 24,
    (1, 3): 25,
    (0, -4): 26,
    (0, -3): 27,
    (0, -2): 28,
    (0, -1): 29,
    (0, 0): 30,
    (0, 1): 31,
    (0, 2): 32,
    (0, 3): 33,
    (0, 4): 34,
    (-1, -3): 35,
    (-1, -2): 36,
    (-1, -1): 37,
    (-1, 0): 38,
    (-1, 1): 39,
    (-1, 2): 40,
    (-1, 3): 41,
    (-1, 4): 42,
    (-2, -2): 43,
    (-2, -1): 44,
    (-2, 0): 45,
    (-2, 1): 46,
    (-2, 2): 47,
    (-2, 3): 48,
    (-2, 4): 49,
    (-3, -1): 50,
    (-3, 0): 51,
    (-3, 1): 52,
    (-3, 2): 53,
    (-3, 3): 54,
    (-3, 4): 55,
    (-4, 0): 56,
    (-4, 1): 57,
    (-4, 2): 58,
    (-4, 3): 59,
    (-4, 4): 60
}

LINEAR_TO_AXIAL = {v: k for k, v in AXIAL_TO_LINEAR.items()}


def get_adjacent_hexes(axial_coord):
    """Returns a set of the valid hexes adjacent to the piece, takes a coordinate as an optional keyword argument
    and uses that coordinate instead of this piece's if provided.

    This code is actually kind of sexy even if it is a bit complicated."""

    # Zips together the coordinate with each valid adjacent offset (makes an iterator of tuple pairs)
    pairs = [(axial_coord, i) for i in ADJACENT_OFFSETS]
    # Zips together the tuple pairs so that the row coordinates and the column coordinates are together
    zipped_pairs = [list(zip(*i)) for i in pairs]
    # Sums the row and column coordinates with the offsets and forms them into a coordinate tuple
    adjacent_hexes = [(sum(i[0]), sum(i[1])) for i in zipped_pairs]
    # Keeps only the coordinates which are on the board
    adjacent_hexes = [i for i in adjacent_hexes if valid_axial_hex(i)]

    return set(adjacent_hexes)


def valid_axial_hex(axial_coord):
    """Validates a hex on the game board in the axial coordinate system."""
    return -RADIUS < sum(axial_coord) < RADIUS and \
           -RADIUS < axial_coord[0] < RADIUS and \
           -RADIUS < axial_coord[1] < RADIUS


def get_valid_slides(axial_coord):
    """Returns a list of coordinates which a given axial coordinate can perform a slide move to."""
    return get_adjacent_hexes(axial_coord)  # - here is set difference


def get_valid_swings(axial_coord, player_n_pieces):
    """Returns a list of coordinates which a given axial coordinate can perform a swing move to."""
    valid_swings = set()
    for coord in get_adjacent_friendlies(axial_coord, player_n_pieces):
        valid_swings |= get_adjacent_hexes(coord)

    # Valid swings are any hex adjacent to an adjacent friendly which isn't blocked, slideable, or current.
    return valid_swings - get_valid_slides(axial_coord) - {axial_coord}


def get_adjacent_friendlies(axial_coord, friendly_pieces):
    """Given a token symbol, board, and a coordinate, returns the coordinates adjacent which contain friendly tokens."""
    adjacent_friendlies = set()
    for row, col in get_adjacent_hexes(axial_coord):
        linear_coord = AXIAL_TO_LINEAR[row, col]

        if np.any(friendly_pieces[linear_coord]):
            adjacent_friendlies |= {(row, col)}

    return adjacent_friendlies
