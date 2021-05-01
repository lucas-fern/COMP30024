from enum import Enum, auto

ADJACENT_OFFSETS = ((+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1))

KILL_RELATIONS = {'r': 'p', 'p': 's', 's': 'r'}


class Team(Enum):
    UPPER = auto()
    LOWER = auto()
    BLOCK = None


class Move(Enum):
    SLIDE = auto()
    SWING = auto()
    THROW = auto()


def centered_to_array_coord(centered_coord: tuple, board_radius) -> tuple:
    """Converts an axial coordinate with (0, 0) in the center of the grid to an axial coordinate with (0, 0) in the
    top left hex. Reverses the positive row direction so that increasing the row moves down the grid.

    Example:
        Centered Coordinates:
            _,-' `-._,-' `-._
            |       |       |
            |  1,-1 |  1, 0 |
        _,-' `-._,-' `-._,-' `-._
        |       |       |       |
        |  0,-1 |  0, 0 |  0, 1 |
         `-._,-' `-._,-' `-._,-'
            |       |       |
            | -1, 0 | -1, 1 |
             `-._,-' `-._,-'

        Array Coordinates:
            _,-' `-._,-' `-._
            |       |       |
            |  0, 0 |  0, 1 |
        _,-' `-._,-' `-._,-' `-._
        |       |       |       |
        |  1, 0 |  1, 1 |  1, 2 |
         `-._,-' `-._,-' `-._,-'
            |       |       |
            |  2, 1 |  2, 2 |
             `-._,-' `-._,-'

        Array Coordinates in Array:
        +--------+--------+--------+
        | (0, 0) | (0, 1) | None   |
        +--------+--------+--------+
        | (1, 0) | (1, 1) | (1, 2) |
        +--------+--------+--------+
        | None   | (2, 1) | (2, 2) |
        +--------+--------+--------+
            > Notice how the array coordinates correspond to a real array index that we can plug straight into
            > a numpy array to access that hex! nice

        """
    offset = (-(board_radius - 1), (board_radius - 1))
    translated = [sum(x) for x in zip(centered_coord, offset)]

    return -translated[0], translated[1]  # Element wise sum of the coordinate and offset


def centered_to_cube_coord(centered_coord: tuple) -> tuple:
    """Converts centered coords to cubic notation for use in manhattan distance calculation"""
    return centered_coord[0], centered_coord[1], 0 - centered_coord[0] - centered_coord[1]


def array_to_centered_coord(array_coord: tuple, board_radius) -> tuple:
    """Performs the inverse transformation to centered_to_array_coord(). Transforms coordinates from the array
    coordinate system to centered axial coordinates."""
    inverted = (-array_coord[0], array_coord[1])
    offset = ((board_radius - 1), -(board_radius - 1))

    return tuple(sum(x) for x in zip(inverted, offset))  # Element wise sum of the coordinate and offset


def get_adjacent_hexes(centered_coord, board_radius):
    """Returns a set of the valid hexes adjacent to the piece, takes a coordinate as an optional keyword argument
    and uses that coordinate instead of this piece's if provided. Might turn into a @staticmethod?

    This code is actually kind of sexy even if it is a bit complicated."""

    # Zips together the coordinate with each valid adjacent offset (makes an iterator of tuple pairs)
    pairs = [(centered_coord, i) for i in ADJACENT_OFFSETS]
    # Zips together the tuple pairs so that the row coordinates and the column coordinates are together
    zipped_pairs = [list(zip(*i)) for i in pairs]
    # Sums the row and column coordinates with the offsets and forms them into a coordinate tuple
    adjacent_hexes = [(sum(i[0]), sum(i[1])) for i in zipped_pairs]
    # Keeps only the coordinates which are on the board
    adjacent_hexes = [i for i in adjacent_hexes if valid_centered_hex(i, board_radius)]

    return set(adjacent_hexes)


def valid_centered_hex(centered_coord, board_radius):
    return -board_radius < sum(centered_coord) < board_radius and \
           -board_radius < centered_coord[0] < board_radius and \
           -board_radius < centered_coord[1] < board_radius


def get_valid_slides(centered_coord, board_radius, blocked_coords):
    """Returns a list of coordinates which a given centered coordinate can perform a slide move to."""
    return get_adjacent_hexes(centered_coord, board_radius) - blocked_coords  # - here is set difference


def get_valid_swings(centered_coord, identifier, board_grid, board_radius, blocked_coords):
    """Returns a list of coordinates which a given centered coordinate can perform a swing move to."""
    valid_swings = set()
    for coord in get_adjacent_friendlies(centered_coord, identifier, board_grid, board_radius):
        valid_swings |= get_adjacent_hexes(coord, board_radius)

    # Valid swings are any hex adjacent to an adjacent friendly which isn't blocked, slideable, or current.
    return valid_swings - blocked_coords - get_valid_slides(centered_coord, board_radius, blocked_coords) \
        - {centered_coord}


def get_valid_moves(centered_coord, identifier, board_grid, board_radius, blocked_coords):
    """Returns the union of valid slide and swing moves for a piece from a board coordinate."""
    return get_valid_swings(centered_coord, identifier, board_grid, board_radius, blocked_coords) | \
           get_valid_slides(centered_coord, board_radius, blocked_coords)


def get_adjacent_friendlies(centered_coord, identifier, board_grid, board_radius):
    """Given a token symbol, board, and a coordinate, returns the coordinates adjacent which contain friendly tokens."""
    adjacent_friendlies = set()
    for row, col in get_adjacent_hexes(centered_coord, board_radius):
        array_coord = centered_to_array_coord((row, col), board_radius)
        for piece in board_grid[array_coord]:
            if get_team(piece) == get_team(identifier):
                adjacent_friendlies |= {(row, col)}
                break

    return adjacent_friendlies


def get_team(identifier):
    """Returns the Team enum for a given identifier / symbol."""
    if identifier.isupper():
        return Team.UPPER
    elif identifier.islower():
        return Team.LOWER
    else:
        return Team.BLOCK


def manhattan_distance(a, b):
    length_a = len(a)
    if length_a != len(b):
        raise Exception('Mismatched vector dimensions for distance calculation.')

    distance = 0

    for i in range(length_a):
        distance += abs(a[i] - b[i])

    return distance


def manhattan_distance_cube(a, b):
    a, b = centered_to_cube_coord(a), centered_to_cube_coord(b)
    length_a = len(a)
    if length_a != len(b):
        raise Exception('Mismatched vector dimensions for distance calculation.')
    distance = 0

    for i in range(length_a):
        distance += abs(a[i] - b[i])

    return distance/2
