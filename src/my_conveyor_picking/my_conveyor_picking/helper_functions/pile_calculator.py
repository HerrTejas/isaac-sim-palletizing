"""Pile position calculator for palletizing operations.

Calculates box positions in a 3D pile grid based on configurable parameters.
"""

from typing import Optional


class PileCalculator:
    """Calculates positions for boxes in a palletizing pile.

    The pile is organized as a 3D grid with configurable:
    - Starting position (first box center)
    - Box dimensions
    - Gaps between boxes on each axis
    - Stacking direction on each axis
    - Number of boxes on each axis
    """

    def __init__(
        self,
        starting_position: list[float],
        box_size: list[float],
        x_gap: float = 0.0,
        y_gap: float = 0.0,
        z_gap: float = 0.0,
        x_direction: int = 1,
        y_direction: int = 1,
        z_direction: int = 1,
        box_x_count: int = 1,
        box_y_count: int = 1,
        box_z_count: int = 1
    ) -> None:
        """Initialize the pile calculator.

        Args:
            starting_position: First box center position [x, y, z] in meters.
            box_size: Box dimensions [x, y, z] in meters.
            x_gap: Gap between boxes along X axis in meters.
            y_gap: Gap between boxes along Y axis in meters.
            z_gap: Gap between boxes along Z axis in meters.
            x_direction: Stacking direction on X (-1 or 1).
            y_direction: Stacking direction on Y (-1 or 1).
            z_direction: Stacking direction on Z (-1 or 1).
            box_x_count: Number of boxes along X axis.
            box_y_count: Number of boxes along Y axis.
            box_z_count: Number of layers along Z axis.
        """
        self._starting_position = starting_position
        self._box_size = box_size
        self._x_gap = x_gap
        self._y_gap = y_gap
        self._z_gap = z_gap
        self._x_direction = x_direction
        self._y_direction = y_direction
        self._z_direction = z_direction
        self._box_x_count = box_x_count
        self._box_y_count = box_y_count
        self._box_z_count = box_z_count

        # Current box indices
        self._x_index = 0
        self._y_index = 0
        self._z_index = 0

        # Total boxes placed
        self._boxes_placed = 0
        self._total_capacity = box_x_count * box_y_count * box_z_count

    def get_position(
        self,
        x_index: int,
        y_index: int,
        z_index: int
    ) -> list[float]:
        """Calculate position for a box at given indices.

        Args:
            x_index: Index along X axis (0 to box_x_count-1).
            y_index: Index along Y axis (0 to box_y_count-1).
            z_index: Index along Z axis (0 to box_z_count-1).

        Returns:
            Box center position [x, y, z] in meters.
        """
        x = (
            self._starting_position[0] +
            self._x_direction * x_index * (self._box_size[0] + self._x_gap)
        )
        y = (
            self._starting_position[1] +
            self._y_direction * y_index * (self._box_size[1] + self._y_gap)
        )
        z = (
            self._starting_position[2] +
            self._z_direction * z_index * (self._box_size[2] + self._z_gap)
        )
        return [x, y, z]

    def get_next_position(self) -> Optional[list[float]]:
        """Get the position for the next box to place.

        Fills the pile in order: X -> Y -> Z (completes each row,
        then each layer, then moves up).

        Returns:
            Box center position [x, y, z] or None if pile is full.
        """
        if self.is_full():
            return None

        position = self.get_position(
            self._x_index,
            self._y_index,
            self._z_index
        )

        # Advance indices for next call
        self._advance_indices()

        return position

    def _advance_indices(self) -> None:
        """Advance indices to the next position in the pile."""
        self._boxes_placed += 1
        self._x_index += 1

        # Move to next row (Y)
        if self._x_index >= self._box_x_count:
            self._x_index = 0
            self._y_index += 1

            # Move to next layer (Z)
            if self._y_index >= self._box_y_count:
                self._y_index = 0
                self._z_index += 1

    def is_full(self) -> bool:
        """Check if the pile is at full capacity.

        Returns:
            True if no more boxes can be placed.
        """
        return self._boxes_placed >= self._total_capacity

    def reset(self) -> None:
        """Reset the pile to empty state."""
        self._x_index = 0
        self._y_index = 0
        self._z_index = 0
        self._boxes_placed = 0

    @property
    def boxes_placed(self) -> int:
        """Get the number of boxes already placed."""
        return self._boxes_placed

    @property
    def total_capacity(self) -> int:
        """Get the total pile capacity."""
        return self._total_capacity

    @property
    def remaining_capacity(self) -> int:
        """Get the remaining pile capacity."""
        return self._total_capacity - self._boxes_placed
