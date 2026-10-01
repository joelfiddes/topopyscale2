"""Bounding box handling for spatial subsets."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BBox:
    """Bounding box in (west, south, east, north) format.

    Coordinates are in degrees. Longitude is in -180:180 by default but
    can be converted to 0:360 for Google ARCO-ERA5.

    Args:
        west: Western longitude.
        south: Southern latitude.
        east: Eastern longitude.
        north: Northern latitude.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self):
        if self.south > self.north:
            logger.warning("south > north, swapping: %s > %s", self.south, self.north)
            object.__setattr__(self, "south", self.north)
            object.__setattr__(self, "north", self.south)
        if self.west > self.east:
            # Only swap if both are in the same hemisphere — otherwise it's
            # a wrap-around box that we don't support yet.
            if not (self.west > 0 and self.east < 0):
                logger.warning("west > east, swapping: %s > %s", self.west, self.east)
                object.__setattr__(self, "west", self.east)
                object.__setattr__(self, "east", self.west)

    @classmethod
    def from_tuple(cls, bbox: tuple[float, float, float, float]) -> "BBox":
        """Create from (west, south, east, north) tuple."""
        return cls(*bbox)

    def to_cds_area(self) -> list[str]:
        """Convert to CDS API area format: [N, W, S, E] as strings."""
        return [str(self.north), str(self.west), str(self.south), str(self.east)]

    def to_0_360(self) -> "BBox":
        """Convert longitudes from -180:180 to 0:360 for Google ARCO-ERA5."""
        w = self.west % 360
        e = self.east % 360
        if w == 180 and e == 180:
            w, e = 0, 360
        return BBox(west=w, south=self.south, east=e, north=self.north)

    def pad(self, degrees: float = 0.4) -> "BBox":
        """Expand the bounding box by a given number of degrees."""
        return BBox(
            west=self.west - degrees,
            south=self.south - degrees,
            east=self.east + degrees,
            north=self.north + degrees,
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as (west, south, east, north) tuple."""
        return (self.west, self.south, self.east, self.north)
