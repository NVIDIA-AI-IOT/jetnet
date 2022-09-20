from pydantic import BaseModel
from jetnet.point import Point

from typing import Sequence


class Polygon(BaseModel):
    points: Sequence[Point]