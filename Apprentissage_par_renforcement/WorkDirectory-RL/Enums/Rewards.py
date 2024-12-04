# rewards.py
from enum import Enum

# Simple rewards
class Rewards(Enum):
    NORMAL = 0
    ENEMIES = -2
    END = 2
    WALL = -1

# More efficient rewards
# class Rewards(Enum):
#     NORMAL = -1
#     ENEMIES = -10
#     END = 30
#     WALL = -2