# rewards.py
from enum import Enum

# Simple rewards
# class Rewards(Enum):
#     NORMAL = 0
#     ENEMIES = -2
#     END = 2
#     WALL = -1
#     LOOP = -5

# More efficient rewards
# class Rewards(Enum):
#     NORMAL = -1
#     ENEMIES = -10
#     END = 30
#     WALL = -2
#     LOOP = -15

# Deep rewards
class Rewards(Enum):
    NORMAL = -1
    ENEMIES = -20
    END = 100
    WALL = -5
    LOOP = -25