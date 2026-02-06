"""
Bohai Sea Database (bohai_sdb)
A lightweight database management system for marine environmental data.
"""

__version__ = "0.1.0"
__author__ = "Bohai SDB Team"

from .database import Database
from .models import MarineStation, WaterQuality, SeabedSample

__all__ = ["Database", "MarineStation", "WaterQuality", "SeabedSample"]
