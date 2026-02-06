"""
Data models for the Bohai Sea Database.
"""

from typing import Optional
from datetime import date


class MarineStation:
    """Represents a marine monitoring station."""
    
    def __init__(self, db, station_id: Optional[int] = None):
        """
        Initialize a marine station.
        
        Args:
            db: Database instance
            station_id: Existing station ID (optional)
        """
        self.db = db
        self.id = station_id
        self.name = None
        self.latitude = None
        self.longitude = None
        self.depth = None
        self.description = None
        
        if station_id:
            self._load()
            
    def _load(self):
        """Load station data from database."""
        row = self.db.fetch_one(
            "SELECT * FROM marine_stations WHERE id = ?",
            (self.id,)
        )
        if row:
            self.name = row["name"]
            self.latitude = row["latitude"]
            self.longitude = row["longitude"]
            self.depth = row["depth"]
            self.description = row["description"]
            
    def save(self):
        """Save station to database."""
        if self.id:
            # Update existing
            self.db.execute(
                """UPDATE marine_stations 
                   SET name=?, latitude=?, longitude=?, depth=?, description=?
                   WHERE id=?""",
                (self.name, self.latitude, self.longitude, self.depth, 
                 self.description, self.id)
            )
        else:
            # Insert new
            cursor = self.db.execute(
                """INSERT INTO marine_stations (name, latitude, longitude, depth, description)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.name, self.latitude, self.longitude, self.depth, self.description)
            )
            self.id = cursor.lastrowid
        return self
        
    @staticmethod
    def get_all(db):
        """Get all marine stations."""
        rows = db.fetch_all("SELECT * FROM marine_stations")
        return [MarineStation._from_row(db, row) for row in rows]
        
    @staticmethod
    def _from_row(db, row):
        """Create station instance from database row."""
        station = MarineStation(db)
        station.id = row["id"]
        station.name = row["name"]
        station.latitude = row["latitude"]
        station.longitude = row["longitude"]
        station.depth = row["depth"]
        station.description = row["description"]
        return station


class WaterQuality:
    """Represents a water quality measurement."""
    
    def __init__(self, db, measurement_id: Optional[int] = None):
        """
        Initialize a water quality measurement.
        
        Args:
            db: Database instance
            measurement_id: Existing measurement ID (optional)
        """
        self.db = db
        self.id = measurement_id
        self.station_id = None
        self.measurement_date = None
        self.temperature = None
        self.salinity = None
        self.ph = None
        self.dissolved_oxygen = None
        self.turbidity = None
        
        if measurement_id:
            self._load()
            
    def _load(self):
        """Load measurement data from database."""
        row = self.db.fetch_one(
            "SELECT * FROM water_quality WHERE id = ?",
            (self.id,)
        )
        if row:
            self.station_id = row["station_id"]
            self.measurement_date = row["measurement_date"]
            self.temperature = row["temperature"]
            self.salinity = row["salinity"]
            self.ph = row["ph"]
            self.dissolved_oxygen = row["dissolved_oxygen"]
            self.turbidity = row["turbidity"]
            
    def save(self):
        """Save measurement to database."""
        if self.id:
            # Update existing
            self.db.execute(
                """UPDATE water_quality 
                   SET station_id=?, measurement_date=?, temperature=?, 
                       salinity=?, ph=?, dissolved_oxygen=?, turbidity=?
                   WHERE id=?""",
                (self.station_id, self.measurement_date, self.temperature,
                 self.salinity, self.ph, self.dissolved_oxygen, self.turbidity, self.id)
            )
        else:
            # Insert new
            cursor = self.db.execute(
                """INSERT INTO water_quality 
                   (station_id, measurement_date, temperature, salinity, ph, 
                    dissolved_oxygen, turbidity)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.station_id, self.measurement_date, self.temperature,
                 self.salinity, self.ph, self.dissolved_oxygen, self.turbidity)
            )
            self.id = cursor.lastrowid
        return self
        
    @staticmethod
    def get_by_station(db, station_id: int):
        """Get all measurements for a station."""
        rows = db.fetch_all(
            "SELECT * FROM water_quality WHERE station_id = ?",
            (station_id,)
        )
        return [WaterQuality._from_row(db, row) for row in rows]
        
    @staticmethod
    def _from_row(db, row):
        """Create measurement instance from database row."""
        measurement = WaterQuality(db)
        measurement.id = row["id"]
        measurement.station_id = row["station_id"]
        measurement.measurement_date = row["measurement_date"]
        measurement.temperature = row["temperature"]
        measurement.salinity = row["salinity"]
        measurement.ph = row["ph"]
        measurement.dissolved_oxygen = row["dissolved_oxygen"]
        measurement.turbidity = row["turbidity"]
        return measurement


class SeabedSample:
    """Represents a seabed sample."""
    
    def __init__(self, db, sample_id: Optional[int] = None):
        """
        Initialize a seabed sample.
        
        Args:
            db: Database instance
            sample_id: Existing sample ID (optional)
        """
        self.db = db
        self.id = sample_id
        self.station_id = None
        self.sample_date = None
        self.depth = None
        self.sediment_type = None
        self.organic_content = None
        self.grain_size = None
        self.notes = None
        
        if sample_id:
            self._load()
            
    def _load(self):
        """Load sample data from database."""
        row = self.db.fetch_one(
            "SELECT * FROM seabed_samples WHERE id = ?",
            (self.id,)
        )
        if row:
            self.station_id = row["station_id"]
            self.sample_date = row["sample_date"]
            self.depth = row["depth"]
            self.sediment_type = row["sediment_type"]
            self.organic_content = row["organic_content"]
            self.grain_size = row["grain_size"]
            self.notes = row["notes"]
            
    def save(self):
        """Save sample to database."""
        if self.id:
            # Update existing
            self.db.execute(
                """UPDATE seabed_samples 
                   SET station_id=?, sample_date=?, depth=?, sediment_type=?,
                       organic_content=?, grain_size=?, notes=?
                   WHERE id=?""",
                (self.station_id, self.sample_date, self.depth, self.sediment_type,
                 self.organic_content, self.grain_size, self.notes, self.id)
            )
        else:
            # Insert new
            cursor = self.db.execute(
                """INSERT INTO seabed_samples 
                   (station_id, sample_date, depth, sediment_type, 
                    organic_content, grain_size, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (self.station_id, self.sample_date, self.depth, self.sediment_type,
                 self.organic_content, self.grain_size, self.notes)
            )
            self.id = cursor.lastrowid
        return self
        
    @staticmethod
    def get_by_station(db, station_id: int):
        """Get all samples for a station."""
        rows = db.fetch_all(
            "SELECT * FROM seabed_samples WHERE station_id = ?",
            (station_id,)
        )
        return [SeabedSample._from_row(db, row) for row in rows]
        
    @staticmethod
    def _from_row(db, row):
        """Create sample instance from database row."""
        sample = SeabedSample(db)
        sample.id = row["id"]
        sample.station_id = row["station_id"]
        sample.sample_date = row["sample_date"]
        sample.depth = row["depth"]
        sample.sediment_type = row["sediment_type"]
        sample.organic_content = row["organic_content"]
        sample.grain_size = row["grain_size"]
        sample.notes = row["notes"]
        return sample
