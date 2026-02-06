"""
Test suite for Bohai Sea Database.
"""

import unittest
import os
import sys
from datetime import date

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bohai_sdb import Database, MarineStation, WaterQuality, SeabedSample


class TestDatabase(unittest.TestCase):
    """Test Database class."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db = "test_bohai.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_database_creation(self):
        """Test database creation and connection."""
        with Database(self.test_db) as db:
            db.create_tables()
            self.assertIsNotNone(db.conn)
            
    def test_context_manager(self):
        """Test database context manager."""
        with Database(self.test_db) as db:
            self.assertIsNotNone(db.conn)
        # Connection should be closed after context
        self.assertIsNone(db.conn)


class TestMarineStation(unittest.TestCase):
    """Test MarineStation model."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db = "test_bohai.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = Database(self.test_db).connect()
        self.db.create_tables()
        
    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_create_station(self):
        """Test creating a marine station."""
        station = MarineStation(self.db)
        station.name = "Test Station"
        station.latitude = 38.5
        station.longitude = 117.5
        station.depth = 20.0
        station.description = "Test description"
        station.save()
        
        self.assertIsNotNone(station.id)
        self.assertEqual(station.name, "Test Station")
        
    def test_load_station(self):
        """Test loading a station from database."""
        # Create station
        station1 = MarineStation(self.db)
        station1.name = "Station 1"
        station1.latitude = 38.5
        station1.longitude = 117.5
        station1.save()
        station_id = station1.id
        
        # Load station
        station2 = MarineStation(self.db, station_id)
        self.assertEqual(station2.name, "Station 1")
        self.assertEqual(station2.latitude, 38.5)
        
    def test_get_all_stations(self):
        """Test getting all stations."""
        # Create multiple stations
        for i in range(3):
            station = MarineStation(self.db)
            station.name = f"Station {i}"
            station.latitude = 38.0 + i
            station.longitude = 117.0 + i
            station.save()
            
        stations = MarineStation.get_all(self.db)
        self.assertEqual(len(stations), 3)


class TestWaterQuality(unittest.TestCase):
    """Test WaterQuality model."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db = "test_bohai.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = Database(self.test_db).connect()
        self.db.create_tables()
        
        # Create a test station
        self.station = MarineStation(self.db)
        self.station.name = "Test Station"
        self.station.latitude = 38.5
        self.station.longitude = 117.5
        self.station.save()
        
    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_create_measurement(self):
        """Test creating a water quality measurement."""
        measurement = WaterQuality(self.db)
        measurement.station_id = self.station.id
        measurement.measurement_date = date.today().isoformat()
        measurement.temperature = 15.0
        measurement.salinity = 30.0
        measurement.ph = 8.0
        measurement.save()
        
        self.assertIsNotNone(measurement.id)
        self.assertEqual(measurement.temperature, 15.0)
        
    def test_get_by_station(self):
        """Test getting measurements by station."""
        # Create multiple measurements
        for i in range(3):
            measurement = WaterQuality(self.db)
            measurement.station_id = self.station.id
            measurement.measurement_date = date.today().isoformat()
            measurement.temperature = 15.0 + i
            measurement.save()
            
        measurements = WaterQuality.get_by_station(self.db, self.station.id)
        self.assertEqual(len(measurements), 3)


class TestSeabedSample(unittest.TestCase):
    """Test SeabedSample model."""
    
    def setUp(self):
        """Set up test database."""
        self.test_db = "test_bohai.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.db = Database(self.test_db).connect()
        self.db.create_tables()
        
        # Create a test station
        self.station = MarineStation(self.db)
        self.station.name = "Test Station"
        self.station.latitude = 38.5
        self.station.longitude = 117.5
        self.station.save()
        
    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_create_sample(self):
        """Test creating a seabed sample."""
        sample = SeabedSample(self.db)
        sample.station_id = self.station.id
        sample.sample_date = date.today().isoformat()
        sample.depth = 10.0
        sample.sediment_type = "Sand"
        sample.save()
        
        self.assertIsNotNone(sample.id)
        self.assertEqual(sample.sediment_type, "Sand")
        
    def test_get_by_station(self):
        """Test getting samples by station."""
        # Create multiple samples
        for i in range(3):
            sample = SeabedSample(self.db)
            sample.station_id = self.station.id
            sample.sample_date = date.today().isoformat()
            sample.depth = 10.0 + i
            sample.sediment_type = f"Type {i}"
            sample.save()
            
        samples = SeabedSample.get_by_station(self.db, self.station.id)
        self.assertEqual(len(samples), 3)


if __name__ == "__main__":
    unittest.main()
