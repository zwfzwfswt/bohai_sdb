#!/usr/bin/env python3
"""
Bohai Sea Seabed Database (bohai_sdb)
A simple database system for managing Bohai Sea seabed engineering geology data.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class BohaiSDB:
    """Main class for Bohai Sea Seabed Database."""
    
    def __init__(self, db_path: str = "bohai_sdb.db"):
        """Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Survey Areas table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS survey_areas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                water_depth REAL,
                survey_date TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Seabed Samples table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS seabed_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                survey_area_id INTEGER,
                sample_id TEXT UNIQUE NOT NULL,
                depth REAL NOT NULL,
                soil_type TEXT,
                moisture_content REAL,
                density REAL,
                porosity REAL,
                collection_date TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (survey_area_id) REFERENCES survey_areas(id)
            )
        """)
        
        # Geotechnical Measurements table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS geotechnical_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER,
                measurement_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                measurement_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id) REFERENCES seabed_samples(id)
            )
        """)
        
        self.conn.commit()
    
    def add_survey_area(self, name: str, latitude: float, longitude: float, 
                       water_depth: Optional[float] = None,
                       survey_date: Optional[str] = None,
                       description: Optional[str] = None) -> int:
        """Add a new survey area.
        
        Args:
            name: Name of the survey area
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            water_depth: Water depth in meters
            survey_date: Date of survey (ISO format)
            description: Additional description
            
        Returns:
            ID of the newly created survey area
        """
        self.cursor.execute("""
            INSERT INTO survey_areas (name, latitude, longitude, water_depth, survey_date, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, latitude, longitude, water_depth, survey_date, description))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_seabed_sample(self, survey_area_id: int, sample_id: str, depth: float,
                         soil_type: Optional[str] = None,
                         moisture_content: Optional[float] = None,
                         density: Optional[float] = None,
                         porosity: Optional[float] = None,
                         collection_date: Optional[str] = None,
                         notes: Optional[str] = None) -> int:
        """Add a new seabed sample.
        
        Args:
            survey_area_id: ID of the associated survey area
            sample_id: Unique identifier for the sample
            depth: Depth of sample collection in meters
            soil_type: Type of soil
            moisture_content: Moisture content percentage
            density: Density in g/cm³
            porosity: Porosity percentage
            collection_date: Date of collection (ISO format)
            notes: Additional notes
            
        Returns:
            ID of the newly created sample
        """
        self.cursor.execute("""
            INSERT INTO seabed_samples 
            (survey_area_id, sample_id, depth, soil_type, moisture_content, 
             density, porosity, collection_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (survey_area_id, sample_id, depth, soil_type, moisture_content,
              density, porosity, collection_date, notes))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_geotechnical_measurement(self, sample_id: int, measurement_type: str,
                                    value: float, unit: str,
                                    measurement_date: Optional[str] = None) -> int:
        """Add a geotechnical measurement.
        
        Args:
            sample_id: ID of the associated sample
            measurement_type: Type of measurement (e.g., 'shear_strength', 'compression')
            value: Measurement value
            unit: Unit of measurement
            measurement_date: Date of measurement (ISO format)
            
        Returns:
            ID of the newly created measurement
        """
        self.cursor.execute("""
            INSERT INTO geotechnical_measurements 
            (sample_id, measurement_type, value, unit, measurement_date)
            VALUES (?, ?, ?, ?, ?)
        """, (sample_id, measurement_type, value, unit, measurement_date))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_survey_areas(self) -> List[Dict]:
        """Get all survey areas.
        
        Returns:
            List of survey area dictionaries
        """
        self.cursor.execute("SELECT * FROM survey_areas")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_seabed_samples(self, survey_area_id: Optional[int] = None) -> List[Dict]:
        """Get seabed samples, optionally filtered by survey area.
        
        Args:
            survey_area_id: Optional survey area ID to filter by
            
        Returns:
            List of sample dictionaries
        """
        if survey_area_id:
            self.cursor.execute(
                "SELECT * FROM seabed_samples WHERE survey_area_id = ?",
                (survey_area_id,)
            )
        else:
            self.cursor.execute("SELECT * FROM seabed_samples")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_geotechnical_measurements(self, sample_id: Optional[int] = None) -> List[Dict]:
        """Get geotechnical measurements, optionally filtered by sample.
        
        Args:
            sample_id: Optional sample ID to filter by
            
        Returns:
            List of measurement dictionaries
        """
        if sample_id:
            self.cursor.execute(
                "SELECT * FROM geotechnical_measurements WHERE sample_id = ?",
                (sample_id,)
            )
        else:
            self.cursor.execute("SELECT * FROM geotechnical_measurements")
        return [dict(row) for row in self.cursor.fetchall()]
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Demo usage of the Bohai SDB."""
    print("Bohai Sea Seabed Database - Demo")
    print("=" * 50)
    
    # Initialize database
    with BohaiSDB("demo_bohai.db") as db:
        # Add a survey area
        area_id = db.add_survey_area(
            name="Bohai Bay Area A",
            latitude=38.5,
            longitude=118.2,
            water_depth=25.5,
            survey_date="2024-03-15",
            description="Northern section of Bohai Bay"
        )
        print(f"\nAdded survey area with ID: {area_id}")
        
        # Add a seabed sample
        sample_id = db.add_seabed_sample(
            survey_area_id=area_id,
            sample_id="BH-2024-001",
            depth=5.2,
            soil_type="Clay",
            moisture_content=28.5,
            density=1.85,
            porosity=42.3,
            collection_date="2024-03-15",
            notes="High plasticity clay"
        )
        print(f"Added seabed sample with ID: {sample_id}")
        
        # Add geotechnical measurements
        meas_id = db.add_geotechnical_measurement(
            sample_id=sample_id,
            measurement_type="shear_strength",
            value=45.2,
            unit="kPa",
            measurement_date="2024-03-16"
        )
        print(f"Added measurement with ID: {meas_id}")
        
        # Query data
        print("\n" + "=" * 50)
        print("Survey Areas:")
        for area in db.get_survey_areas():
            print(f"  - {area['name']} ({area['latitude']}, {area['longitude']})")
        
        print("\nSeabed Samples:")
        for sample in db.get_seabed_samples():
            print(f"  - {sample['sample_id']}: {sample['soil_type']} at {sample['depth']}m")
        
        print("\nGeotechnical Measurements:")
        for meas in db.get_geotechnical_measurements():
            print(f"  - {meas['measurement_type']}: {meas['value']} {meas['unit']}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
