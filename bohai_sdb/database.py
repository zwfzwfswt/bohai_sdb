"""
Database connection and management module for Bohai Sea Database.
"""

import sqlite3
from pathlib import Path
from typing import Optional


class Database:
    """Main database class for managing connections and operations."""
    
    def __init__(self, db_path: str = "bohai_sdb.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Marine stations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS marine_stations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                depth REAL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Water quality measurements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS water_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id INTEGER NOT NULL,
                measurement_date DATE NOT NULL,
                temperature REAL,
                salinity REAL,
                ph REAL,
                dissolved_oxygen REAL,
                turbidity REAL,
                FOREIGN KEY (station_id) REFERENCES marine_stations(id)
            )
        """)
        
        # Seabed samples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seabed_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id INTEGER NOT NULL,
                sample_date DATE NOT NULL,
                depth REAL NOT NULL,
                sediment_type TEXT,
                organic_content REAL,
                grain_size REAL,
                notes TEXT,
                FOREIGN KEY (station_id) REFERENCES marine_stations(id)
            )
        """)
        
        self.conn.commit()
        
    def execute(self, query: str, params: tuple = ()):
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Cursor object
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        self.conn.commit()
        return cursor
        
    def fetch_all(self, query: str, params: tuple = ()):
        """
        Fetch all results from a query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of Row objects
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
        
    def fetch_one(self, query: str, params: tuple = ()):
        """
        Fetch one result from a query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Row object or None
        """
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchone()
