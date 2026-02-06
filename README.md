# Bohai Sea Database (bohai_sdb)

A lightweight Python-based database management system for marine environmental data from the Bohai Sea region. This system provides a simple interface for storing, managing, and querying marine monitoring station data, water quality measurements, and seabed samples.

## Features

- **Marine Station Management**: Store and manage information about marine monitoring stations
- **Water Quality Tracking**: Record and query water quality measurements including temperature, salinity, pH, dissolved oxygen, and turbidity
- **Seabed Sample Data**: Manage seabed sample information including sediment type, organic content, and grain size
- **SQLite Backend**: Lightweight, file-based database requiring no external database server
- **Python API**: Simple, object-oriented interface for data management

## Installation

### From Source

```bash
git clone https://github.com/zwfzwfswt/bohai_sdb.git
cd bohai_sdb
pip install -e .
```

### Requirements

- Python 3.7 or higher
- sqlite3 (included in Python standard library)

## Quick Start

```python
from bohai_sdb import Database, MarineStation, WaterQuality, SeabedSample
from datetime import date

# Create and initialize database
with Database("my_bohai_data.db") as db:
    db.create_tables()
    
    # Create a marine station
    station = MarineStation(db)
    station.name = "Bohai Bay Station 1"
    station.latitude = 38.8744
    station.longitude = 117.8731
    station.depth = 15.5
    station.description = "Primary monitoring station in Bohai Bay"
    station.save()
    
    # Add water quality measurement
    measurement = WaterQuality(db)
    measurement.station_id = station.id
    measurement.measurement_date = date.today().isoformat()
    measurement.temperature = 12.5
    measurement.salinity = 30.2
    measurement.ph = 8.1
    measurement.dissolved_oxygen = 7.8
    measurement.turbidity = 2.5
    measurement.save()
    
    # Retrieve all stations
    stations = MarineStation.get_all(db)
    for s in stations:
        print(f"{s.name} at ({s.latitude}, {s.longitude})")
```

## Database Schema

### Marine Stations
- **id**: Unique identifier
- **name**: Station name
- **latitude**: Geographic latitude
- **longitude**: Geographic longitude
- **depth**: Water depth at station (meters)
- **description**: Station description
- **created_at**: Creation timestamp

### Water Quality
- **id**: Unique identifier
- **station_id**: Reference to marine station
- **measurement_date**: Date of measurement
- **temperature**: Water temperature (°C)
- **salinity**: Salinity (PSU)
- **ph**: pH level
- **dissolved_oxygen**: Dissolved oxygen (mg/L)
- **turbidity**: Turbidity (NTU)

### Seabed Samples
- **id**: Unique identifier
- **station_id**: Reference to marine station
- **sample_date**: Date of sample collection
- **depth**: Sample depth (meters)
- **sediment_type**: Type of sediment
- **organic_content**: Organic content percentage
- **grain_size**: Average grain size (mm)
- **notes**: Additional notes

## Usage Examples

See the `examples/` directory for more detailed usage examples:

- `basic_usage.py`: Basic database operations

### Running Examples

```bash
python examples/basic_usage.py
```

## Testing

Run the test suite:

```bash
python -m unittest tests/test_bohai_sdb.py
```

Or with verbose output:

```bash
python -m unittest tests/test_bohai_sdb.py -v
```

## API Reference

### Database Class

```python
Database(db_path: str = "bohai_sdb.db")
```

Main database class for managing connections and operations.

**Methods:**
- `connect()`: Establish database connection
- `close()`: Close database connection
- `create_tables()`: Create database tables if they don't exist
- `execute(query, params)`: Execute a SQL query
- `fetch_all(query, params)`: Fetch all results from a query
- `fetch_one(query, params)`: Fetch one result from a query

### MarineStation Class

```python
MarineStation(db, station_id=None)
```

Represents a marine monitoring station.

**Attributes:**
- `id`, `name`, `latitude`, `longitude`, `depth`, `description`

**Methods:**
- `save()`: Save station to database
- `get_all(db)`: Get all marine stations (static method)

### WaterQuality Class

```python
WaterQuality(db, measurement_id=None)
```

Represents a water quality measurement.

**Attributes:**
- `id`, `station_id`, `measurement_date`, `temperature`, `salinity`, `ph`, `dissolved_oxygen`, `turbidity`

**Methods:**
- `save()`: Save measurement to database
- `get_by_station(db, station_id)`: Get all measurements for a station (static method)

### SeabedSample Class

```python
SeabedSample(db, sample_id=None)
```

Represents a seabed sample.

**Attributes:**
- `id`, `station_id`, `sample_date`, `depth`, `sediment_type`, `organic_content`, `grain_size`, `notes`

**Methods:**
- `save()`: Save sample to database
- `get_by_station(db, station_id)`: Get all samples for a station (static method)

## Project Structure

```
bohai_sdb/
├── bohai_sdb/           # Main package
│   ├── __init__.py      # Package initialization
│   ├── database.py      # Database management
│   └── models.py        # Data models
├── examples/            # Usage examples
│   └── basic_usage.py
├── tests/               # Test suite
│   └── test_bohai_sdb.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## About Bohai Sea

The Bohai Sea, also known as the Bohai Gulf, is the innermost gulf of the Yellow Sea on the coast of Northeastern China. It is approximately 78,000 km² and is one of China's most important fishing grounds and marine resource areas.

## Acknowledgments

This project is designed for marine environmental monitoring and research purposes, supporting the collection and management of data from Bohai Sea monitoring stations.
