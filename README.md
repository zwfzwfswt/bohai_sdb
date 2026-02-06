# bohai_sdb

Bohai Sea Seabed Database - A simple database system for managing Bohai Sea seabed engineering geology data.

## Overview

`bohai_sdb` is a lightweight Python-based database system designed to store, manage, and query geological and engineering data from the Bohai Sea seabed. It provides a simple API for managing:

- Survey areas (geographical locations and metadata)
- Seabed samples (soil samples with physical properties)
- Geotechnical measurements (engineering test results)

## Features

- SQLite-based storage for easy deployment
- Simple Python API for CRUD operations
- Support for spatial coordinates (latitude/longitude)
- Track multiple survey areas and their samples
- Store geotechnical measurements with units
- Built-in data validation and referential integrity

## Installation

```bash
# Clone the repository
git clone https://github.com/zwfzwfswt/bohai_sdb.git
cd bohai_sdb

# Install dependencies (SQLite3 is included with Python)
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from bohai_sdb import BohaiSDB

# Initialize database
with BohaiSDB("my_database.db") as db:
    # Add a survey area
    area_id = db.add_survey_area(
        name="Bohai Bay Area A",
        latitude=38.5,
        longitude=118.2,
        water_depth=25.5,
        survey_date="2024-03-15"
    )
    
    # Add a seabed sample
    sample_id = db.add_seabed_sample(
        survey_area_id=area_id,
        sample_id="BH-2024-001",
        depth=5.2,
        soil_type="Clay",
        moisture_content=28.5
    )
    
    # Add measurements
    db.add_geotechnical_measurement(
        sample_id=sample_id,
        measurement_type="shear_strength",
        value=45.2,
        unit="kPa"
    )
    
    # Query data
    areas = db.get_survey_areas()
    samples = db.get_seabed_samples(area_id)
    measurements = db.get_geotechnical_measurements(sample_id)
```

### Running the Demo

```bash
python bohai_sdb.py
```

## Database Schema

### Survey Areas
- `id`: Primary key
- `name`: Name of the survey area
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `water_depth`: Water depth in meters
- `survey_date`: Date of survey
- `description`: Additional description
- `created_at`: Timestamp

### Seabed Samples
- `id`: Primary key
- `survey_area_id`: Foreign key to survey_areas
- `sample_id`: Unique sample identifier
- `depth`: Depth of sample in meters
- `soil_type`: Type of soil (e.g., Clay, Sand)
- `moisture_content`: Moisture content percentage
- `density`: Density in g/cm³
- `porosity`: Porosity percentage
- `collection_date`: Date of collection
- `notes`: Additional notes
- `created_at`: Timestamp

### Geotechnical Measurements
- `id`: Primary key
- `sample_id`: Foreign key to seabed_samples
- `measurement_type`: Type of measurement
- `value`: Measurement value
- `unit`: Unit of measurement
- `measurement_date`: Date of measurement
- `created_at`: Timestamp

## Testing

Run the test suite:

```bash
python -m unittest test_bohai_sdb.py
```

Or run with verbose output:

```bash
python -m unittest test_bohai_sdb.py -v
```

## License

This project is open source and available for use in research and engineering applications.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
