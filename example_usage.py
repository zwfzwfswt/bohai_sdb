#!/usr/bin/env python3
"""
Example usage of Bohai Sea Seabed Database with sample data.
This script demonstrates how to populate the database with realistic data.
"""

from bohai_sdb import BohaiSDB


def populate_sample_data():
    """Populate database with sample survey data."""
    
    with BohaiSDB("example_bohai.db") as db:
        print("Creating sample Bohai Sea database...")
        print("=" * 60)
        
        # Add multiple survey areas
        areas = [
            {
                "name": "Bohai Bay North",
                "latitude": 39.0,
                "longitude": 118.5,
                "water_depth": 18.5,
                "survey_date": "2024-01-15",
                "description": "Northern section of Bohai Bay"
            },
            {
                "name": "Liaodong Bay West",
                "latitude": 40.5,
                "longitude": 121.0,
                "water_depth": 32.0,
                "survey_date": "2024-02-20",
                "description": "Western section of Liaodong Bay"
            },
            {
                "name": "Laizhou Bay East",
                "latitude": 37.5,
                "longitude": 119.8,
                "water_depth": 15.2,
                "survey_date": "2024-03-10",
                "description": "Eastern section of Laizhou Bay"
            }
        ]
        
        area_ids = []
        for area in areas:
            area_id = db.add_survey_area(**area)
            area_ids.append(area_id)
            print(f"Added survey area: {area['name']}")
        
        print()
        
        # Add samples for each area
        samples_data = [
            # Bohai Bay North samples
            {
                "survey_area_id": area_ids[0],
                "sample_id": "BBN-2024-001",
                "depth": 3.5,
                "soil_type": "Silty Clay",
                "moisture_content": 32.5,
                "density": 1.78,
                "porosity": 45.2,
                "collection_date": "2024-01-15"
            },
            {
                "survey_area_id": area_ids[0],
                "sample_id": "BBN-2024-002",
                "depth": 8.0,
                "soil_type": "Clay",
                "moisture_content": 28.0,
                "density": 1.85,
                "porosity": 42.0,
                "collection_date": "2024-01-15"
            },
            # Liaodong Bay West samples
            {
                "survey_area_id": area_ids[1],
                "sample_id": "LBW-2024-001",
                "depth": 5.2,
                "soil_type": "Sandy Silt",
                "moisture_content": 25.8,
                "density": 1.92,
                "porosity": 38.5,
                "collection_date": "2024-02-20"
            },
            {
                "survey_area_id": area_ids[1],
                "sample_id": "LBW-2024-002",
                "depth": 12.5,
                "soil_type": "Silt",
                "moisture_content": 30.2,
                "density": 1.82,
                "porosity": 43.8,
                "collection_date": "2024-02-20"
            },
            # Laizhou Bay East samples
            {
                "survey_area_id": area_ids[2],
                "sample_id": "LBE-2024-001",
                "depth": 4.0,
                "soil_type": "Fine Sand",
                "moisture_content": 18.5,
                "density": 2.05,
                "porosity": 32.0,
                "collection_date": "2024-03-10"
            }
        ]
        
        sample_ids = []
        for sample in samples_data:
            sample_id = db.add_seabed_sample(**sample)
            sample_ids.append(sample_id)
            print(f"Added sample: {sample['sample_id']} ({sample['soil_type']})")
        
        print()
        
        # Add geotechnical measurements
        measurements_data = [
            # Measurements for BBN-2024-001
            {"sample_id": sample_ids[0], "measurement_type": "shear_strength", 
             "value": 42.5, "unit": "kPa", "measurement_date": "2024-01-16"},
            {"sample_id": sample_ids[0], "measurement_type": "compression_modulus", 
             "value": 3.2, "unit": "MPa", "measurement_date": "2024-01-16"},
            # Measurements for BBN-2024-002
            {"sample_id": sample_ids[1], "measurement_type": "shear_strength", 
             "value": 48.0, "unit": "kPa", "measurement_date": "2024-01-16"},
            {"sample_id": sample_ids[1], "measurement_type": "permeability", 
             "value": 1.2e-8, "unit": "m/s", "measurement_date": "2024-01-17"},
            # Measurements for LBW-2024-001
            {"sample_id": sample_ids[2], "measurement_type": "shear_strength", 
             "value": 55.3, "unit": "kPa", "measurement_date": "2024-02-21"},
            {"sample_id": sample_ids[2], "measurement_type": "compression_modulus", 
             "value": 4.5, "unit": "MPa", "measurement_date": "2024-02-21"},
            # Measurements for LBW-2024-002
            {"sample_id": sample_ids[3], "measurement_type": "shear_strength", 
             "value": 45.8, "unit": "kPa", "measurement_date": "2024-02-21"},
            # Measurements for LBE-2024-001
            {"sample_id": sample_ids[4], "measurement_type": "shear_strength", 
             "value": 62.0, "unit": "kPa", "measurement_date": "2024-03-11"},
            {"sample_id": sample_ids[4], "measurement_type": "permeability", 
             "value": 5.5e-5, "unit": "m/s", "measurement_date": "2024-03-11"},
        ]
        
        for meas in measurements_data:
            db.add_geotechnical_measurement(**meas)
        
        print(f"Added {len(measurements_data)} geotechnical measurements")
        
        print("\n" + "=" * 60)
        print("DATABASE SUMMARY")
        print("=" * 60)
        
        # Display summary
        all_areas = db.get_survey_areas()
        print(f"\nTotal Survey Areas: {len(all_areas)}")
        for area in all_areas:
            samples = db.get_seabed_samples(area['id'])
            print(f"\n  {area['name']}:")
            print(f"    Location: ({area['latitude']}°N, {area['longitude']}°E)")
            print(f"    Water Depth: {area['water_depth']}m")
            print(f"    Samples: {len(samples)}")
            
            for sample in samples:
                measurements = db.get_geotechnical_measurements(sample['id'])
                print(f"      - {sample['sample_id']}: {sample['soil_type']} "
                      f"at {sample['depth']}m ({len(measurements)} measurements)")
        
        print("\n" + "=" * 60)
        print("Sample data created successfully!")
        print(f"Database file: example_bohai.db")


if __name__ == "__main__":
    populate_sample_data()
