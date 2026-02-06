"""
Basic example of using the Bohai Sea Database.
"""

from bohai_sdb import Database, MarineStation, WaterQuality, SeabedSample
from datetime import date


def main():
    # Create and initialize database
    print("Initializing Bohai Sea Database...")
    with Database("example_bohai.db") as db:
        db.create_tables()
        print("✓ Database tables created")
        
        # Create a marine station
        print("\nCreating marine monitoring station...")
        station = MarineStation(db)
        station.name = "Bohai Bay Station 1"
        station.latitude = 38.8744
        station.longitude = 117.8731
        station.depth = 15.5
        station.description = "Primary monitoring station in Bohai Bay"
        station.save()
        print(f"✓ Station created with ID: {station.id}")
        
        # Add water quality measurement
        print("\nAdding water quality measurement...")
        measurement = WaterQuality(db)
        measurement.station_id = station.id
        measurement.measurement_date = date.today().isoformat()
        measurement.temperature = 12.5
        measurement.salinity = 30.2
        measurement.ph = 8.1
        measurement.dissolved_oxygen = 7.8
        measurement.turbidity = 2.5
        measurement.save()
        print(f"✓ Water quality measurement added with ID: {measurement.id}")
        
        # Add seabed sample
        print("\nAdding seabed sample...")
        sample = SeabedSample(db)
        sample.station_id = station.id
        sample.sample_date = date.today().isoformat()
        sample.depth = 14.0
        sample.sediment_type = "Sandy silt"
        sample.organic_content = 3.2
        sample.grain_size = 0.15
        sample.notes = "Sample collected near station anchor point"
        sample.save()
        print(f"✓ Seabed sample added with ID: {sample.id}")
        
        # Retrieve all stations
        print("\nRetrieving all stations...")
        stations = MarineStation.get_all(db)
        for s in stations:
            print(f"  - {s.name} ({s.latitude}, {s.longitude})")
            
        # Retrieve measurements for station
        print(f"\nRetrieving measurements for station {station.id}...")
        measurements = WaterQuality.get_by_station(db, station.id)
        for m in measurements:
            print(f"  - Date: {m.measurement_date}, Temp: {m.temperature}°C, pH: {m.ph}")
            
        # Retrieve samples for station
        print(f"\nRetrieving samples for station {station.id}...")
        samples = SeabedSample.get_by_station(db, station.id)
        for sp in samples:
            print(f"  - Date: {sp.sample_date}, Depth: {sp.depth}m, Type: {sp.sediment_type}")
    
    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()
