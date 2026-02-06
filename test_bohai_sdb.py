#!/usr/bin/env python3
"""
Tests for Bohai Sea Seabed Database (bohai_sdb)
"""

import os
import unittest
import tempfile
from bohai_sdb import BohaiSDB


class TestBohaiSDB(unittest.TestCase):
    """Test cases for BohaiSDB class."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = BohaiSDB(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_add_survey_area(self):
        """Test adding a survey area."""
        area_id = self.db.add_survey_area(
            name="Test Area",
            latitude=38.5,
            longitude=118.2,
            water_depth=25.0
        )
        self.assertIsNotNone(area_id)
        self.assertGreater(area_id, 0)
        
        areas = self.db.get_survey_areas()
        self.assertEqual(len(areas), 1)
        self.assertEqual(areas[0]['name'], "Test Area")
        self.assertEqual(areas[0]['latitude'], 38.5)
        self.assertEqual(areas[0]['longitude'], 118.2)
    
    def test_add_seabed_sample(self):
        """Test adding a seabed sample."""
        area_id = self.db.add_survey_area(
            name="Test Area",
            latitude=38.5,
            longitude=118.2
        )
        
        sample_id = self.db.add_seabed_sample(
            survey_area_id=area_id,
            sample_id="TEST-001",
            depth=5.0,
            soil_type="Clay",
            moisture_content=30.0
        )
        self.assertIsNotNone(sample_id)
        self.assertGreater(sample_id, 0)
        
        samples = self.db.get_seabed_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]['sample_id'], "TEST-001")
        self.assertEqual(samples[0]['soil_type'], "Clay")
    
    def test_add_geotechnical_measurement(self):
        """Test adding a geotechnical measurement."""
        area_id = self.db.add_survey_area(
            name="Test Area",
            latitude=38.5,
            longitude=118.2
        )
        
        sample_id = self.db.add_seabed_sample(
            survey_area_id=area_id,
            sample_id="TEST-001",
            depth=5.0
        )
        
        meas_id = self.db.add_geotechnical_measurement(
            sample_id=sample_id,
            measurement_type="shear_strength",
            value=45.2,
            unit="kPa"
        )
        self.assertIsNotNone(meas_id)
        self.assertGreater(meas_id, 0)
        
        measurements = self.db.get_geotechnical_measurements()
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]['measurement_type'], "shear_strength")
        self.assertEqual(measurements[0]['value'], 45.2)
    
    def test_get_samples_by_area(self):
        """Test filtering samples by survey area."""
        area1_id = self.db.add_survey_area("Area 1", 38.5, 118.2)
        area2_id = self.db.add_survey_area("Area 2", 39.0, 119.0)
        
        self.db.add_seabed_sample(area1_id, "S1-001", 5.0)
        self.db.add_seabed_sample(area1_id, "S1-002", 10.0)
        self.db.add_seabed_sample(area2_id, "S2-001", 7.5)
        
        area1_samples = self.db.get_seabed_samples(area1_id)
        self.assertEqual(len(area1_samples), 2)
        
        area2_samples = self.db.get_seabed_samples(area2_id)
        self.assertEqual(len(area2_samples), 1)
    
    def test_get_measurements_by_sample(self):
        """Test filtering measurements by sample."""
        area_id = self.db.add_survey_area("Test Area", 38.5, 118.2)
        sample1_id = self.db.add_seabed_sample(area_id, "S-001", 5.0)
        sample2_id = self.db.add_seabed_sample(area_id, "S-002", 10.0)
        
        self.db.add_geotechnical_measurement(sample1_id, "shear_strength", 45.2, "kPa")
        self.db.add_geotechnical_measurement(sample1_id, "compression", 0.85, "MPa")
        self.db.add_geotechnical_measurement(sample2_id, "shear_strength", 52.0, "kPa")
        
        sample1_measurements = self.db.get_geotechnical_measurements(sample1_id)
        self.assertEqual(len(sample1_measurements), 2)
        
        sample2_measurements = self.db.get_geotechnical_measurements(sample2_id)
        self.assertEqual(len(sample2_measurements), 1)
    
    def test_context_manager(self):
        """Test using BohaiSDB as a context manager."""
        with BohaiSDB(self.temp_db.name) as db:
            area_id = db.add_survey_area("Test Area", 38.5, 118.2)
            self.assertIsNotNone(area_id)


if __name__ == '__main__':
    unittest.main()
