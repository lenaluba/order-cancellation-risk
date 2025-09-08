"""Tests for data_ingestion.py module."""

import hashlib
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from data_ingestion import sha256sum, download_data, DATA_URL, EXPECTED_SHA256
except ImportError:
    pytest.skip("data_ingestion module not available", allow_module_level=True)


class TestSha256sum:
    """Test cases for sha256sum function."""
    
    def test_sha256sum_with_known_content(self):
        """Test SHA-256 calculation with known content."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)
        
        try:
            result = sha256sum(temp_path)
            # Calculate expected hash dynamically
            expected = hashlib.sha256("test content".encode()).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()
    
    def test_sha256sum_empty_file(self):
        """Test SHA-256 calculation for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = sha256sum(temp_path)
            # Expected SHA-256 for empty file
            expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            assert result == expected
        finally:
            temp_path.unlink()


class TestDownloadData:
    """Test cases for download_data function."""
    
    def test_download_data_file_already_exists(self, tmp_path):
        """Test when target file already exists."""
        # Create mock target file
        target_file = tmp_path / "online_retail_II.xlsx"
        target_file.write_text("existing file")
        
        # Import here to avoid module-level import issues
        import data_ingestion
        
        with patch.object(data_ingestion, 'TARGET_PATH', target_file):
            with patch('builtins.print') as mock_print:
                result = download_data()
                
                assert result == target_file
                mock_print.assert_called_with("Dataset already downloaded. Skipping download.")
    
    def test_download_data_successful_with_zip(self, tmp_path):
        """Test successful extraction from existing zip."""
        import data_ingestion
        
        target_file = tmp_path / "online_retail_II.xlsx"
        zip_file = tmp_path / "online_retail_ii.zip"
        
        # Create a valid zip file with Excel file
        excel_content = b"fake excel content"
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.writestr("online_retail_II.xlsx", excel_content)
        
        with patch.object(data_ingestion, 'TARGET_PATH', target_file):
            with patch.object(data_ingestion, 'ZIP_PATH', zip_file):
                with patch.object(data_ingestion, 'sha256sum', return_value=EXPECTED_SHA256):
                    with patch('builtins.print') as mock_print:
                        result = download_data()
                        
                        assert result == target_file
                        assert target_file.exists()
                        assert target_file.read_bytes() == excel_content
                        mock_print.assert_any_call("Zip file already present. Skipping download.")
    
    def test_download_data_hash_mismatch(self, tmp_path):
        """Test hash mismatch raises ValueError."""
        import data_ingestion
        
        zip_file = tmp_path / "online_retail_ii.zip"
        target_file = tmp_path / "online_retail_II.xlsx"
        
        # Create zip file
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.writestr("online_retail_II.xlsx", b"content")
        
        with patch.object(data_ingestion, 'TARGET_PATH', target_file):
            with patch.object(data_ingestion, 'ZIP_PATH', zip_file):
                with patch.object(data_ingestion, 'sha256sum', return_value="wrong_hash"):
                    with pytest.raises(ValueError, match="SHA-256 mismatch"):
                        download_data()
    
    def test_download_data_no_excel_in_zip(self, tmp_path):
        """Test ValueError when no Excel file found in zip."""
        import data_ingestion
        
        zip_file = tmp_path / "online_retail_ii.zip"
        target_file = tmp_path / "online_retail_II.xlsx"
        
        # Create zip without Excel file
        with zipfile.ZipFile(zip_file, 'w') as zf:
            zf.writestr("other_file.txt", b"content")
        
        with patch.object(data_ingestion, 'TARGET_PATH', target_file):
            with patch.object(data_ingestion, 'ZIP_PATH', zip_file):
                with patch.object(data_ingestion, 'sha256sum', return_value=EXPECTED_SHA256):
                    with pytest.raises(ValueError, match="No Excel file found in the zip"):
                        download_data()


class TestConstants:
    """Test module constants."""
    
    def test_data_url_format(self):
        """Test that DATA_URL is properly formatted."""
        assert DATA_URL.startswith("https://")
        assert "archive.ics.uci.edu" in DATA_URL
        assert DATA_URL.endswith(".zip")
    
    def test_expected_sha256_format(self):
        """Test that EXPECTED_SHA256 is properly formatted."""
        # Should be 64 character hex string or empty
        assert len(EXPECTED_SHA256) == 64 or EXPECTED_SHA256 == ""
        if EXPECTED_SHA256:
            # Should be valid hex
            int(EXPECTED_SHA256, 16)