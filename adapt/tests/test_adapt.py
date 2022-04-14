"""
Unit and regression test for the adapt package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import adapt


def test_adapt_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "adapt" in sys.modules
