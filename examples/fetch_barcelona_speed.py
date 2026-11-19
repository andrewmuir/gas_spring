"""Refresh the Barcelona speed and estimated acceleration example.

Run: python examples/fetch_barcelona_speed.py
All eleven circuits: python examples/fetch_f1_lap_profiles.py
"""

from fetch_f1_lap_profiles import main


if __name__ == "__main__":
    main(["--tracks", "barcelona"])
