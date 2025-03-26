import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import pandas as pd

# Enable cache
print("Enabling cache...")
fastf1.Cache.enable_cache('cache')

# Load the session data
print("Loading session data...")
session = fastf1.get_session(2023, 'Monza', 'Q')
session.load()

# Get the qualifying results
print("\nQualifying results:")
result = session.results

# Convert to pandas DataFrame and select relevant columns
df = pd.DataFrame(result)[['DriverNumber', 'BroadcastName', 'Q1', 'Q2', 'Q3']]

# Convert time deltas to strings where they exist
for col in ['Q1', 'Q2', 'Q3']:
    df[col] = df[col].astype(str)

print(df.to_string(index=False))

# Get fastest lap of the session
fastest_lap = session.laps.pick_fastest()
print(f"\nFastest lap was set by {fastest_lap['Driver']} with a time of {fastest_lap['LapTime']}")

print("\nData loading successful!") 