import pandas as pd
import requests
from datetime import datetime
import time

def get_ergast_data(endpoint):
    """Helper function to get data from Ergast API with rate limiting"""
    base_url = 'http://ergast.com/api/f1'
    url = f"{base_url}/{endpoint}.json"
    
    # Rate limiting - maximum 4 requests per second
    time.sleep(0.25)
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from {url}: {response.status_code}")
        return None

def get_race_results(year, round_num):
    """Get race results for a specific race"""
    data = get_ergast_data(f"{year}/{round_num}/results")
    if data:
        results = data['MRData']['RaceTable']['Races']
        if results:
            race = results[0]
            return pd.DataFrame([{
                'Year': year,
                'Round': round_num,
                'GrandPrix': race['raceName'],
                'Date': race['date'],
                'DriverId': result['Driver']['driverId'],
                'DriverNumber': result['number'],
                'DriverName': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                'Constructor': result['Constructor']['name'],
                'Grid': int(result['grid']),
                'RacePosition': int(result['position']),
                'Points': float(result['points']),
                'Status': result['status']
            } for result in race['Results']])
    return None

def get_qualifying_results(year, round_num):
    """Get qualifying results for a specific race"""
    data = get_ergast_data(f"{year}/{round_num}/qualifying")
    if data:
        results = data['MRData']['RaceTable']['Races']
        if results:
            quali = results[0]
            return pd.DataFrame([{
                'Year': year,
                'Round': round_num,
                'DriverId': result['Driver']['driverId'],
                'QualifyingPosition': int(result['position']),
                'Q1': result.get('Q1', None),
                'Q2': result.get('Q2', None),
                'Q3': result.get('Q3', None)
            } for result in quali['QualifyingResults']])
    return None

# Years to collect data for
years = [2021, 2022, 2023, 2024]
all_data = []

print("Starting data collection...")

for year in years:
    print(f"\nProcessing year {year}...")
    
    # Get the number of races in the season
    season_data = get_ergast_data(str(year))
    if not season_data:
        continue
        
    total_races = int(season_data['MRData']['total'])
    print(f"Found {total_races} races")
    
    for round_num in range(1, total_races + 1):
        print(f"Processing round {round_num}/{total_races}")
        
        # Get race and qualifying results
        race_df = get_race_results(year, round_num)
        quali_df = get_qualifying_results(year, round_num)
        
        if race_df is not None and quali_df is not None:
            # Merge race and qualifying data
            merged_df = pd.merge(
                race_df,
                quali_df[['DriverId', 'QualifyingPosition', 'Q1', 'Q2', 'Q3']],
                on='DriverId',
                how='outer'
            )
            all_data.append(merged_df)

# Combine all data
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Convert qualifying times to seconds where possible
    for col in ['Q1', 'Q2', 'Q3']:
        # Convert string times (e.g., "1:23.456") to seconds
        def convert_to_seconds(time_str):
            if pd.isna(time_str):
                return None
            try:
                if ':' in time_str:
                    minutes, seconds = time_str.split(':')
                    return float(minutes) * 60 + float(seconds)
                return float(time_str)
            except:
                return None
        
        final_df[col] = final_df[col].apply(convert_to_seconds)
    
    # Save to CSV
    output_file = 'f1_historical_data.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    print(f"Total records collected: {len(final_df)}")
    
    # Display summary statistics
    print("\nData summary:")
    print(f"Years covered: {sorted(final_df['Year'].unique())}")
    print(f"Number of unique drivers: {final_df['DriverName'].nunique()}")
    print(f"Number of races: {len(final_df[['Year', 'GrandPrix']].drop_duplicates())}")
    
    # Display sample of the data
    print("\nSample of collected data:")
    print(final_df.head())
else:
    print("No data was collected successfully") 