import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Load the historical F1 data"""
    print("Loading historical data...")
    df = pd.read_csv('f1_historical_data.csv')
    return df

def calculate_time_weighted_stats(df):
    """Calculate time-weighted statistics for drivers and constructors"""
    print("Calculating time-weighted statistics...")
    
    # Sort data by date to ensure correct time weighting
    df = df.sort_values('Date')
    
    # Calculate time weights (exponential decay)
    decay_factor = 0.9
    max_races = df.groupby('DriverId').size().max()
    
    # Driver statistics with time weighting
    driver_stats = []
    for driver_id, driver_data in df.groupby('DriverId'):
        # Calculate time weights (newer races have higher weights)
        weights = [decay_factor ** (max_races - i) for i in range(len(driver_data))]
        
        # Calculate weighted statistics
        stats = {
            'DriverId': driver_id,
            'RecentAvgQualiPos': np.average(driver_data['QualifyingPosition'], weights=weights),
            'RecentStdQualiPos': np.average((driver_data['QualifyingPosition'] - driver_data['QualifyingPosition'].mean())**2, weights=weights)**0.5,
            'RecentAvgRacePos': np.average(driver_data['RacePosition'], weights=weights),
            'RecentStdRacePos': np.average((driver_data['RacePosition'] - driver_data['RacePosition'].mean())**2, weights=weights)**0.5,
            'RecentAvgPoints': np.average(driver_data['Points'], weights=weights),
            'RecentStdPoints': np.average((driver_data['Points'] - driver_data['Points'].mean())**2, weights=weights)**0.5
        }
        driver_stats.append(stats)
    
    driver_stats = pd.DataFrame(driver_stats)
    
    # Constructor statistics with time weighting
    constructor_stats = []
    for constructor, constructor_data in df.groupby('Constructor'):
        weights = [decay_factor ** (max_races - i) for i in range(len(constructor_data))]
        
        stats = {
            'Constructor': constructor,
            'RecentConstructorAvgQualiPos': np.average(constructor_data['QualifyingPosition'], weights=weights),
            'RecentConstructorStdQualiPos': np.average((constructor_data['QualifyingPosition'] - constructor_data['QualifyingPosition'].mean())**2, weights=weights)**0.5,
            'RecentConstructorAvgRacePos': np.average(constructor_data['RacePosition'], weights=weights),
            'RecentConstructorStdRacePos': np.average((constructor_data['RacePosition'] - constructor_data['RacePosition'].mean())**2, weights=weights)**0.5,
            'RecentConstructorAvgPoints': np.average(constructor_data['Points'], weights=weights),
            'RecentConstructorStdPoints': np.average((constructor_data['Points'] - constructor_data['Points'].mean())**2, weights=weights)**0.5
        }
        constructor_stats.append(stats)
    
    constructor_stats = pd.DataFrame(constructor_stats)
    
    return driver_stats, constructor_stats

def calculate_team_specific_stats(df):
    """Calculate statistics specific to driver-constructor combinations"""
    print("Calculating team-specific statistics...")
    
    # Group by driver-constructor combination
    team_stats = []
    for (driver_id, constructor), team_data in df.groupby(['DriverId', 'Constructor']):
        # Calculate statistics for this specific combination
        stats = {
            'DriverId': driver_id,
            'Constructor': constructor,
            'TeamAvgQualiPos': team_data['QualifyingPosition'].mean(),
            'TeamStdQualiPos': team_data['QualifyingPosition'].std(),
            'TeamAvgRacePos': team_data['RacePosition'].mean(),
            'TeamStdRacePos': team_data['RacePosition'].std(),
            'TeamAvgPoints': team_data['Points'].mean(),
            'TeamStdPoints': team_data['Points'].std(),
            'TeamRaces': len(team_data)  # Number of races with this combination
        }
        team_stats.append(stats)
    
    return pd.DataFrame(team_stats)

def calculate_track_history(df):
    """Calculate historical performance for each track with time weighting"""
    print("Calculating track history...")
    
    # Sort data by date
    df = df.sort_values('Date')
    
    # Calculate time weights
    decay_factor = 0.9
    max_races = df.groupby(['GrandPrix', 'DriverId']).size().max()
    
    track_stats = []
    for (track, driver_id), track_data in df.groupby(['GrandPrix', 'DriverId']):
        weights = [decay_factor ** (max_races - i) for i in range(len(track_data))]
        
        stats = {
            'GrandPrix': track,
            'DriverId': driver_id,
            'RecentTrackAvgQualiPos': np.average(track_data['QualifyingPosition'], weights=weights),
            'RecentTrackStdQualiPos': np.average((track_data['QualifyingPosition'] - track_data['QualifyingPosition'].mean())**2, weights=weights)**0.5,
            'RecentTrackAvgRacePos': np.average(track_data['RacePosition'], weights=weights),
            'RecentTrackStdRacePos': np.average((track_data['RacePosition'] - track_data['RacePosition'].mean())**2, weights=weights)**0.5,
            'RecentTrackAvgPoints': np.average(track_data['Points'], weights=weights),
            'RecentTrackStdPoints': np.average((track_data['Points'] - track_data['Points'].mean())**2, weights=weights)**0.5
        }
        track_stats.append(stats)
    
    return pd.DataFrame(track_stats)

def prepare_training_data():
    """Prepare the data for model training with time-weighted features"""
    # Load the data
    df = load_data()
    
    # Calculate various statistics with time weighting
    driver_stats, constructor_stats = calculate_time_weighted_stats(df)
    team_stats = calculate_team_specific_stats(df)
    track_stats = calculate_track_history(df)
    
    # Merge all statistics with the original data
    print("Merging all statistics...")
    training_data = df.merge(driver_stats, on='DriverId', how='left')
    training_data = training_data.merge(constructor_stats, on='Constructor', how='left')
    training_data = training_data.merge(team_stats, on=['DriverId', 'Constructor'], how='left')
    training_data = training_data.merge(track_stats, on=['GrandPrix', 'DriverId'], how='left')
    
    # Create target variable (Top 3 finish)
    training_data['Top3Finish'] = (training_data['RacePosition'] <= 3).astype(int)
    
    # Select features for the model
    feature_columns = [
        # Direct race features
        'QualifyingPosition', 'Grid',
        
        # Recent driver statistics
        'RecentAvgQualiPos', 'RecentStdQualiPos',
        'RecentAvgRacePos', 'RecentStdRacePos',
        'RecentAvgPoints', 'RecentStdPoints',
        
        # Recent constructor statistics
        'RecentConstructorAvgQualiPos', 'RecentConstructorStdQualiPos',
        'RecentConstructorAvgRacePos', 'RecentConstructorStdRacePos',
        'RecentConstructorAvgPoints', 'RecentConstructorStdPoints',
        
        # Team-specific statistics
        'TeamAvgQualiPos', 'TeamStdQualiPos',
        'TeamAvgRacePos', 'TeamStdRacePos',
        'TeamAvgPoints', 'TeamStdPoints',
        'TeamRaces',
        
        # Recent track statistics
        'RecentTrackAvgQualiPos', 'RecentTrackStdQualiPos',
        'RecentTrackAvgRacePos', 'RecentTrackStdRacePos',
        'RecentTrackAvgPoints', 'RecentTrackStdPoints'
    ]
    
    # Create final training dataset
    X = training_data[feature_columns]
    y = training_data['Top3Finish']
    
    # Save the prepared data
    print("Saving prepared data...")
    X.to_csv('X_train.csv', index=False)
    y.to_csv('y_train.csv', index=False)
    
    # Save feature names for later use
    pd.DataFrame({'feature': feature_columns}).to_csv('feature_names.csv', index=False)
    
    print("\nData preparation complete!")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Top 3 finishes: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Display sample of the prepared data
    print("\nSample of prepared data:")
    print(X.head())
    
    return X, y

if __name__ == "__main__":
    X, y = prepare_training_data() 