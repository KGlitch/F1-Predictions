import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Load the historical F1 data"""
    print("Loading historical data...")
    df = pd.read_csv('f1_historical_data.csv')
    return df

def calculate_driver_stats(df):
    """Calculate rolling statistics for each driver"""
    print("Calculating driver statistics...")
    
    # Sort by date to ensure correct rolling calculations
    df = df.sort_values(['DriverId', 'Date'])
    
    # Calculate rolling averages for each driver
    driver_stats = df.groupby('DriverId').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    driver_stats.columns = ['DriverId', 
                          'AvgQualiPos', 'StdQualiPos',
                          'AvgRacePos', 'StdRacePos',
                          'AvgPoints', 'StdPoints']
    
    return driver_stats

def calculate_constructor_stats(df):
    """Calculate rolling statistics for each constructor"""
    print("Calculating constructor statistics...")
    
    # Sort by date
    df = df.sort_values(['Constructor', 'Date'])
    
    # Calculate rolling averages for each constructor
    constructor_stats = df.groupby('Constructor').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    constructor_stats.columns = ['Constructor',
                               'ConstructorAvgQualiPos', 'ConstructorStdQualiPos',
                               'ConstructorAvgRacePos', 'ConstructorStdRacePos',
                               'ConstructorAvgPoints', 'ConstructorStdPoints']
    
    return constructor_stats

def calculate_track_history(df):
    """Calculate historical performance for each track"""
    print("Calculating track history...")
    
    track_stats = df.groupby(['GrandPrix', 'DriverId']).agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    track_stats.columns = ['GrandPrix', 'DriverId',
                          'TrackAvgQualiPos', 'TrackStdQualiPos',
                          'TrackAvgRacePos', 'TrackStdRacePos',
                          'TrackAvgPoints', 'TrackStdPoints']
    
    return track_stats

def prepare_training_data():
    """Prepare the data for model training"""
    # Load the data
    df = load_data()
    
    # Calculate various statistics
    driver_stats = calculate_driver_stats(df)
    constructor_stats = calculate_constructor_stats(df)
    track_stats = calculate_track_history(df)
    
    # Merge all statistics with the original data
    print("Merging all statistics...")
    training_data = df.merge(driver_stats, on='DriverId', how='left')
    training_data = training_data.merge(constructor_stats, on='Constructor', how='left')
    training_data = training_data.merge(track_stats, on=['GrandPrix', 'DriverId'], how='left')
    
    # Create target variable (Top 3 finish)
    training_data['Top3Finish'] = (training_data['RacePosition'] <= 3).astype(int)
    
    # Select features for the model
    feature_columns = [
        'QualifyingPosition', 'Grid',
        'AvgQualiPos', 'StdQualiPos',
        'AvgRacePos', 'StdRacePos',
        'AvgPoints', 'StdPoints',
        'ConstructorAvgQualiPos', 'ConstructorStdQualiPos',
        'ConstructorAvgRacePos', 'ConstructorStdRacePos',
        'ConstructorAvgPoints', 'ConstructorStdPoints',
        'TrackAvgQualiPos', 'TrackStdQualiPos',
        'TrackAvgRacePos', 'TrackStdRacePos',
        'TrackAvgPoints', 'TrackStdPoints'
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