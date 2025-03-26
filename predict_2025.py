import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def load_models():
    """Load the trained model and preprocessors"""
    print("Loading model and preprocessors...")
    model = joblib.load('f1_model.joblib')
    scaler = joblib.load('f1_scaler.joblib')
    imputer = joblib.load('f1_imputer.joblib')
    feature_names = pd.read_csv('feature_names.csv')['feature'].tolist()
    
    print("\nRequired features:")
    print(feature_names)
    
    return model, scaler, imputer, feature_names

def calculate_driver_stats(df):
    """Calculate statistics for each driver"""
    # Use last 2 seasons for driver stats
    recent_df = df[df['Year'] >= 2022].copy()
    
    driver_stats = recent_df.groupby('DriverId').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    
    # Flatten column names
    driver_stats.columns = ['AvgQualiPos', 'StdQualiPos',
                          'AvgRacePos', 'StdRacePos',
                          'AvgPoints', 'StdPoints']
    
    # Fill missing values with overall means
    for col in driver_stats.columns:
        driver_stats[col] = driver_stats[col].fillna(driver_stats[col].mean())
    
    return driver_stats

def calculate_constructor_stats(df):
    """Calculate statistics for each constructor"""
    # Use last 2 seasons for constructor stats
    recent_df = df[df['Year'] >= 2022].copy()
    
    constructor_stats = recent_df.groupby('Constructor').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    
    # Flatten column names
    constructor_stats.columns = ['ConstructorAvgQualiPos', 'ConstructorStdQualiPos',
                               'ConstructorAvgRacePos', 'ConstructorStdRacePos',
                               'ConstructorAvgPoints', 'ConstructorStdPoints']
    
    # Fill missing values with overall means
    for col in constructor_stats.columns:
        constructor_stats[col] = constructor_stats[col].fillna(constructor_stats[col].mean())
    
    return constructor_stats

def calculate_track_stats(df):
    """Calculate statistics for each track"""
    # Use all historical data for track stats
    track_stats = df.groupby('GrandPrix').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    
    # Flatten column names
    track_stats.columns = ['TrackAvgQualiPos', 'TrackStdQualiPos',
                          'TrackAvgRacePos', 'TrackStdRacePos',
                          'TrackAvgPoints', 'TrackStdPoints']
    
    # Fill missing values with overall means
    for col in track_stats.columns:
        track_stats[col] = track_stats[col].fillna(track_stats[col].mean())
    
    return track_stats

def load_and_prepare_stats():
    """Load the historical data and calculate all statistics"""
    print("Loading and preparing statistics...")
    df = pd.read_csv('f1_historical_data.csv')
    
    print("\nAvailable columns in historical data:")
    print(df.columns.tolist())
    
    # Calculate statistics
    driver_stats = calculate_driver_stats(df)
    constructor_stats = calculate_constructor_stats(df)
    track_stats = calculate_track_stats(df)
    
    # Get the most recent qualifying and grid positions
    latest_positions = df.sort_values('Date').groupby('DriverId').last()[['QualifyingPosition', 'Grid']]
    
    return driver_stats, constructor_stats, track_stats, latest_positions

def create_2025_calendar():
    """Create the 2025 F1 calendar"""
    return [
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Australian Grand Prix",
        "Japanese Grand Prix",
        "Chinese Grand Prix",
        "Miami Grand Prix",
        "Emilia Romagna Grand Prix",
        "Monaco Grand Prix",
        "Canadian Grand Prix",
        "Spanish Grand Prix",
        "Austrian Grand Prix",
        "British Grand Prix",
        "Hungarian Grand Prix",
        "Belgian Grand Prix",
        "Dutch Grand Prix",
        "Italian Grand Prix",
        "Azerbaijan Grand Prix",
        "Singapore Grand Prix",
        "United States Grand Prix",
        "Mexico City Grand Prix",
        "São Paulo Grand Prix",
        "Las Vegas Grand Prix",
        "Qatar Grand Prix",
        "Abu Dhabi Grand Prix"
    ]

def prepare_prediction_data(driver_stats, constructor_stats, track_stats, latest_positions, feature_names):
    """Prepare data for 2025 predictions"""
    print("Preparing prediction data...")
    
    # Current driver-constructor pairs for 2025
    driver_teams_2025 = {
        'NOR': 'McLaren',
        'VER': 'Red Bull Racing',
        'RUS': 'Mercedes',
        'PIA': 'McLaren',
        'ANT': 'Mercedes',
        'ALB': 'Williams',
        'OCO': 'Haas',
        'STR': 'Aston Martin',
        'HAM': 'Ferrari',
        'LEC': 'Ferrari',
        'HUL': 'Sauber',
        'BEA': 'Haas',
        'TSU': 'VCARB',
        'SAI': 'Williams',
        'HAD': 'VCARB',
        'GAS': 'Alpine',
        'LAW': 'Red Bull Racing',
        'DOO': 'Alpine',
        'BOR': 'Sauber',
        'ALO': 'Aston Martin'
    }
    
    races = create_2025_calendar()
    prediction_data = []
    
    for driver_id, constructor in driver_teams_2025.items():
        # Get driver statistics
        driver_data = driver_stats.loc[driver_id] if driver_id in driver_stats.index else pd.Series(np.nan, index=driver_stats.columns)
        
        # Get constructor statistics
        constructor_data = constructor_stats.loc[constructor] if constructor in constructor_stats.index else pd.Series(np.nan, index=constructor_stats.columns)
        
        # Get latest positions
        latest_pos = latest_positions.loc[driver_id] if driver_id in latest_positions.index else pd.Series({'QualifyingPosition': np.nan, 'Grid': np.nan})
        
        for race in races:
            # Get track statistics
            track_data = track_stats.loc[race] if race in track_stats.index else pd.Series(np.nan, index=track_stats.columns)
            
            race_data = {
                'Driver': driver_id,
                'Constructor': constructor,
                'GrandPrix': race,
                'QualifyingPosition': latest_pos['QualifyingPosition'],
                'Grid': latest_pos['Grid']
            }
            
            # Add driver statistics
            for col in driver_stats.columns:
                race_data[col] = driver_data[col]
            
            # Add constructor statistics
            for col in constructor_stats.columns:
                race_data[col] = constructor_data[col]
            
            # Add track statistics
            for col in track_stats.columns:
                race_data[col] = track_data[col]
            
            prediction_data.append(race_data)
    
    df = pd.DataFrame(prediction_data)
    
    print("\nColumns in prediction data:")
    print(df.columns.tolist())
    
    print("\nFeature names required:")
    print(feature_names)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            print(f"Missing feature: {feature}")
            df[feature] = np.nan
    
    return df[['Driver', 'Constructor', 'GrandPrix'] + feature_names]

def predict_top3(model, scaler, imputer, X_pred, drivers):
    """Make predictions and return top 3 for each race"""
    # Prepare features
    X_pred_imputed = imputer.transform(X_pred)
    X_pred_scaled = scaler.transform(X_pred_imputed)
    
    # Get probabilities for top 3 finish
    probas = model.predict_proba(X_pred_scaled)[:, 1]
    
    # Add probabilities to the drivers DataFrame
    drivers['Probability'] = probas
    
    # Get top 3 for each race
    results = []
    for race in drivers['GrandPrix'].unique():
        race_results = drivers[drivers['GrandPrix'] == race].sort_values('Probability', ascending=False).head(3)
        results.append({
            'Race': race,
            'Predicted_Top3': [
                f"{pos+1}. {driver} ({constructor}) - {prob:.1%} probability"
                for pos, (_, driver, constructor, prob) in enumerate(
                    race_results[['Driver', 'Constructor', 'Probability']].itertuples()
                )
            ]
        })
    
    return results

def main():
    # Load models and data
    model, scaler, imputer, feature_names = load_models()
    driver_stats, constructor_stats, track_stats, latest_positions = load_and_prepare_stats()
    
    # Prepare prediction data
    prediction_df = prepare_prediction_data(
        driver_stats, 
        constructor_stats, 
        track_stats, 
        latest_positions,
        feature_names
    )
    
    # Make predictions
    print("\nMaking predictions for 2025 season...")
    results = predict_top3(
        model, 
        scaler, 
        imputer, 
        prediction_df[feature_names],
        prediction_df[['Driver', 'Constructor', 'GrandPrix']]
    )
    
    # Print predictions
    print("\nPredicted Top 3 for each race in 2025:")
    print("=======================================")
    for race_result in results:
        print(f"\n{race_result['Race']}:")
        for position in race_result['Predicted_Top3']:
            print(position)

if __name__ == "__main__":
    main() 