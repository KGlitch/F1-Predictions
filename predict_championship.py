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
    return model, scaler, imputer, feature_names

def calculate_points(position):
    """Calculate points for a given position"""
    points_system = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1
    }
    return points_system.get(position, 0)

def get_current_2025_results():
    """Get the current 2025 season results"""
    return {
        'Bahrain Grand Prix': [
            {'Position': 1, 'Driver': 'NOR', 'Constructor': 'McLaren', 'Points': 25},
            {'Position': 2, 'Driver': 'VER', 'Constructor': 'Red Bull Racing', 'Points': 18},
            {'Position': 3, 'Driver': 'RUS', 'Constructor': 'Mercedes', 'Points': 15},
            {'Position': 4, 'Driver': 'PIA', 'Constructor': 'McLaren', 'Points': 12},
            {'Position': 5, 'Driver': 'ANT', 'Constructor': 'Mercedes', 'Points': 10},
            {'Position': 6, 'Driver': 'ALB', 'Constructor': 'Williams', 'Points': 8},
            {'Position': 7, 'Driver': 'OCO', 'Constructor': 'Haas', 'Points': 6},
            {'Position': 8, 'Driver': 'STR', 'Constructor': 'Aston Martin', 'Points': 4},
            {'Position': 9, 'Driver': 'HAM', 'Constructor': 'Ferrari', 'Points': 2},
            {'Position': 10, 'Driver': 'LEC', 'Constructor': 'Ferrari', 'Points': 1}
        ],
        'Saudi Arabian Grand Prix': [
            {'Position': 1, 'Driver': 'NOR', 'Constructor': 'McLaren', 'Points': 25},
            {'Position': 2, 'Driver': 'VER', 'Constructor': 'Red Bull Racing', 'Points': 18},
            {'Position': 3, 'Driver': 'RUS', 'Constructor': 'Mercedes', 'Points': 15},
            {'Position': 4, 'Driver': 'PIA', 'Constructor': 'McLaren', 'Points': 12},
            {'Position': 5, 'Driver': 'ANT', 'Constructor': 'Mercedes', 'Points': 10},
            {'Position': 6, 'Driver': 'ALB', 'Constructor': 'Williams', 'Points': 8},
            {'Position': 7, 'Driver': 'OCO', 'Constructor': 'Haas', 'Points': 6},
            {'Position': 8, 'Driver': 'STR', 'Constructor': 'Aston Martin', 'Points': 4},
            {'Position': 9, 'Driver': 'HAM', 'Constructor': 'Ferrari', 'Points': 2},
            {'Position': 10, 'Driver': 'LEC', 'Constructor': 'Ferrari', 'Points': 1}
        ]
    }

def adjust_probabilities(probas, drivers, current_results):
    """Adjust probabilities based on current season results"""
    # Create a mapping of driver performance
    driver_performance = {}
    for race, results in current_results.items():
        for result in results:
            driver = result['Driver']
            if driver not in driver_performance:
                driver_performance[driver] = {'points': 0, 'races': 0}
            driver_performance[driver]['points'] += result['Points']
            driver_performance[driver]['races'] += 1
    
    # Calculate average points per race
    for driver in driver_performance:
        driver_performance[driver]['avg_points'] = driver_performance[driver]['points'] / driver_performance[driver]['races']
    
    # Create a copy of probabilities
    adjusted_probas = probas.copy()
    
    # Adjust probabilities based on current performance
    for i, driver in enumerate(drivers['Driver']):
        if driver in driver_performance:
            # Increase probability for drivers performing well
            avg_points = driver_performance[driver]['avg_points']
            adjusted_probas[i] *= (1 + avg_points / 25)  # Normalize by max points
    
    return adjusted_probas

def get_2025_drivers():
    """Get the list of drivers on the 2025 grid"""
    return {
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

def predict_race_results(model, scaler, imputer, X_pred, drivers, current_results, top_positions=10):
    """Make predictions and return top positions for each race"""
    # Get current 2025 drivers
    current_drivers = get_2025_drivers()
    
    # Filter out drivers not on the 2025 grid
    mask = drivers['Driver'].isin(current_drivers.keys())
    drivers = drivers[mask]
    X_pred = X_pred[mask]
    
    # Prepare features
    X_pred_imputed = imputer.transform(X_pred)
    X_pred_scaled = scaler.transform(X_pred_imputed)
    
    # Get probabilities for top 3 finish
    probas = model.predict_proba(X_pred_scaled)[:, 1]
    
    # Adjust probabilities based on current season results
    probas = adjust_probabilities(probas, drivers, current_results)
    
    # Add probabilities to the drivers DataFrame
    drivers['Probability'] = probas
    
    # Get results for each race
    results = []
    for race in drivers['GrandPrix'].unique():
        # If we have current results for this race, use them
        if race in current_results:
            results.append({
                'Race': race,
                'Results': current_results[race]
            })
        else:
            # Otherwise, use model predictions
            race_results = drivers[drivers['GrandPrix'] == race].sort_values('Probability', ascending=False).head(top_positions)
            results.append({
                'Race': race,
                'Results': [
                    {
                        'Position': pos+1,
                        'Driver': driver,
                        'Constructor': current_drivers[driver],  # Use actual 2025 constructor
                        'Points': calculate_points(pos+1),
                        'Probability': prob
                    }
                    for pos, (_, driver, _, prob) in enumerate(
                        race_results[['Driver', 'Constructor', 'Probability']].itertuples()
                    )
                ]
            })
    
    return results

def calculate_championship_standings(race_results):
    """Calculate WDC and WCC standings from race results"""
    # Initialize standings
    driver_points = {}
    constructor_points = {}
    
    # Calculate points for each race
    for race in race_results:
        for result in race['Results']:
            # Update driver points
            driver = result['Driver']
            driver_points[driver] = driver_points.get(driver, 0) + result['Points']
            
            # Update constructor points
            constructor = result['Constructor']
            constructor_points[constructor] = constructor_points.get(constructor, 0) + result['Points']
    
    # Convert to DataFrames and sort
    wdc_standings = pd.DataFrame([
        {'Driver': driver, 'Points': points}
        for driver, points in driver_points.items()
    ]).sort_values('Points', ascending=False)
    
    wcc_standings = pd.DataFrame([
        {'Constructor': constructor, 'Points': points}
        for constructor, points in constructor_points.items()
    ]).sort_values('Points', ascending=False)
    
    return wdc_standings, wcc_standings

def main():
    # Load models and data
    model, scaler, imputer, feature_names = load_models()
    driver_stats, constructor_stats, track_stats, latest_positions = load_and_prepare_stats()
    
    # Get current 2025 season results
    current_results = get_current_2025_results()
    
    # Prepare prediction data
    prediction_df = prepare_prediction_data(
        driver_stats, 
        constructor_stats, 
        track_stats, 
        latest_positions,
        feature_names
    )
    
    # Make predictions for all races
    print("\nPredicting race results for 2025 season...")
    race_results = predict_race_results(
        model, 
        scaler, 
        imputer, 
        prediction_df[feature_names],
        prediction_df[['Driver', 'Constructor', 'GrandPrix']],
        current_results,
        top_positions=10
    )
    
    # Calculate championship standings
    wdc_standings, wcc_standings = calculate_championship_standings(race_results)
    
    # Print WDC standings
    print("\nPredicted World Drivers' Championship 2025:")
    print("==========================================")
    print(wdc_standings.to_string(index=False))
    
    # Print WCC standings
    print("\nPredicted World Constructors' Championship 2025:")
    print("=============================================")
    print(wcc_standings.to_string(index=False))
    
    # Print detailed race results
    print("\nDetailed Race Results:")
    print("====================")
    for race in race_results:
        print(f"\n{race['Race']}:")
        for result in race['Results']:
            print(f"{result['Position']}. {result['Driver']} ({result['Constructor']}) - {result['Points']} points ({result.get('Probability', 'N/A')} probability)")

if __name__ == "__main__":
    # Import required functions from predict_2025.py
    from predict_2025 import load_and_prepare_stats, prepare_prediction_data
    main() 