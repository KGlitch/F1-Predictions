import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_models():
    """Load the trained model and preprocessors"""
    print("Loading model and preprocessors...")
    model = joblib.load('f1_model.joblib')
    scaler = joblib.load('f1_scaler.joblib')
    imputer = joblib.load('f1_imputer.joblib')
    feature_names = pd.read_csv('feature_names.csv')['feature'].tolist()
    return model, scaler, imputer, feature_names

def calculate_points(position, is_sprint=False):
    """Calculate points for a given position in either a normal race or sprint race"""
    normal_points_system = {
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
    
    sprint_points_system = {
        1: 8,
        2: 7,
        3: 6,
        4: 5,
        5: 4,
        6: 3,
        7: 2,
        8: 1
    }
    
    return (sprint_points_system if is_sprint else normal_points_system).get(position, 0)

def get_current_2025_results():
    """Get the current 2025 season results"""
    return {
        'Australian Grand Prix': [
            {'Position': 1, 'Driver': 'NOR', 'Constructor': 'McLaren', 'Points': 25},
            {'Position': 2, 'Driver': 'VER', 'Constructor': 'Red Bull Racing', 'Points': 18},
            {'Position': 3, 'Driver': 'RUS', 'Constructor': 'Mercedes', 'Points': 15},
            {'Position': 4, 'Driver': 'ANT', 'Constructor': 'Mercedes', 'Points': 12},
            {'Position': 5, 'Driver': 'ALB', 'Constructor': 'Williams', 'Points': 10},
            {'Position': 6, 'Driver': 'STR', 'Constructor': 'Aston Martin', 'Points': 8},
            {'Position': 7, 'Driver': 'HUL', 'Constructor': 'Sauber', 'Points': 6},
            {'Position': 8, 'Driver': 'LEC', 'Constructor': 'Ferrari', 'Points': 4},
            {'Position': 9, 'Driver': 'PIA', 'Constructor': 'McLaren', 'Points': 2},
            {'Position': 10, 'Driver': 'HAM', 'Constructor': 'Ferrari', 'Points': 1}
        ],
        'Chinese Grand Prix Sprint': [
            {'Position': 1, 'Driver': 'HAM', 'Constructor': 'Ferrari', 'Points': 8},
            {'Position': 2, 'Driver': 'PIA', 'Constructor': 'McLaren', 'Points': 7},
            {'Position': 3, 'Driver': 'VER', 'Constructor': 'Red Bull Racing', 'Points': 6},
            {'Position': 4, 'Driver': 'RUS', 'Constructor': 'Mercedes', 'Points': 5},
            {'Position': 5, 'Driver': 'LEC', 'Constructor': 'Ferrari', 'Points': 4},
            {'Position': 6, 'Driver': 'TSU', 'Constructor': 'VCARB', 'Points': 3},
            {'Position': 7, 'Driver': 'ANT', 'Constructor': 'Mercedes', 'Points': 2},
            {'Position': 8, 'Driver': 'NOR', 'Constructor': 'McLaren', 'Points': 1}
        ],
        'Chinese Grand Prix': [
            {'Position': 1, 'Driver': 'PIA', 'Constructor': 'McLaren', 'Points': 25},
            {'Position': 2, 'Driver': 'NOR', 'Constructor': 'McLaren', 'Points': 18},
            {'Position': 3, 'Driver': 'RUS', 'Constructor': 'Mercedes', 'Points': 15},
            {'Position': 4, 'Driver': 'VER', 'Constructor': 'Red Bull Racing', 'Points': 12},
            {'Position': 5, 'Driver': 'LEC', 'Constructor': 'Ferrari', 'Points': 0},
            {'Position': 6, 'Driver': 'HAM', 'Constructor': 'Ferrari', 'Points': 0},
            {'Position': 7, 'Driver': 'OCO', 'Constructor': 'Haas', 'Points': 10},
            {'Position': 8, 'Driver': 'ANT', 'Constructor': 'Mercedes', 'Points': 8},
            {'Position': 9, 'Driver': 'ALB', 'Constructor': 'Williams', 'Points': 6},
            {'Position': 10, 'Driver': 'BEA', 'Constructor': 'Haas', 'Points': 4},
            {'Position': 11, 'Driver': 'GAS', 'Constructor': 'Alpine', 'Points': 0},
            {'Position': 12, 'Driver': 'STR', 'Constructor': 'Aston Martin', 'Points': 2},
            {'Position': 13, 'Driver': 'SAI', 'Constructor': 'Williams', 'Points': 1},
            {'Position': 14, 'Driver': 'HAD', 'Constructor': 'VCARB', 'Points': 0},
            {'Position': 15, 'Driver': 'DOO', 'Constructor': 'Alpine', 'Points': 0},
            {'Position': 16, 'Driver': 'LAW', 'Constructor': 'Red Bull Racing', 'Points': 0},
            {'Position': 17, 'Driver': 'BOR', 'Constructor': 'Sauber', 'Points': 0},
            {'Position': 18, 'Driver': 'HUL', 'Constructor': 'Sauber', 'Points': 0},
            {'Position': 19, 'Driver': 'TSU', 'Constructor': 'VCARB', 'Points': 0},
            {'Position': 20, 'Driver': 'ALO', 'Constructor': 'Aston Martin', 'Points': 0}
        ]
    }

def adjust_probabilities(probas, drivers, current_results):
    """Adjust probabilities based on current season results and historical performance"""
    # Create a mapping of driver performance
    driver_performance = {}
    constructor_performance = {}
    
    for race, results in current_results.items():
        for result in results:
            driver = result['Driver']
            constructor = result['Constructor']
            
            # Track driver performance
            if driver not in driver_performance:
                driver_performance[driver] = {'points': 0, 'races': 0, 'positions': []}
            driver_performance[driver]['points'] += result['Points']
            driver_performance[driver]['races'] += 1
            driver_performance[driver]['positions'].append(result['Position'])
            
            # Track constructor performance
            if constructor not in constructor_performance:
                constructor_performance[constructor] = {'points': 0, 'races': 0, 'positions': []}
            constructor_performance[constructor]['points'] += result['Points']
            constructor_performance[constructor]['races'] += 1
            constructor_performance[constructor]['positions'].append(result['Position'])
    
    # Calculate performance metrics for drivers
    for driver in driver_performance:
        races = driver_performance[driver]['races']
        if races > 0:
            driver_performance[driver].update({
                'avg_points': driver_performance[driver]['points'] / races,
                'avg_position': sum(driver_performance[driver]['positions']) / len(driver_performance[driver]['positions']),
                'best_position': min(driver_performance[driver]['positions']),
                'podiums': sum(1 for pos in driver_performance[driver]['positions'] if pos <= 3),
                'wins': sum(1 for pos in driver_performance[driver]['positions'] if pos == 1)
            })
    
    # Calculate performance metrics for constructors
    for constructor in constructor_performance:
        races = constructor_performance[constructor]['races']
        if races > 0:
            constructor_performance[constructor].update({
                'avg_points': constructor_performance[constructor]['points'] / races,
                'avg_position': sum(constructor_performance[constructor]['positions']) / len(constructor_performance[constructor]['positions']),
                'best_position': min(constructor_performance[constructor]['positions']),
                'podiums': sum(1 for pos in constructor_performance[constructor]['positions'] if pos <= 3),
                'wins': sum(1 for pos in constructor_performance[constructor]['positions'] if pos == 1)
            })
    
    # Create a copy of probabilities
    adjusted_probas = probas.copy()
    
    # Calculate maximum points possible from current races
    max_possible_points = sum(25 for race in current_results if not race.endswith('Sprint'))
    max_possible_points += sum(8 for race in current_results if race.endswith('Sprint'))
    
    # Calculate team strength factors based on actual performance
    team_strength = {}
    for constructor in constructor_performance:
        perf = constructor_performance[constructor]
        # Calculate team strength based on:
        # 40% average points, 30% average position, 20% podiums, 10% wins
        points_factor = (perf['avg_points'] / (max_possible_points / 2))  # Normalize to max possible points per race
        position_factor = 1.0 / (perf['avg_position'] / 10)  # Better positions give higher factor
        podium_factor = 1.0 + (perf['podiums'] / perf['races']) if perf['races'] > 0 else 1.0
        win_factor = 1.0 + (perf['wins'] / perf['races']) if perf['races'] > 0 else 1.0
        
        team_strength[constructor] = (
            points_factor * 0.4 +
            position_factor * 0.3 +
            podium_factor * 0.2 +
            win_factor * 0.1
        )
    
    # Normalize team strength factors to be between 0.8 and 1.5
    if team_strength:
        min_strength = min(team_strength.values())
        max_strength = max(team_strength.values())
        strength_range = max_strength - min_strength
        if strength_range > 0:
            for constructor in team_strength:
                team_strength[constructor] = 0.8 + (0.7 * (team_strength[constructor] - min_strength) / strength_range)
    
    # Adjust probabilities based on current performance and team strength
    for i, (driver, constructor) in enumerate(zip(drivers['Driver'], drivers['Constructor'])):
        base_prob = adjusted_probas[i]
        
        if driver in driver_performance:
            perf = driver_performance[driver]
            
            # Calculate performance factors
            position_factor = 1.0 / (perf['avg_position'] / 10)  # Better positions give higher factor
            points_factor = (perf['points'] / max_possible_points) * 2 if max_possible_points > 0 else 1.0
            podium_factor = 1.0 + (perf['podiums'] / perf['races']) if perf['races'] > 0 else 1.0
            win_factor = 1.5 if perf['wins'] > 0 else 1.0
            
            # Combine factors with balanced weights
            # 35% position, 25% points, 20% podiums, 10% wins, 10% team strength
            performance_factor = (
                position_factor * 0.35 +
                points_factor * 0.25 +
                podium_factor * 0.20 +
                win_factor * 0.10
            )
            
            # Apply team strength factor if available
            team_factor = team_strength.get(constructor, 1.0)
            
            # Combine all factors
            combined_factor = (performance_factor * 0.9 + team_factor * 0.1)
            
            # Apply moderate adjustment with upper limit
            adjusted_probas[i] = base_prob * min(combined_factor, 1.8)
        else:
            # For drivers with no current results, use team strength as baseline
            team_factor = team_strength.get(constructor, 1.0)
            adjusted_probas[i] = base_prob * team_factor * 0.9  # Slightly reduced for no current results
    
    # Normalize probabilities with smoothing to prevent extreme values
    total_prob = sum(adjusted_probas)
    if total_prob > 0:
        # Add small constant to prevent zero probabilities
        smoothing_factor = 0.01
        adjusted_probas = (adjusted_probas + smoothing_factor) / (total_prob + len(adjusted_probas) * smoothing_factor)
    
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
    
    # Define which races are sprint races
    sprint_races = {
        'Miami Grand Prix': True,
        'Austrian Grand Prix': True,
        'United States Grand Prix': True,
        'Brazilian Grand Prix': True,
        'Qatar Grand Prix': True,
        'Chinese Grand Prix Sprint': True
    }
    
    # Get results for each race
    results = []
    for race in drivers['GrandPrix'].unique():
        # If we have current results for this race, use them
        if race in current_results:
            results.append({
                'Race': race,
                'Results': current_results[race],
                'IsSprint': sprint_races.get(race, False)
            })
        else:
            # Otherwise, use model predictions
            race_results = drivers[drivers['GrandPrix'] == race].sort_values('Probability', ascending=False).head(top_positions)
            is_sprint = sprint_races.get(race, False)
            results.append({
                'Race': race,
                'Results': [
                    {
                        'Position': pos+1,
                        'Driver': driver,
                        'Constructor': current_drivers[driver],  # Use actual 2025 constructor
                        'Points': calculate_points(pos+1, is_sprint),
                        'Probability': prob
                    }
                    for pos, (_, driver, _, prob) in enumerate(
                        race_results[['Driver', 'Constructor', 'Probability']].itertuples()
                    )
                ],
                'IsSprint': is_sprint
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

def create_correlation_matrix(prediction_df, feature_names):
    """Create and display a correlation matrix of the features"""
    # 1. Select only numeric columns and handle missing values
    numeric_df = prediction_df[feature_names].select_dtypes(include=[np.number])
    
    # Print initial diagnostics
    print("\nInitial Data Diagnostics:")
    print("=======================")
    print(f"Total rows: {len(numeric_df)}")
    print(f"Total features: {len(numeric_df.columns)}")
    print("\nMissing values per column:")
    print(numeric_df.isnull().sum())
    
    # 2. Handle missing values by filling with median (more robust than mean)
    numeric_df = numeric_df.fillna(numeric_df.median())
    
    # 3. Check for and remove constant columns
    constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() == 1]
    if constant_cols:
        print("\nRemoving constant columns:", constant_cols)
        numeric_df = numeric_df.drop(columns=constant_cols)
    
    # 4. Check for highly correlated features (>0.95) to identify potential duplicates
    def find_highly_correlated_pairs(corr_matrix, threshold=0.95):
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        return pairs
    
    # Calculate initial correlations
    initial_corr = numeric_df.corr(method='spearman')  # Using Spearman for better handling of non-linear relationships
    high_corr_pairs = find_highly_correlated_pairs(initial_corr)
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (>0.95):")
        for col1, col2, corr in high_corr_pairs:
            print(f"{col1} - {col2}: {corr:.3f}")
    
    # Create shorter labels for better readability
    short_labels = {
        'QualifyingPosition': 'QualiPos',
        'Grid': 'Grid',
        'AvgQualiPos': 'AvgQuali',
        'StdQualiPos': 'StdQuali',
        'AvgRacePos': 'AvgRace',
        'StdRacePos': 'StdRace',
        'AvgPoints': 'AvgPts',
        'StdPoints': 'StdPts',
        'ConstructorAvgQualiPos': 'ConQuali',
        'ConstructorStdQualiPos': 'ConQualiStd',
        'ConstructorAvgRacePos': 'ConRace',
        'ConstructorStdRacePos': 'ConRaceStd',
        'ConstructorAvgPoints': 'ConPts',
        'ConstructorStdPoints': 'ConPtsStd',
        'TrackAvgQualiPos': 'TrkQuali',
        'TrackStdQualiPos': 'TrkQualiStd',
        'TrackAvgRacePos': 'TrkRace',
        'TrackStdRacePos': 'TrkRaceStd',
        'TrackAvgPoints': 'TrkPts',
        'TrackStdPoints': 'TrkPtsStd'
    }
    
    # Calculate final correlation matrix using Spearman correlation
    corr_matrix = numeric_df.corr(method='spearman')
    
    # Rename columns and index with shorter labels
    corr_matrix.columns = [short_labels.get(col, col) for col in corr_matrix.columns]
    corr_matrix.index = [short_labels.get(idx, idx) for idx in corr_matrix.index]
    
    # Create figure with adjusted size and higher DPI
    plt.figure(figsize=(20, 18), dpi=300)
    
    # Create mask for upper triangle only (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create heatmap with improved styling
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,  # Show correlation values
                cmap='RdBu_r',  # Use a diverging color palette
                center=0,  # Center the colormap at 0
                fmt='.2f',  # Round to 2 decimal places
                square=True,  # Make the plot square
                annot_kws={'size': 10},  # Larger font size for annotations
                cbar_kws={'shrink': .8, 'label': 'Spearman Correlation'},
                vmin=-1, vmax=1)  # Set fixed range for better comparison
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add title and adjust layout
    plt.title('Feature Correlation Matrix (Spearman)\nWhite cells indicate missing correlations', pad=20, fontsize=14)
    
    # Ensure all elements are visible
    plt.tight_layout()
    
    # Save the plot with high DPI and tight bounding box
    plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    # Print strongest correlations
    print("\nStrongest Feature Correlations:")
    print("============================")
    
    # Get all correlations (excluding self-correlations and upper triangle)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            if not np.isnan(corr):  # Only include non-NaN correlations
                correlations.append((col1, col2, abs(corr)))
    
    # Sort by absolute correlation value and print top 10
    correlations.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 Strongest Correlations:")
    for col1, col2, corr in correlations[:10]:
        print(f"{col1} - {col2}: {corr:.3f}")
        
    # Print summary statistics of correlations
    corr_values = [x[2] for x in correlations]
    print("\nCorrelation Statistics:")
    print(f"Mean correlation: {np.mean(corr_values):.3f}")
    print(f"Median correlation: {np.median(corr_values):.3f}")
    print(f"Std deviation: {np.std(corr_values):.3f}")
    
    # Print matrix shape and completeness
    print("\nMatrix Information:")
    print(f"Shape: {corr_matrix.shape}")
    print(f"Total elements: {corr_matrix.size}")
    print(f"Non-NaN elements: {np.sum(~np.isnan(corr_matrix.values))}")
    print(f"Completeness: {(np.sum(~np.isnan(corr_matrix.values)) / corr_matrix.size) * 100:.2f}%")
    
    # Print feature groups with missing correlations
    print("\nFeature Groups with Missing Correlations:")
    feature_groups = {
        'Qualifying': ['QualiPos', 'Grid', 'AvgQuali', 'StdQuali'],
        'Race': ['AvgRace', 'StdRace', 'AvgPts', 'StdPts'],
        'Constructor': ['ConQuali', 'ConQualiStd', 'ConRace', 'ConRaceStd', 'ConPts', 'ConPtsStd'],
        'Track': ['TrkQuali', 'TrkQualiStd', 'TrkRace', 'TrkRaceStd', 'TrkPts', 'TrkPtsStd']
    }
    
    for group_name, features in feature_groups.items():
        missing_count = sum(1 for f in features if f in corr_matrix.columns and corr_matrix[f].isna().any())
        print(f"{group_name}: {missing_count}/{len(features)} features have missing correlations")

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
    
    # Create correlation matrix
    create_correlation_matrix(prediction_df, feature_names)
    
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
        print(f"\n{race['Race']} {'(Sprint)' if race['IsSprint'] else ''}:")
        for result in race['Results']:
            print(f"{result['Position']}. {result['Driver']} ({result['Constructor']}) - {result['Points']} points ({result.get('Probability', 'N/A')} probability)")

if __name__ == "__main__":
    # Import required functions from predict_2025.py
    from predict_2025 import load_and_prepare_stats, prepare_prediction_data
    main() 