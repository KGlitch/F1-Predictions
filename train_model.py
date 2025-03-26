import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_training_data(df):
    """Prepare training data from historical data"""
    print("Preparing training data...")
    
    # Calculate driver statistics
    driver_stats = df.groupby('DriverId').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    driver_stats.columns = ['AvgQualiPos', 'StdQualiPos',
                          'AvgRacePos', 'StdRacePos',
                          'AvgPoints', 'StdPoints']
    
    # Calculate constructor statistics
    constructor_stats = df.groupby('Constructor').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    constructor_stats.columns = ['ConstructorAvgQualiPos', 'ConstructorStdQualiPos',
                               'ConstructorAvgRacePos', 'ConstructorStdRacePos',
                               'ConstructorAvgPoints', 'ConstructorStdPoints']
    
    # Calculate track statistics
    track_stats = df.groupby('GrandPrix').agg({
        'QualifyingPosition': ['mean', 'std'],
        'RacePosition': ['mean', 'std'],
        'Points': ['mean', 'std']
    })
    track_stats.columns = ['TrackAvgQualiPos', 'TrackStdQualiPos',
                          'TrackAvgRacePos', 'TrackStdRacePos',
                          'TrackAvgPoints', 'TrackStdPoints']
    
    # Prepare features for each race
    X = []
    y = []
    
    for _, row in df.iterrows():
        driver_id = row['DriverId']
        constructor = row['Constructor']
        track = row['GrandPrix']
        
        # Get driver statistics
        driver_data = driver_stats.loc[driver_id] if driver_id in driver_stats.index else pd.Series(np.nan, index=driver_stats.columns)
        
        # Get constructor statistics
        constructor_data = constructor_stats.loc[constructor] if constructor in constructor_stats.index else pd.Series(np.nan, index=constructor_stats.columns)
        
        # Get track statistics
        track_data = track_stats.loc[track] if track in track_stats.index else pd.Series(np.nan, index=track_stats.columns)
        
        # Create feature vector
        features = {
            'QualifyingPosition': row['QualifyingPosition'],
            'Grid': row['Grid'],
            'AvgQualiPos': driver_data['AvgQualiPos'],
            'StdQualiPos': driver_data['StdQualiPos'],
            'AvgRacePos': driver_data['AvgRacePos'],
            'StdRacePos': driver_data['StdRacePos'],
            'AvgPoints': driver_data['AvgPoints'],
            'StdPoints': driver_data['StdPoints'],
            'ConstructorAvgQualiPos': constructor_data['ConstructorAvgQualiPos'],
            'ConstructorStdQualiPos': constructor_data['ConstructorStdQualiPos'],
            'ConstructorAvgRacePos': constructor_data['ConstructorAvgRacePos'],
            'ConstructorStdRacePos': constructor_data['ConstructorStdRacePos'],
            'ConstructorAvgPoints': constructor_data['ConstructorAvgPoints'],
            'ConstructorStdPoints': constructor_data['ConstructorStdPoints'],
            'TrackAvgQualiPos': track_data['TrackAvgQualiPos'],
            'TrackStdQualiPos': track_data['TrackStdQualiPos'],
            'TrackAvgRacePos': track_data['TrackAvgRacePos'],
            'TrackStdRacePos': track_data['TrackStdRacePos'],
            'TrackAvgPoints': track_data['TrackAvgPoints'],
            'TrackStdPoints': track_data['TrackStdPoints']
        }
        
        X.append(features)
        y.append(1 if row['RacePosition'] <= 3 else 0)
    
    X = pd.DataFrame(X)
    y = np.array(y)
    
    # Print info about missing values
    print("\nMissing values per feature:")
    print(X.isnull().sum())
    
    return X, y

def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate different models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        test_score = model.score(X_test_scaled, y_test)
        print(f"Test set score: {test_score:.3f}")
        
        # Print detailed classification report
        y_pred = model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Update best model if necessary
        if test_score > best_score:
            best_model = model
            best_score = test_score
    
    return best_model, scaler, imputer, feature_names

def analyze_feature_importance(model, feature_names):
    """Analyze and plot feature importance"""
    print("\nAnalyzing feature importance...")
    
    # Get feature importance (handle both model types)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("This model type doesn't provide feature importances directly.")
        return
    
    # Create DataFrame with feature names and importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Print feature importance
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Top 3 Finish Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def save_model(model, scaler, imputer, feature_names):
    """Save the trained model, scaler, and imputer"""
    import joblib
    
    print("\nSaving model, scaler, and imputer...")
    joblib.dump(model, 'f1_model.joblib')
    joblib.dump(scaler, 'f1_scaler.joblib')
    joblib.dump(imputer, 'f1_imputer.joblib')
    pd.DataFrame({'feature': feature_names}).to_csv('feature_names.csv', index=False)
    print("Model, scaler, and imputer saved successfully!")

def main():
    # Load historical data
    print("Loading historical data...")
    df = pd.read_csv('f1_historical_data.csv')
    
    # Prepare training data
    X, y = prepare_training_data(df)
    feature_names = X.columns.tolist()
    
    # Train and evaluate models
    best_model, scaler, imputer, feature_names = train_and_evaluate_models(X, y, feature_names)
    
    # Analyze feature importance
    analyze_feature_importance(best_model, feature_names)
    
    # Save the model
    save_model(best_model, scaler, imputer, feature_names)

if __name__ == "__main__":
    main() 