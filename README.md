# F1 Championship Prediction Model

This project uses machine learning to predict Formula 1 race results and championship standings. It collects historical F1 data from the Ergast API and uses it to train a model that can predict race outcomes and championship standings.

## Features

- Historical data collection from Ergast API (2021-2024)
- Data preprocessing and feature engineering
- Machine learning model for race result prediction
- Championship standings prediction
- Integration with current season results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/KGlitch/F1-Predictions.git
cd F1-Predictions
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Collect historical data:
```bash
python collect_training_data.py
```

2. Train the model:
```bash
python train_model.py
```

3. Make predictions:
```bash
python predict_championship.py
```

## Project Structure

- `collect_training_data.py`: Collects historical F1 data from Ergast API
- `prepare_training_data.py`: Preprocesses and engineers features from raw data
- `train_model.py`: Trains the machine learning model
- `predict_championship.py`: Makes predictions for race results and championship standings
- `predict_2025.py`: Specific predictions for the 2025 season
- `f1_historical_data.csv`: Historical race data
- `f1_model.joblib`: Trained model
- `f1_scaler.joblib`: Feature scaler
- `f1_imputer.joblib`: Missing value imputer

## Data Sources

- Race results and qualifying data from Ergast API
- Current season results from official F1 sources

## License

MIT License 