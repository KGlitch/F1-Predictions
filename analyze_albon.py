import pandas as pd

def analyze_albon_2024():
    """Analyze Alexander Albon's positions in 2024"""
    # Load the data
    df = pd.read_csv('f1_historical_data.csv')
    
    # Filter for Albon in 2024
    albon_data = df[
        (df['DriverName'].str.contains('Albon', case=False)) & 
        (df['Year'] == 2024)
    ].sort_values('Date')
    
    if len(albon_data) == 0:
        print("Keine Daten für Alexander Albon in 2024 gefunden.")
        return
    
    print("\nAlexander Albon - Positionen 2024:")
    print("=================================")
    for _, race in albon_data.iterrows():
        print(f"\n{race['GrandPrix']}:")
        print(f"Qualifying: Position {race['QualifyingPosition']}")
        print(f"Startposition: Position {race['Grid']}")
        print(f"Rennergebnis: Position {race['RacePosition']}")
        print(f"Punkte: {race['Points']}")
        print(f"Status: {race['Status']}")

if __name__ == "__main__":
    analyze_albon_2024() 