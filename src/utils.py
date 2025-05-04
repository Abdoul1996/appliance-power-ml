import pandas as pd

def load_appliance_energy_data():
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(name='Appliances Energy Prediction')
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    
    # Clean and convert date column
    df['date'] = df['date'].str.replace('–', '-', regex=False)  # Fix dashes
    df['date'] = df['date'].str.replace(r'(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})', r'\1T\2', regex=True)  # Add 'T'
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S')  # Convert to datetime
    
    return df
