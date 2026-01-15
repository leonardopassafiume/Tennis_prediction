import pandas as pd
import tennis_bot
import weather_utils
from datetime import datetime

def test_weather_integration():
    print("Testing Weather Utils...")
    # Test Metadata
    meta = weather_utils.get_tournament_meta("Australian Open")
    print(f"Australian Open Meta: {meta}")
    assert meta['type'] == 'Outdoor'
    
    meta_indoor = weather_utils.get_tournament_meta("Paris Masters")
    print(f"Paris Masters Meta: {meta_indoor}")
    assert meta_indoor['type'] == 'Indoor'
    
    # Test Weather Fetch (cached or live)
    # Melbourne, 2023-01-20
    print("Fetching weather for Melbourne 2023-01-20...")
    w = weather_utils.weather_client.get_weather(-37.814, 144.963, datetime(2023, 1, 20))
    print(f"Weather: {w}")
    assert 'temp' in w and 'humidity' in w
    
    # Test Process Data Integration
    print("\nTesting process_data integration...")
    # Create dummy dataframe
    df_test = pd.DataFrame([{
        'tourney_name': 'Australian Open',
        'surface': 'Hard',
        'tourney_date': 20230116,
        'match_num': 1,
        'winner_name': 'Jannik Sinner', 'loser_name': 'Kyle Edmund',
        'winner_id': 1, 'loser_id': 2,
        'winner_rank': 10, 'loser_rank': 100,
        'winner_ht': 188, 'loser_ht': 185,
        'winner_age': 21, 'loser_age': 28,
        'score': '6-4 6-0 6-2',
        'best_of': 5,
        'winner_ioc': 'ITA', 'loser_ioc': 'GBR',
        'w_ace': 10, 'l_ace': 5
    }])
    
    # Add dummy Date column normally computed
    df_test['Date'] = pd.to_datetime(df_test['tourney_date'], format='%Y%m%d')
    
    processed, _, _, _, _, _ = tennis_bot.process_data(df_test)
    
    print("\nProcessed Data Columns:")
    print(processed.columns)
    
    print("\nFirst Row Features:")
    print(processed.iloc[0][['Temperature', 'Humidity', 'Is_Indoor']])
    
    assert 'Temperature' in processed.columns
    assert 'Humidity' in processed.columns
    assert 'Is_Indoor' in processed.columns
    assert processed.iloc[0]['Is_Indoor'] == 0 # Australian Open is Outdoor
    
    print("\nSUCCESS: Verification Passed.")

if __name__ == "__main__":
    test_weather_integration()
