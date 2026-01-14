import requests
import re
import pandas as pd
import time
import json
import os
from datetime import datetime

# CONFIGURATION
OUTPUT_FILE = "atp_matches_2025_scraped.csv"
PLAYERS_LIMIT = 120 # Cover Top 100 + margin
DELAY = 2 # Moderate delay

def get_top_players():
    print("Fetching top players from Elo ratings...")
    url = "http://www.tennisabstract.com/reports/atp_elo_ratings.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching Elo list: {e}")
        return []

    # Regex to find links like <a href="...player.cgi?p=JannikSinner">Jannik Sinner</a>
    ids = re.findall(r'player\.cgi\?p=([^"&]+)', response.text)
    
    unique_ids = []
    seen = set()
    for pid in ids:
        if pid not in seen and '/' not in pid: 
            unique_ids.append(pid)
            seen.add(pid)
            
    print(f"Found {len(unique_ids)} unique player IDs.")
    return unique_ids[:PLAYERS_LIMIT]

def get_player_matches(player_id):
    url = f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={player_id}"
    print(f"Fetching matches for {player_id}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {player_id}: {e}")
        return [], {}
    
    # Extract player metadata from JS variables
    meta = {}
    try:
        # Use regex to find variables safely, handling potential missing matches
        meta['name'] = re.search(r"var fullname = '(.*?)';", response.text).group(1)
        meta['dob'] = re.search(r"var dob = (\d+);", response.text).group(1)
        meta['ht'] = re.search(r"var ht = (\d+);", response.text).group(1)
        meta['hand'] = re.search(r"var hand = '(.*?)';", response.text).group(1)
    except AttributeError:
        # Fallback if vars are missing
        print(f"Metadata missing for {player_id}")
        return [], {}

    # Extract var matchmx = [[...]];
    # Regex: var matchmx = [ ... ]; (allowing for newlines)
    match = re.search(r'var matchmx\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
    if not match:
        print(f"No match data found for {player_id}")
        return [], {}
    
    js_array = match.group(1)
    
    # Clean JS literals to Python
    js_array = js_array.replace('null', 'None').replace('true', 'True').replace('false', 'False')
    
    try:
        import ast
        matches = ast.literal_eval(js_array)
        return matches, meta
    except Exception as e:
        print(f"Error parsing match data for {player_id}: {e}")
        return [], {}

def parse_matches(raw_matches, meta):
    parsed = []
    
    p1_name = meta.get('name', 'Unknown')
    p1_dob = meta.get('dob', 0)
    p1_ht = meta.get('ht', 0)
    p1_hand = meta.get('hand', 'U')

    for match in raw_matches:
        try:
            date_str = match[0]
            if len(date_str) == 8:
                year = int(date_str[:4])
                
                # Filter for 2025 and 2026
                if year >= 2025:
                    
                    winner_flag = match[4] # "W" or "L"
                    opponent_name = match[11]
                    
                    if winner_flag == 'W':
                        winner_name = p1_name
                        loser_name = opponent_name
                        winner_rank = match[5]
                        loser_rank = match[12]
                        winner_hand = p1_hand
                        loser_hand = match[15]
                        
                        # Age calc
                        try:
                            w_dob = datetime.strptime(str(p1_dob), "%Y%m%d")
                            w_age = (datetime.strptime(date_str, "%Y%m%d") - w_dob).days / 365.25
                        except:
                            w_age = 24.0 
                            
                        try:
                            l_dob_str = match[16]
                            l_dob = datetime.strptime(str(l_dob_str), "%Y%m%d")
                            l_age = (datetime.strptime(date_str, "%Y%m%d") - l_dob).days / 365.25
                        except:
                            l_age = 24.0
                            
                        winner_ht = p1_ht
                        loser_ht = match[17]
                    
                    else:
                        winner_name = opponent_name
                        loser_name = p1_name
                        winner_rank = match[12]
                        loser_rank = match[5]
                        winner_hand = match[15]
                        loser_hand = p1_hand
                        
                        try:
                            w_dob_str = match[16]
                            w_dob = datetime.strptime(str(w_dob_str), "%Y%m%d")
                            w_age = (datetime.strptime(date_str, "%Y%m%d") - w_dob).days / 365.25
                        except:
                            w_age = 24.0
                            
                        try:
                            l_dob = datetime.strptime(str(p1_dob), "%Y%m%d")
                            l_age = (datetime.strptime(date_str, "%Y%m%d") - l_dob).days / 365.25
                        except:
                            l_age = 24.0
                            
                        winner_ht = match[17]
                        loser_ht = p1_ht

                    match_data = {
                        'tourney_date': date_str,
                        'tourney_name': match[1],
                        'surface': match[2],
                        'winner_name': winner_name,
                        'loser_name': loser_name,
                        'score': match[9],
                        'winner_rank': winner_rank,
                        'loser_rank': loser_rank,
                        'winner_age': round(w_age, 2),
                        'loser_age': round(l_age, 2),
                        'winner_ht': winner_ht,
                        'loser_ht': loser_ht,
                        'winner_hand': winner_hand,
                        'loser_hand': loser_hand,
                        'winner_rank_points': 0,
                        'loser_rank_points': 0
                    }
                    parsed.append(match_data)
        except Exception as e:
            print(f"Skipping match: {e}")
            continue
            
    return parsed

def main():
    players = get_top_players()
    all_matches = []
    
    for player_id in players:
        matches, meta = get_player_matches(player_id)
        if matches and meta:
            parsed = parse_matches(matches, meta)
            all_matches.extend(parsed)
            print(f"Scraped {len(parsed)} matches for {meta['name']}")
            time.sleep(DELAY)
        
    # Create DataFrame
    if not all_matches:
        print("No matches scraped.")
        return

    # Dedup on file write to handle the overlap (since we scrape both P1 and P2 pages)
    df = pd.DataFrame(all_matches)
    if not df.empty:
        # Drop duplicates based on date, winner, loser (ignoring other fields which might slightly vary)
        df.drop_duplicates(subset=['tourney_date', 'winner_name', 'loser_name'], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} unique matches to {OUTPUT_FILE}")
    else:
        print("No matches found.")
    
    # Show sample
    print(df.head())

if __name__ == "__main__":
    main()
