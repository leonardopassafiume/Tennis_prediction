import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
import os
import requests
from io import StringIO
import time

# CONFIGURAZIONE
START_YEAR = 2016
END_YEAR = 2025 # Proveremo a scaricare fino al 2025, gestiremo il 404
CACHE_DIR = "tennis_data_cache" # Cartella per salvare i csv locali
os.makedirs(CACHE_DIR, exist_ok=True)


# 1. CARICAMENTO DATI ROBUSTO
def load_data():
    base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    years = range(START_YEAR, END_YEAR + 1) # Usa le variabili globali
    all_data = []
    
    print(f"--- Caricamento Dati ({START_YEAR}-{END_YEAR}) ---")
    
    # 1. Carica dati storici (Jeff Sackmann) - ATP Main + Challenger/Qual
    # Base URL per ATP
    base_url_atp = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    # Base URL per Challenger/Qual
    base_url_chal = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{}.csv"
    
    for year in years:
        # --- ATP Main ---
        file_path_atp = os.path.join(CACHE_DIR, f"atp_matches_{year}.csv")
        if not os.path.exists(file_path_atp):
            print(f"[{year}] Scaricando ATP Main...")
            try:
                response = requests.get(base_url_atp.format(year))
                if response.status_code == 200:
                    with open(file_path_atp, "wb") as f:
                        f.write(response.content)
                    all_data.append(pd.read_csv(StringIO(response.text)))
                else:
                    print(f"[{year}] ATP Main non trovato ({response.status_code})")
            except Exception as e:
                print(f"[{year}] Errore download ATP: {e}")
        else:
            # print(f"[{year}] Caricando ATP da cache...")
            all_data.append(pd.read_csv(file_path_atp))

        # --- Challenger/Qualifying ---
        file_path_chal = os.path.join(CACHE_DIR, f"atp_matches_qual_chall_{year}.csv")
        if not os.path.exists(file_path_chal):
            print(f"[{year}] Scaricando Challenger/Qual...")
            try:
                response = requests.get(base_url_chal.format(year))
                if response.status_code == 200:
                    with open(file_path_chal, "wb") as f:
                        f.write(response.content)
                    try:
                         all_data.append(pd.read_csv(StringIO(response.text)))
                    except:
                        pass # A volte formattazione errata
                else:
                    print(f"[{year}] Challenger non trovato ({response.status_code})")
            except Exception as e:
                 print(f"[{year}] Errore download Challenger: {e}")
        else:
            # print(f"[{year}] Caricando Challenger da cache...")
            try:
                all_data.append(pd.read_csv(file_path_chal))
            except:
                pass

    full_historical_df = pd.concat(all_data, ignore_index=True)
    
    # Costruisci mappa Nome -> ID e IOC dai dati storici
    name_to_id = {}
    name_to_ioc = {}
    # Priorità ai match più recenti per i mapping
    for _, row in full_historical_df.sort_values('tourney_date').iterrows():
        if pd.notna(row['winner_name']) and pd.notna(row['winner_id']):
            name_to_id[row['winner_name'].lower().strip()] = row['winner_id']
        if pd.notna(row['loser_name']) and pd.notna(row['loser_id']):
            name_to_id[row['loser_name'].lower().strip()] = row['loser_id']
            
        # Mappa IOC (Nazionalità)
        if pd.notna(row['winner_name']) and pd.notna(row['winner_ioc']):
            name_to_ioc[row['winner_name'].lower().strip()] = row['winner_ioc']
        if pd.notna(row['loser_name']) and pd.notna(row['loser_ioc']):
            name_to_ioc[row['loser_name'].lower().strip()] = row['loser_ioc']

    # 2. Carica dati 2025/2026 scrapati
    scraped_file = "atp_matches_2025_scraped.csv"
    if os.path.exists(scraped_file):
        print(f"Caricamento dati scrapati 2025/2026 da {scraped_file}...")
        try:
            df_scraped = pd.read_csv(scraped_file)
            
            # Fix IDs using name mapping
            def get_id(name):
                return name_to_id.get(str(name).lower().strip(), 0)
            
            # Backfill IOC codes using name mapping
            def get_ioc(name):
                return name_to_ioc.get(str(name).lower().strip(), 'UNK') # UNK se sconosciuto
                
            df_scraped['winner_id'] = df_scraped['winner_name'].apply(get_id)
            df_scraped['loser_id'] = df_scraped['loser_name'].apply(get_id)
            
            # Create winner_ioc and loser_ioc columns if not exist
            df_scraped['winner_ioc'] = df_scraped['winner_name'].apply(get_ioc)
            df_scraped['loser_ioc'] = df_scraped['loser_name'].apply(get_ioc)
            
            # Aggiungi colonne mancanti
            missing_cols = ['tourney_id', 'draw_size', 'tourney_level', 'match_num', 
                            'winner_seed', 'winner_entry', 'loser_seed', 'loser_entry', 'winner_ioc', 'loser_ioc', 
                            'best_of', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
                            'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 
                            'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank_points', 'loser_rank_points']
            for col in missing_cols:
                if col not in df_scraped.columns:
                    df_scraped[col] = 0 
            
            # Concatena tutto
            full_df = pd.concat([full_historical_df, df_scraped], ignore_index=True)
            return full_df
            
        except Exception as e:
            print(f"Errore lettura dati scrapati: {e}")
            return full_historical_df
            
    return full_historical_df

# --- 2. ELABORAZIONE DATI (FEATURE ENGINEERING) ---

# Mappa Tornei -> Nazione (Global)
TOURNEY_COUNTRY_MAP = {
    'Australian Open': 'AUS', 'Brisbane': 'AUS', 'Sydney': 'AUS', 'Adelaide': 'AUS', 'United Cup': 'AUS',
    'Roland Garros': 'FRA', 'Paris Masters': 'FRA', 'Marseille': 'FRA', 'Montpellier': 'FRA', 'Lyon': 'FRA', 'Metz': 'FRA',
    'Wimbledon': 'GBR', 'Queen\'s Club': 'GBR', 'Eastbourne': 'GBR',
    'US Open': 'USA', 'Indian Wells': 'USA', 'Miami': 'USA', 'Cincinnati': 'USA', 'Washington': 'USA', 'Delray Beach': 'USA', 'Houston': 'USA', 'Atlanta': 'USA', 'Winston-Salem': 'USA', 'Dallas': 'USA',
    'Rome': 'ITA', 'Rome Masters': 'ITA', 'Turin': 'ITA', 'Tour Finals': 'ITA',
    'Madrid': 'ESP', 'Barcelona': 'ESP',
    'Monte Carlo': 'MON', 
    'Canada Masters': 'CAN', 'Toronto': 'CAN', 'Montreal': 'CAN',
    'Shanghai': 'CHN', 'Beijing': 'CHN', 'Chengdu': 'CHN', 'Zhuhai': 'CHN',
    'Tokyo': 'JPN',
    'Halle': 'GER', 'Hamburg': 'GER', 'Munich': 'GER', 'Stuttgart': 'GER',
    'Vienna': 'AUT', 'Kitzbuhel': 'AUT',
    'Swiss Indoors': 'SUI', 'Gstaad': 'SUI', 'Geneva': 'SUI',
    'Bastad': 'SWE', 'Stockholm': 'SWE',
    'Umag': 'CRO',
    'Estoril': 'POR',
    'Rio de Janeiro': 'BRA',
    'Buenos Aires': 'ARG', 'Cordoba': 'ARG',
    'Santiago': 'CHI',
    'Auckland': 'NZL',
    'Dubai': 'UAE', 'Doha': 'QAT',
    'Rotterdam': 'NED', 's-Hertogenbosch': 'NED',
    'Antwerp': 'BEL'
}

def process_data(df, players_last_stats=None, predict_mode=False, tourney_country_map=None):
    if tourney_country_map is None:
        tourney_country_map = {}

    start_time = time.time()
    
    # 1. Dizionari di Storia (Tracking)
    history = {}        # {player_id: {matches: [], surface_wins: {}, ...}}
    surface_history = {} # {player_id: {'Hard': [], 'Clay': [], ...}}
    quality_history = {} # {player_id: []} -> Stores quality points of recent matches
    serve_history = {}   # {player_id: []} -> Stores serve ratings
    h2h_stats = {}       # {(p1, p2): {p1: wins, p2: wins}}
    
    processed_rows = []
    
    # Ordina cronologicamente
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.sort_values(['Date', 'match_num']).reset_index(drop=True)
    
    # Helper per Zero Stats
    def impute_stats(pid, stat_val, stat_name, default_val=0.6):
        if stat_val > 0:
            return stat_val
        # Try to get from history
        if pid in history and len(history[pid]['matches']) > 0:
            last_stats = history[pid]['matches'][-20:] # Last 20
            vals = [m.get(stat_name, 0) for m in last_stats if m.get(stat_name, 0) > 0]
            if vals:
                return np.mean(vals)
        return default_val

    # Helper Surface Form
    def get_surface_form(pid, surf):
        if pid not in surface_history or surf not in surface_history[pid]:
            return 0.0
        # Last 5 matches on this surface
        last_5 = surface_history[pid][surf][-5:]
        if not last_5:
            return 0.0
        return sum(last_5) / len(last_5)

    # Helper Quality Form
    def get_quality_form(pid):
        if pid not in quality_history or not quality_history[pid]:
            return 0.0
        # Rolling avg of last 10 quality scores
        return np.mean(quality_history[pid][-10:])

    # Helper Serve Rating
    def get_serve_rating(pid):
        if pid not in serve_history or not serve_history[pid]:
            return 0.5 # Default avg
        return np.mean(serve_history[pid][-10:])

    print("Engineering features...")
    
    for idx, row in df.iterrows():
        p1 = row['winner_name']
        p2 = row['loser_name']
        
        # Init player history if new
        for p in [p1, p2]:
            if p not in history:
                history[p] = {'matches': [], 'elo': 1500, 'elo_surface': {'Hard': 1500, 'Clay': 1500, 'Grass': 1500, 'Carpet': 1500}}
                surface_history[p] = {'Hard': [], 'Clay': [], 'Grass': [], 'Carpet': []}
                quality_history[p] = []
                serve_history[p] = []

        # --- RECOVER STATS & IMPUTE (Fix Zero Stats) ---
        # Raw percentages
        w_svpt = float(row.get('w_svpt', 0))
        l_svpt = float(row.get('l_svpt', 0))
        
        # Impute if missing (assuming 0 means missing for aces/svpt)
        # We calculate Key Stats: 1st Won %, 2nd Won %, BP Saved %
        
        # Winner
        w_1st_won_pct = float(row.get('w_1stWon', 0)) / float(row.get('w_1stIn', 1)) if float(row.get('w_1stIn', 0)) > 0 else 0
        w_2nd_won_pct = float(row.get('w_2ndWon', 0)) / (float(row.get('w_svpt', 1)) - float(row.get('w_1stIn', 0))) if (float(row.get('w_svpt', 0)) - float(row.get('w_1stIn', 0))) > 0 else 0
        w_bp_saved_pct = float(row.get('w_bpSaved', 0)) / float(row.get('w_bpFaced', 1)) if float(row.get('w_bpFaced', 0)) > 0 else 0.5 
        
        # Impute Winner Stats if suspiciously low
        w_1st_won_pct = impute_stats(p1, w_1st_won_pct, '1st_won_pct', 0.7)
        w_2nd_won_pct = impute_stats(p1, w_2nd_won_pct, '2nd_won_pct', 0.5)
        
        # Loser
        l_1st_won_pct = float(row.get('l_1stWon', 0)) / float(row.get('l_1stIn', 1)) if float(row.get('l_1stIn', 0)) > 0 else 0
        l_2nd_won_pct = float(row.get('l_2ndWon', 0)) / (float(row.get('l_svpt', 1)) - float(row.get('l_1stIn', 0))) if (float(row.get('l_svpt', 0)) - float(row.get('l_1stIn', 0))) > 0 else 0
        l_bp_saved_pct = float(row.get('l_bpSaved', 0)) / float(row.get('l_bpFaced', 1)) if float(row.get('l_bpFaced', 0)) > 0 else 0.5
        
        l_1st_won_pct = impute_stats(p2, l_1st_won_pct, '1st_won_pct', 0.6)
        l_2nd_won_pct = impute_stats(p2, l_2nd_won_pct, '2nd_won_pct', 0.45)
        
        # --- CALCULATE FEATURES (Before Update) ---
        surf = row['surface']
        if surf not in ['Hard', 'Clay', 'Grass', 'Carpet']: surf = 'Hard'
        
        # 1. Elo & Rank
        elo1, elo2 = history[p1]['elo'], history[p2]['elo']
        elo_surf1, elo_surf2 = history[p1]['elo_surface'][surf], history[p2]['elo_surface'][surf]
        rank1, rank2 = float(row['winner_rank'] or 500), float(row['loser_rank'] or 500)
        
        # 2. Form (Global)
        matches1 = history[p1]['matches'][-10:]
        matches2 = history[p2]['matches'][-10:]
        form1 = sum([m['pts'] for m in matches1]) / 10.0
        form2 = sum([m['pts'] for m in matches2]) / 10.0
        
        # 3. Surface Form (NEW)
        surf_form1 = get_surface_form(p1, surf)
        surf_form2 = get_surface_form(p2, surf)
        
        # 4. Quality Form (NEW)
        qual_form1 = get_quality_form(p1)
        qual_form2 = get_quality_form(p2)
        
        # 5. Serve Rating (NEW)
        serve_rat1 = get_serve_rating(p1)
        serve_rat2 = get_serve_rating(p2)
        
        # 6. Fatigue (Days)
        date_curr = row['Date']
        last_date1 = history[p1]['matches'][-1]['date'] if matches1 else date_curr
        last_date2 = history[p2]['matches'][-1]['date'] if matches2 else date_curr
        days1 = (date_curr - last_date1).days
        days2 = (date_curr - last_date2).days
        
        # 7. H2H
        match_key = tuple(sorted([p1, p2])) # Use strings if IDs not avail? Or use P1 name? 
        # Previous code used P1_ID / P2_ID. But here we iterate strings? 
        # Wait, row['winner_id'] exists but sometimes we prefer names if cleaner?
        # Let's use IDs to be safe if consistent. If 0, use name.
        p1_id = row['winner_id']
        p2_id = row['loser_id']
        match_key = tuple(sorted([p1_id, p2_id]))
        
        if match_key not in h2h_stats:
            h2h_stats[match_key] = {p1_id: 0, p2_id: 0}
        
        p1_h2h = h2h_stats[match_key].get(p1_id, 0)
        p2_h2h = h2h_stats[match_key].get(p2_id, 0)
        
        # 8. Pressure/Clutch (Historical Avg)
        def get_avg_stat(matches, key):
            vals = [m[key] for m in matches if key in m]
            return np.mean(vals) if vals else 0.5
            
        bp_save1 = get_avg_stat(matches1, 'bp_saved_pct')
        bp_save2 = get_avg_stat(matches2, 'bp_saved_pct')
        tb_win1 = get_avg_stat(matches1, 'tb_win')
        tb_win2 = get_avg_stat(matches2, 'tb_win')
        decider_win1 = get_avg_stat(matches1, 'decider_win')
        decider_win2 = get_avg_stat(matches2, 'decider_win')

        # 9. Home Advantage
        p1_ioc = row.get('winner_ioc', 'UNK')
        p2_ioc = row.get('loser_ioc', 'UNK')
        tourney_name = row['tourney_name']
        t_country = tourney_country_map.get(tourney_name, 'UNK')
        
        is_home1 = 1 if p1_ioc == t_country and t_country != 'UNK' else 0
        is_home2 = 1 if p2_ioc == t_country and t_country != 'UNK' else 0
        
        # 10. Big Server (NEW)
        ht1 = float(row.get('winner_ht', 185))
        ht2 = float(row.get('loser_ht', 185))
        big_server1 = 1 if ht1 > 195 else 0
        big_server2 = 1 if ht2 > 195 else 0
        
        # 11. Lefty Matchup (NEW)
        hand1 = str(row.get('winner_hand', 'R'))
        hand2 = str(row.get('loser_hand', 'R'))
        is_lefty_matchup = 1 if (hand1 == 'L' and hand2 == 'R') or (hand1 == 'R' and hand2 == 'L') else 0

        # --- PREPARE ROW FOR TRAINING ---
        # Features: Diff (P1 - P2) for symmetric training
        
        common_features = {
            'Diff_Rank': np.log(rank2 + 1) - np.log(rank1 + 1), 
            'Diff_Elo': elo1 - elo2,
            'Diff_Elo_Surface': elo_surf1 - elo_surf2,
            'Diff_Form': form1 - form2,
            'Diff_Form_Surface': surf_form1 - surf_form2,      # NEW
            'Diff_Quality': qual_form1 - qual_form2,           # NEW
            'Diff_Serve_Rating': serve_rat1 - serve_rat2,      # NEW
            'Diff_Big_Server': big_server1 - big_server2,      # NEW
            'Diff_Height': ht1 - ht2,
            'Diff_Age': float(row['winner_age']) - float(row['loser_age']),
            'Diff_Days': days1 - days2,
            'Diff_BP_Save': bp_save1 - bp_save2,
            'Diff_TB_Win': tb_win1 - tb_win2,
            'Diff_Decider': decider_win1 - decider_win2,
            'Diff_Home': is_home1 - is_home2, 
            'Is_Lefty': is_lefty_matchup,                      # NEW
            'Target': 1, # P1 (Winner) won
            'Date': row['Date'],
            'P1_ID': p1_id,
            'P2_ID': p2_id
        }
        
        processed_rows.append(common_features)
        
        # Reverse (Loser perspective)
        reverse_features = common_features.copy()
        for k in list(reverse_features.keys()):
            if k.startswith('Diff_'):
                reverse_features[k] = -reverse_features[k]
        reverse_features['Target'] = 0
        processed_rows.append(reverse_features)
        
        # --- UPDATE HISTORY (Post-Match) ---
        
        # Current Match Stats
        sr1 = (w_1st_won_pct * 0.4) + (w_2nd_won_pct * 0.6)
        sr2 = (l_1st_won_pct * 0.4) + (l_2nd_won_pct * 0.6)
        
        # Quality Points: 1 + (100 / (Opponent_Rank + 1))
        # Logic: Winner gets points based on loser rank. Loser gets 0.
        q_pts1 = 1 + (100 / (rank2 + 1))
        
        # Update Elo
        k_factor = 20
        exp1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        new_elo1 = elo1 + k_factor * (1 - exp1)
        new_elo2 = elo2 + k_factor * (0 - (1 - exp1))
        
        # Update Surface Elo
        k_surf = 20
        exp_s1 = 1 / (1 + 10 ** ((elo_surf2 - elo_surf1) / 400))
        new_surf1 = elo_surf1 + k_surf * (1 - exp_s1)
        new_surf2 = elo_surf2 + k_surf * (0 - (1 - exp_s1))
        
        # Update Trackers P1 (Winner)
        history[p1]['elo'] = new_elo1
        history[p1]['elo_surface'][surf] = new_surf1
        
        score_str = str(row['score']) if pd.notna(row['score']) else ""
        
        history[p1]['matches'].append({
            'date': date_curr, 'opponent': p2, 'result': 'W', 'pts': 1, 
            'bp_saved_pct': w_bp_saved_pct, 
            'tb_win': 1 if '7-6' in score_str else 0.5, 
            'decider_win': 1 if len(score_str) > 15 else 0.5,
            '1st_won_pct': w_1st_won_pct,
            '2nd_won_pct': w_2nd_won_pct
        })
        surface_history[p1][surf].append(1) # Win
        quality_history[p1].append(q_pts1)
        serve_history[p1].append(sr1)
        
        # Update Trackers P2 (Loser)
        history[p2]['elo'] = new_elo2
        history[p2]['elo_surface'][surf] = new_surf2
        history[p2]['matches'].append({
            'date': date_curr, 'opponent': p1, 'result': 'L', 'pts': 0,
            'bp_saved_pct': l_bp_saved_pct,
            'tb_win': 0 if '7-6' in score_str else 0.5,
            'decider_win': 0 if len(score_str) > 15 else 0.5,
            '1st_won_pct': l_1st_won_pct,
            '2nd_won_pct': l_2nd_won_pct
        })
        surface_history[p2][surf].append(0) # Loss
        quality_history[p2].append(0)       
        serve_history[p2].append(sr2)
        
        # Update H2H (Post-match)
        if match_key not in h2h_stats: h2h_stats[match_key] = {p1_id: 0, p2_id: 0}
        h2h_stats[match_key][p1_id] += 1

    print(f"Dataset finale pronto: {len(processed_rows)} match.")
    return pd.DataFrame(processed_rows), history, surface_history, quality_history, serve_history, h2h_stats

def predict_2026_match_data(model, player_stats, h2h_stats, elo_ratings, elo_surface, last_match_date, pressure_stats, last_rankings, p1_name, p2_name, match_surface='Hard', tourney_country_code='UNK'):
    """Versione che ritorna dati grezzi per la GUI"""
    
    p1_info = last_rankings.get(p1_name.lower())
    p2_info = last_rankings.get(p2_name.lower())
    
    if not p1_info or not p2_info:
        return None

    # Helper: Get Stats from history (reconstructed in last_known)
    def get_stats(info):
        # Default empty
        matches = info.get('matches', [])
        # Form
        form = sum([m['pts'] for m in matches[-10:]]) / 10.0 if matches else 0.0
        # Surface Form (Need to fetch from surface_history if available in info, otherwise mock)
        surf_hist = info.get('surface_history', {}).get(match_surface, [])
        surf_form = sum(surf_hist[-5:]) / len(surf_hist[-5:]) if surf_hist else 0.0
        # Quality
        qual_hist = info.get('quality_history', [])
        qual = np.mean(qual_hist[-10:]) if qual_hist else 0.0
        # Serve Rating
        serv_hist = info.get('serve_history', [])
        serv = np.mean(serv_hist[-10:]) if serv_hist else 0.5
        # Pressure
        vals = [m['bp_saved_pct'] for m in matches if 'bp_saved_pct' in m]
        bp = np.mean(vals) if vals else 0.5
        vals = [m['decider_win'] for m in matches if 'decider_win' in m]
        dec = np.mean(vals) if vals else 0.5
        vals = [m['tb_win'] for m in matches if 'tb_win' in m]
        tb = np.mean(vals) if vals else 0.5
        
        return form, surf_form, qual, serv, bp, dec, tb
        
    p1_form, p1_sform, p1_qual, p1_serv, p1_bp, p1_dec, p1_tb = get_stats(p1_info)
    p2_form, p2_sform, p2_qual, p2_serv, p2_bp, p2_dec, p2_tb = get_stats(p2_info)
    
    # Elo
    p1_elo = p1_info['elo']
    p2_elo = p2_info['elo']
    p1_elo_s = p1_info['elo_surface'].get(match_surface, 1500)
    p2_elo_s = p2_info['elo_surface'].get(match_surface, 1500)
    
    # Days (Assume 3-4 days if tournament ongoing, simplified)
    p1_days = 4
    p2_days = 4
    
    # Features Differentials
    # Home Adv
    p1_ioc = p1_info.get('ioc', 'UNK')
    p2_ioc = p2_info.get('ioc', 'UNK')
    is_home1 = 1 if p1_ioc == tourney_country_code and tourney_country_code != 'UNK' else 0
    is_home2 = 1 if p2_ioc == tourney_country_code and tourney_country_code != 'UNK' else 0
    
    # Big Server & Lefty
    big_server1 = 1 if p1_info['ht'] > 195 else 0
    big_server2 = 1 if p2_info['ht'] > 195 else 0
    
    hand1 = p1_info['hand']
    hand2 = p2_info['hand']
    is_lefty = 1 if (hand1 == 'L' and hand2 == 'R') or (hand1 == 'R' and hand2 == 'L') else 0
    
    features = pd.DataFrame([{
        'Diff_Rank': np.log(p2_info['rank'] + 1) - np.log(p1_info['rank'] + 1),
        'Diff_Elo': p1_elo - p2_elo,
        'Diff_Elo_Surface': p1_elo_s - p2_elo_s,
        'Diff_Form': p1_form - p2_form,
        'Diff_Form_Surface': p1_sform - p2_sform,
        'Diff_Quality': p1_qual - p2_qual,
        'Diff_Serve_Rating': p1_serv - p2_serv,
        'Diff_Big_Server': big_server1 - big_server2,
        'Diff_Height': p1_info['ht'] - p2_info['ht'],
        'Diff_Age': p1_info['age'] - p2_info['age'],
        'Diff_Days': 0, 
        'Diff_BP_Save': p1_bp - p2_bp,
        'Diff_TB_Win': p1_tb - p2_tb,
        'Diff_Decider': p1_dec - p2_dec,
        'Diff_Home': is_home1 - is_home2,
        'Is_Lefty': is_lefty
    }])
    
    # 4. Predizione
    prob = model.predict_proba(features)[0][1] # Prob che P1 vinca
    
    return {
        'p1_name': p1_info['name'],
        'p2_name': p2_info['name'],
        'prob_p1': prob,
        'stats_p1': {'rank': p1_info['rank'], 'elo': p1_elo, 'elo_surface': p1_elo_s, 'form': p1_form, 'bp_save': p1_bp, 'decider': p1_dec},
        'stats_p2': {'rank': p2_info['rank'], 'elo': p2_elo, 'elo_surface': p2_elo_s, 'form': p2_form, 'bp_save': p2_bp, 'decider': p2_dec},
        'h2h': "N/A" # Simplified
    }

def predict_2026_match(model, player_stats, h2h_stats, elo_ratings, elo_surface, last_match_date, pressure_stats, last_rankings, p1_name, p2_name):
    """Wrapper per compatibilità con il vecchio main"""
    data = predict_2026_match_data(model, player_stats, h2h_stats, elo_ratings, elo_surface, last_match_date, pressure_stats, last_rankings, p1_name, p2_name)
    if data:
        winner = data['p1_name'] if data['prob_p1'] > 0.5 else data['p2_name']
        conf = data['prob_p1'] if data['prob_p1'] > 0.5 else 1 - data['prob_p1']
        print(f"Vincitore Previsto: {winner} ({conf*100:.1f}%)")
        print(f"Dettagli: Rank {data['stats_p1']['rank']}vs{data['stats_p2']['rank']}, Elo {int(data['stats_p1']['elo'])}vs{int(data['stats_p2']['elo'])}")

def build_model():
    """funzione principale per allenare il modello e ritornare gli oggetti necessari alla GUI"""
    # 1. Caricamento
    df_raw = load_data()
    print(f"Totale match caricati: {len(df_raw)}")
    
    # Elaborazione
    df_proc, history, surface_history, quality_history, serve_history, h2h_stats = process_data(df_raw, tourney_country_map=TOURNEY_COUNTRY_MAP)
    
    # ... Training (Standard XGBoost) ...
    # Salvo stato giocatori per previsioni future
    last_known = {}
    for p, info in history.items():
        if len(info['matches']) > 0:
            last_known[p.lower()] = {
                'elo': info['elo'],
                'elo_surface': info['elo_surface'],
                'matches': info['matches'][-20:], # Keep more history for stats imputation
                'surface_history': surface_history.get(p, {}), 
                'quality_history': quality_history.get(p, []),
                'serve_history': serve_history.get(p, []),
                'rank': 100, # Fallback
                'name': p,
                'ht': 185, # Fallback
                'hand': 'R',
                'age': 25,
                'ioc': 'UNK' # Default
            }

    # Populate Metadata from last appearance in DF
    print("Updating player metadata...")
    df_sorted = df_raw.sort_values('tourney_date', ascending=True) 
    meta_dict = {}
    for idx, row in df_sorted.iterrows():
        if pd.notna(row['winner_name']):
            n = str(row['winner_name']).lower()
            if n not in meta_dict: meta_dict[n] = {}
            meta_dict[n] = {'rank': row['winner_rank'], 'ht': row['winner_ht'], 'hand': row['winner_hand'], 'age': row['winner_age'], 'ioc': row['winner_ioc']}
        if pd.notna(row['loser_name']):
            n = str(row['loser_name']).lower()
            if n not in meta_dict: meta_dict[n] = {}
            meta_dict[n] = {'rank': row['loser_rank'], 'ht': row['loser_ht'], 'hand': row['loser_hand'], 'age': row['loser_age'], 'ioc': row['loser_ioc']}
            
    # Apply metadata
    for p_name, p_data in last_known.items():
        clean_name = str(p_name).lower()
        if clean_name in meta_dict:
            m = meta_dict[clean_name]
            p_data['rank'] = m['rank']
            p_data['ht'] = m['ht'] if pd.notna(m['ht']) else 185
            p_data['hand'] = m['hand'] if pd.notna(m['hand']) else 'R'
            p_data['age'] = m['age'] if pd.notna(m['age']) else 25
            p_data['ioc'] = str(m['ioc']).upper() if pd.notna(m['ioc']) else 'UNK'

    # Filter NaN
    df_train_full = df_proc.dropna()
    print(f"Dataset finale pronto: {len(df_train_full)} match.")
    
    # Split
    df_train_full = df_train_full.sort_values('Date')
    split_date = df_train_full['Date'].max() - pd.Timedelta(days=365)
    train = df_train_full[df_train_full['Date'] < split_date]
    test = df_train_full[df_train_full['Date'] >= split_date]
    
    # Add Diff_Home here!
    features_cols = [
        'Diff_Rank', 'Diff_Elo', 'Diff_Elo_Surface', 'Diff_Form', 
        'Diff_Form_Surface', 'Diff_Quality', 'Diff_Serve_Rating',
        'Diff_Big_Server', 'Diff_Height', 'Diff_Age', 'Diff_Days',
        'Diff_BP_Save', 'Diff_TB_Win', 'Diff_Decider', 'Diff_Home', 'Is_Lefty'
    ]
    
    X_train = train[features_cols]
    y_train = train['Target']
    X_test = test[features_cols]
    y_test = test['Target']
    
    # Train
    print("\n--- Training XGBoost (Optimized) ---")
    model = xgb.XGBClassifier(
        n_estimators=500, 
        learning_rate=0.01, 
        max_depth=5, 
        subsample=0.6, 
        colsample_bytree=0.7, 
        gamma=0.5,
        reg_lambda=1,
        reg_alpha=0,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Eval
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuratezza Modello: {acc*100:.2f}%")
    
    # Feature Importance
    print("\nFeature Importance:")
    print(pd.Series(model.feature_importances_, index=features_cols).sort_values(ascending=False))
    
    # Return compatible signature for tennis_app.py
    # model, stats, h2h, elo, elo_surf, dates, pressure, players
    return model, history, h2h_stats, {}, {}, {}, {}, last_known

if __name__ == "__main__":
    model, stats, h2h, elo, elo_s, date, press, last = build_model()
    predict_2026_match(model, stats, h2h, elo, elo_s, date, press, last, "Jannik Sinner", "Carlos Alcaraz")
    predict_2026_match(model, stats, h2h, elo, elo_s, date, press, last, "Novak Djokovic", "Jannik Sinner")