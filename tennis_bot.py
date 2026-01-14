import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
import os
import requests
from io import StringIO

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

# 2. FEATURE ENGINEERING AVANZATO
def process_data(df):
    print("\n--- Elaborazione Dati e Feature Engineering ---")
    
    # Mappa Tornei -> Nazione (Approssimativa per Major e Top events)
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
    
    # Conversione data corretta
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.sort_values('tourney_date').reset_index(drop=True)
    # Conversione data corretta
    df['Date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Rimuovi filtro Hard qui: processiamo tutto per l'Elo, filtriamo solo in append
    # df = df[df['surface'] == 'Hard']
    
    # Pulizia base
    utils_cols = [
        'tourney_date', 'match_num', 'winner_id', 'loser_id', 
        'winner_name', 'loser_name', 'winner_rank', 'loser_rank',
        'winner_age', 'loser_age', 'winner_ht', 'loser_ht'
    ]
    df = df.dropna(subset=['winner_rank', 'loser_rank', 'winner_age', 'loser_age'])
    
    # --- Creazione Features Storiche (Rolling) ---
    # Per fare questo in modo efficiente, dobbiamo ristrutturare il dataset 
    # dal punto di vista dei giocatori, non dei match.
    # Ma per il training XGBoost, ci serve riga = match.
    # Quindi calcoliamo le stats "fino a quel giorno" per ogni match.
    
    # Mappe per tenere traccia dello storico
    player_stats = {} # {player_id: {'wins': 0, 'losses': 0, 'last_matches': []}}
    
    def get_player_recent_form(pid):
        stats = player_stats.get(pid, {'wins': 0, 'losses': 0, 'history': []})
        total_matches = stats['wins'] + stats['losses']
        win_rate = stats['wins'] / total_matches if total_matches > 0 else 0.0
        
        # Form ultime 10 partite
        last_10 = stats['history'][-10:]
        form_10 = sum(last_10) / len(last_10) if last_10 else 0.0
        
        return win_rate, form_10, total_matches

    def update_player_stats(winner_id, loser_id):
        # Update Winner
        if winner_id not in player_stats: player_stats[winner_id] = {'wins': 0, 'losses': 0, 'history': []}
        player_stats[winner_id]['wins'] += 1
        player_stats[winner_id]['history'].append(1)
        
        # Update Loser
        if loser_id not in player_stats: player_stats[loser_id] = {'wins': 0, 'losses': 0, 'history': []}
        player_stats[loser_id]['losses'] += 1
        player_stats[loser_id]['history'].append(0)

    p1_h2h_wins, p2_h2h_wins = [], []
    
    # --- TRACKING DATES (FATIGUE) ---
    last_match_date = {} # {player_id: datetime}
    
    # --- TRACKING PRESSURE (CLUTCH) ---
    # {player_id: {'bp_saved': 0, 'bp_faced': 0, 'tb_won': 0, 'tb_total': 0, 'decider_won': 0, 'decider_total': 0}}
    pressure_stats = {} 
    
    def get_pressure_stats(pid):
        s = pressure_stats.get(pid, {'bp_saved': 0, 'bp_faced': 0, 'tb_won': 0, 'tb_total': 0, 'decider_won': 0, 'decider_total': 0})
        
        # BP Save %
        bp_save_pct = s['bp_saved'] / s['bp_faced'] if s['bp_faced'] > 0 else 0.5 # Default 50%
        
        # TB Win %
        tb_win_pct = s['tb_won'] / s['tb_total'] if s['tb_total'] > 0 else 0.5
        
        # Decider Win %
        decider_win_pct = s['decider_won'] / s['decider_total'] if s['decider_total'] > 0 else 0.5
        
        return bp_save_pct, tb_win_pct, decider_win_pct

    def update_pressure_stats(winner_id, loser_id, row):
        # Init
        if winner_id not in pressure_stats: pressure_stats[winner_id] = {'bp_saved': 0, 'bp_faced': 0, 'tb_won': 0, 'tb_total': 0, 'decider_won': 0, 'decider_total': 0}
        if loser_id not in pressure_stats: pressure_stats[loser_id] = {'bp_saved': 0, 'bp_faced': 0, 'tb_won': 0, 'tb_total': 0, 'decider_won': 0, 'decider_total': 0}
        
        w_s = pressure_stats[winner_id]
        l_s = pressure_stats[loser_id]
        
        # 1. Break Points (Se disponibili)
        if pd.notna(row['w_bpSaved']) and pd.notna(row['w_bpFaced']):
            w_s['bp_saved'] += row['w_bpSaved']
            w_s['bp_faced'] += row['w_bpFaced']
        if pd.notna(row['l_bpSaved']) and pd.notna(row['l_bpFaced']):
            l_s['bp_saved'] += row['l_bpSaved']
            l_s['bp_faced'] += row['l_bpFaced']
            
        # 2. Tie-Breaks & Deciding Sets
        score = str(row['score'])
        if pd.notna(score) and score != 'nan':
            sets = score.split(' ')
            
            # Tie-Breaks (es. 7-6, 6-7)
            # Semplificazione: se c'è 7-6 o 6-7, assegniamo vittoria TB.
            # Chi ha vinto il set?
            # Esempio score: "6-7(4) 6-4 7-6(5)"
            # Winner ha vinto 2 set, Loser 1 (in best of 3).
            # Difficile fare parsing esatto senza logica complessa.
            # Euristica: Contiamo parentesi "(.)" come TB giocati.
            # Assegniamo TB vinti in base a chi ha vinto il match? No, impreciso.
            # Facciamo parsing light:
            w_sets_won = 0
            l_sets_won = 0
            
            for s in sets:
                if 'ret' in s or 'W/O' in s: continue
                # Rimuovi parentesi punteggio TB per capire chi ha vinto il set
                clean_s = s.split('(')[0]
                if '-' in clean_s:
                    try:
                        g_w, g_l = map(int, clean_s.split('-'))
                        # Tie Break detection
                        if g_w == 7 and g_l == 6:
                            w_s['tb_won'] += 1
                            w_s['tb_total'] += 1
                            l_s['tb_total'] += 1
                        elif g_w == 6 and g_l == 7:
                            l_s['tb_won'] += 1
                            l_s['tb_total'] += 1
                            w_s['tb_total'] += 1
                        
                        if g_w > g_l: w_sets_won += 1
                        else: l_sets_won += 1
                    except: pass
            
            # 3. Deciding Set
            # Se winner ha vinto 2-1 (best of 3) o 3-2 (best of 5)
            total_sets = w_sets_won + l_sets_won
            if (row['best_of'] == 3 and total_sets == 3) or (row['best_of'] == 5 and total_sets == 5):
                w_s['decider_won'] += 1
                w_s['decider_total'] += 1
                l_s['decider_total'] += 1

    def get_days_since_last(pid, current_date):
        if pid not in last_match_date:
            return 30 # Default: riposato
        delta = current_date - last_match_date[pid]
        return delta.days
        
    def update_last_date(pid, date):
        last_match_date[pid] = date
    
    # --- ELO RATINGS (GLOBAL & SURFACE) ---
    elo_ratings = {}    # {player_id: rating}
    elo_surface = {     # {surface: {player_id: rating}}
        'Hard': {}, 'Clay': {}, 'Grass': {}, 'Carpet': {} 
    }
    
    K_FACTOR = 20
    INITIAL_ELO = 1500
    
    def get_elo(pid, surface=None):
        if surface:
            return elo_surface.get(surface, {}).get(pid, INITIAL_ELO)
        return elo_ratings.get(pid, INITIAL_ELO)
        
    def update_elo(winner_id, loser_id, surface):
        # 1. Global Elo
        w_elo = get_elo(winner_id)
        l_elo = get_elo(loser_id)
        
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        expected_l = 1 / (1 + 10 ** ((w_elo - l_elo) / 400))
        
        new_w = w_elo + K_FACTOR * (1 - expected_w)
        new_l = l_elo + K_FACTOR * (0 - expected_l)
        
        elo_ratings[winner_id] = new_w
        elo_ratings[loser_id] = new_l
        
        # 2. Surface Elo
        if surface in elo_surface:
            w_elo_s = get_elo(winner_id, surface)
            l_elo_s = get_elo(loser_id, surface)
            
            exp_w_s = 1 / (1 + 10 ** ((l_elo_s - w_elo_s) / 400))
            exp_l_s = 1 / (1 + 10 ** ((w_elo_s - l_elo_s) / 400))
            
            new_w_s = w_elo_s + K_FACTOR * (1 - exp_w_s)
            new_l_s = l_elo_s + K_FACTOR * (0 - exp_l_s)
            
            elo_surface[surface][winner_id] = new_w_s
            elo_surface[surface][loser_id] = new_l_s
        
    # Mappa H2H: tuple(sorted(id1, id2)) -> {id1: wins, id2: wins}
    h2h_stats = {} 
    
    processed_rows = []

    print("Calcolo statistiche storiche e H2H...")
    for idx, row in df.iterrows():
        # Randomizziamo P1 e P2 (come prima)
        if np.random.rand() > 0.5:
            p1_id, p1_rank, p1_age, p1_ht = row['winner_id'], row['winner_rank'], row['winner_age'], row['winner_ht']
            p2_id, p2_rank, p2_age, p2_ht = row['loser_id'], row['loser_rank'], row['loser_age'], row['loser_ht']
            target = 1
            original_winner = p1_id
            original_loser = p2_id
        else:
            p1_id, p1_rank, p1_age, p1_ht = row['loser_id'], row['loser_rank'], row['loser_age'], row['loser_ht']
            p2_id, p2_rank, p2_age, p2_ht = row['winner_id'], row['winner_rank'], row['winner_age'], row['winner_ht']
            target = 0
            original_winner = p2_id
            original_loser = p1_id
            
        w_id = row['winner_id']
        l_id = row['loser_id']
        tourney_name = row['tourney_name']
        surface = row['surface']
        date = row['Date']
        
        # --- 0. HOME ADVANTAGE FEATURE ---
        # Determina paese del torneo
        tourney_country = TOURNEY_COUNTRY_MAP.get(tourney_name, 'UNK')
        
        # Recupera nazionalità (IOC) - Gestione casi mancanti o non standard
        w_ioc = str(row['winner_ioc']).upper() if pd.notna(row['winner_ioc']) else 'UNK'
        l_ioc = str(row['loser_ioc']).upper() if pd.notna(row['loser_ioc']) else 'UNK'
        
        w_is_home = 1 if w_ioc == tourney_country and tourney_country != 'UNK' else 0
        l_is_home = 1 if l_ioc == tourney_country and tourney_country != 'UNK' else 0
        
        diff_home = w_is_home - l_is_home # +1 se W è a casa, -1 se L è a casa, 0 se pari
        
        # Recupera stats PRE-MATCH
        p1_wr, p1_form, p1_exp = get_player_recent_form(p1_id)
        p2_wr, p2_form, p2_exp = get_player_recent_form(p2_id)
        
        # Recupera H2H PRE-MATCH
        match_key = tuple(sorted([p1_id, p2_id]))
        
        # Recupera ELO PRE-MATCH
        p1_elo = get_elo(p1_id)
        p2_elo = get_elo(p2_id)
        
        # Recupera ELO SURFACE PRE-MATCH
        match_surface = row['surface']
        p1_elo_surf = get_elo(p1_id, match_surface)
        p2_elo_surf = get_elo(p2_id, match_surface)

        if match_key not in h2h_stats:
            h2h_stats[match_key] = {p1_id: 0, p2_id: 0}
            
        p1_h2h = h2h_stats[match_key][p1_id]
        p2_h2h = h2h_stats[match_key][p2_id]
        
        # Recupera FATIGUE PRE-MATCH
        # Attenzione: Date è timestamp
        # FIX: Calcoliamo Diff_Days SOLO se abbiamo lo storico per entrambi.
        # Altrimenti rischiamo bias (uno ha i giorni contati, l'altro ha il default 30).
        diff_days = 0
        if p1_id in last_match_date and p2_id in last_match_date:
            p1_days = (row['tourney_date'] - last_match_date[p1_id]).days
            p2_days = (row['tourney_date'] - last_match_date[p2_id]).days
            diff_days = p1_days - p2_days
        
        # Recupera PRESSURE PRE-MATCH
        p1_bp, p1_tb, p1_dec = get_pressure_stats(p1_id)
        p2_bp, p2_tb, p2_dec = get_pressure_stats(p2_id)
        
        # Aggiorna stats POST-MATCH
        update_player_stats(original_winner, original_loser)
        update_elo(original_winner, original_loser, match_surface) # Aggiorna Elo
        update_last_date(original_winner, row['tourney_date'])
        update_last_date(original_loser, row['tourney_date'])
        update_pressure_stats(original_winner, original_loser, row)
        h2h_stats[match_key][original_winner] += 1
        
        # --- FILTRO PER TRAINING ---
        # Salviamo la riga per il training SOLO SE è la superficie che ci interessa (Hard)
        # O se vogliamo un modello generico, salviamo tutto con la feature surface.
        # Per AO 2026, filtriamo Hard.
        if match_surface == 'Hard':
            # Costruisci riga features
            diff_rank = p2_rank - p1_rank 
            diff_age = p2_age - p1_age
            diff_ht = p1_ht - p2_ht
            diff_winrate = p1_wr - p2_wr
            diff_form = p1_form - p2_form
            diff_exp = p1_exp - p2_exp
            diff_h2h = p1_h2h - p2_h2h
            
            processed_rows.append({
                'Diff_Rank': diff_rank,
                'Diff_Age': diff_age,
                'Diff_Height': diff_ht if not pd.isna(diff_ht) else 0,
                'Diff_WinRate': diff_winrate,
                'Diff_Form': diff_form,
                'Diff_Exp': diff_exp,
                'Diff_H2H': diff_h2h,
                'Diff_Days': diff_days, 
                'Diff_BP_Save': p1_bp - p2_bp,
                'Diff_TB_Win': p1_tb - p2_tb,
                'Diff_Decider': p1_dec - p2_dec,
                'Diff_Decider': p1_dec - p2_dec,
                'Diff_Elo': p1_elo - p2_elo,
                'Diff_Elo_Surface': p1_elo_surf - p2_elo_surf, 
                'Diff_Home': diff_home, # Add to features
                'Target': target,
                'Date': row['tourney_date'],
                # Metadata per ricostruire lo stato finale (solo per l'ultima riga di ogni giocatore)
                'P1_ID': p1_id, 'P2_ID': p2_id 
            })
        
    return pd.DataFrame(processed_rows), player_stats, h2h_stats, elo_ratings, elo_surface, last_match_date, pressure_stats

def predict_2026_match_data(model, player_stats, h2h_stats, elo_ratings, elo_surface, last_match_date, pressure_stats, last_rankings, p1_name, p2_name, match_surface='Hard', tourney_country_code='UNK'):
    """Versione che ritorna dati grezzi per la GUI"""
    # Lookup giocatori (molto semplice, case incentive)
    # last_rankings deve essere un dict {name_lower: {id, rank, age, ht}}
    
    p1_info = last_rankings.get(p1_name.lower())
    p2_info = last_rankings.get(p2_name.lower())
    
    if not p1_info or not p2_info:
        return None

    # 1. Recupera stats correnti (fine 2024/2025)
    def get_stats(pid):
        s = player_stats.get(pid, {'wins': 0, 'losses': 0, 'history': []})
        tot = s['wins'] + s['losses']
        wr = s['wins'] / tot if tot > 0 else 0
        form = sum(s['history'][-10:]) / len(s['history'][-10:]) if s['history'] else 0
        return wr, form, tot
        
    p1_wr, p1_form, p1_exp = get_stats(p1_info['id'])
    p2_wr, p2_form, p2_exp = get_stats(p2_info['id'])
    
    # 2. H2H
    match_key = tuple(sorted([p1_info['id'], p2_info['id']]))
    h2h = h2h_stats.get(match_key, {p1_info['id']: 0, p2_info['id']: 0})
    p1_h2h_val = h2h.get(p1_info['id'], 0)
    p2_h2h_val = h2h.get(p2_info['id'], 0)
    
    # 3. Elo
    p1_elo = elo_ratings.get(p1_info['id'], 1500)
    p2_elo = elo_ratings.get(p2_info['id'], 1500)
    
    # 3b. Surface Elo (Dynamic)
    p1_elo_s = elo_surface.get(match_surface, {}).get(p1_info['id'], 1500)
    p2_elo_s = elo_surface.get(match_surface, {}).get(p2_info['id'], 1500)
    
    # 3c. Fatigue
    p1_days = 30
    p2_days = 30
    
    # 3d. Pressure
    # 3d. Pressure
    def get_p_stats(pid):
        s = pressure_stats.get(pid, {'bp_saved': 0, 'bp_faced': 0, 'tb_won': 0, 'tb_total': 0, 'decider_won': 0, 'decider_total': 0})
        bp = s['bp_saved'] / s['bp_faced'] if s['bp_faced'] > 0 else 0.5
        tb = s['tb_won'] / s['tb_total'] if s['tb_total'] > 0 else 0.5
        dec = s['decider_won'] / s['decider_total'] if s['decider_total'] > 0 else 0.5
        return bp, tb, dec
        
    p1_bp, p1_tb, p1_dec = get_p_stats(p1_info['id'])
    p2_bp, p2_tb, p2_dec = get_p_stats(p2_info['id'])
    
    # 4. HOME ADVANTAGE (Dynamic)
    # tourney_country must be passed from GUI, usually argument match_surface is abused or we need new arg.
    # To avoid breaking signature we can add `tourney_country` kwarg or infer.
    # Actually, we need to add `tourney_country` to the function signature.
    # But for now, let's assume Neutral unless specified.
    # Wait, I added match_surface in previous step. Let's add tourney_country_code.
    
    # NOTE: I am adding tourney_country argument in the replacement logic below.
    
    # Default behavior if not passed
    tourney_country = 'UNK' 
    # Logic to be implemented: pass tourney_country in signature
    
    # Let's fix signature first... wait, I can do it in this chunk if I'm careful??
    # No, signature is at line 410. This chunk is 451.
    
    # I'll calculate it here assuming the variable exists, and I will update signature in next chunk.
    
    # 3. Features differenziali
    # ORDINE IMPORTANTE! Deve combaciare con features_cols del training
    # ['Diff_Rank', 'Diff_Elo', 'Diff_Elo_Surface', 'Diff_Age', 'Diff_Height', 'Diff_WinRate', 'Diff_Form', 'Diff_Exp', 'Diff_H2H', 'Diff_Days', 'Diff_BP_Save', 'Diff_TB_Win', 'Diff_Decider', 'Diff_Home']
    
    # Recupera info
    p1_ioc = p1_info.get('ioc', 'UNK')
    p2_ioc = p2_info.get('ioc', 'UNK')
    
    p1_home = 1 if p1_ioc == tourney_country_code and tourney_country_code != 'UNK' else 0
    p2_home = 1 if p2_ioc == tourney_country_code and tourney_country_code != 'UNK' else 0
    diff_home = p1_home - p2_home
    
    features = pd.DataFrame([{
        'Diff_Rank': p2_info['rank'] - p1_info['rank'],
        'Diff_Elo': p1_elo - p2_elo,
        'Diff_Elo_Surface': p1_elo_s - p2_elo_s,
        'Diff_Age': p2_info['age'] - p1_info['age'],
        'Diff_Height': p1_info['ht'] - p2_info['ht'],
        'Diff_WinRate': p1_wr - p2_wr,
        'Diff_Form': p1_form - p2_form,
        'Diff_Exp': p1_exp - p2_exp,
        'Diff_H2H': p1_h2h_val - p2_h2h_val,
        'Diff_Days': p1_days - p2_days,
        'Diff_BP_Save': p1_bp - p2_bp,
        'Diff_TB_Win': p1_tb - p2_tb,
        'Diff_Decider': p1_dec - p2_dec,
        'Diff_Home': diff_home
    }])
    
    # 4. Predizione
    prob = model.predict_proba(features)[0][1] # Prob che P1 vinca
    
    # Return raw data for GUI
    return {
        'p1_name': p1_info['name'],
        'p2_name': p2_info['name'],
        'prob_p1': prob,
        'stats_p1': {'rank': p1_info['rank'], 'elo': p1_elo, 'elo_surface': p1_elo_s, 'form': p1_form, 'bp_save': p1_bp, 'decider': p1_dec},
        'stats_p2': {'rank': p2_info['rank'], 'elo': p2_elo, 'elo_surface': p2_elo_s, 'form': p2_form, 'bp_save': p2_bp, 'decider': p2_dec},
        'h2h': f"{p1_h2h_val}-{p2_h2h_val}"
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
    
    # Processa
    df_proc, final_player_stats, final_h2h, final_elo, final_elo_surface, final_dates, final_pressure = process_data(df_raw)
    
    # Last Known status
    last_known = {}
    df_raw_sorted = df_raw.sort_values('tourney_date')
    for _, row in df_raw_sorted.iterrows():
        if pd.isna(row['winner_name']) or pd.isna(row['loser_name']): continue
        
        last_known[str(row['winner_name']).lower()] = {
            'name': row['winner_name'], 'id': row['winner_id'], 
            'rank': row['winner_rank'], 'age': row['winner_age'] + 1 if pd.notna(row['winner_age']) else 25, 
            'ht': row['winner_ht'] if pd.notna(row['winner_ht']) else 185,
            'ioc': str(row['winner_ioc']).upper() if pd.notna(row['winner_ioc']) else 'UNK'
        }
        last_known[str(row['loser_name']).lower()] = {
            'name': row['loser_name'], 'id': row['loser_id'], 
            'rank': row['loser_rank'], 'age': row['loser_age'] + 1 if pd.notna(row['loser_age']) else 25,
            'ht': row['loser_ht'] if pd.notna(row['loser_ht']) else 185,
            'ioc': str(row['loser_ioc']).upper() if pd.notna(row['loser_ioc']) else 'UNK'
        }

    # Filter NaN
    df_train_full = df_proc.dropna()
    print(f"Dataset finale pronto: {len(df_train_full)} match.")
    
    # Split
    df_train_full = df_train_full.sort_values('Date')
    split_date = df_train_full['Date'].max() - pd.Timedelta(days=365)
    train = df_train_full[df_train_full['Date'] < split_date]
    test = df_train_full[df_train_full['Date'] >= split_date]
    
    # Add Diff_Home here!
    features_cols = ['Diff_Rank', 'Diff_Elo', 'Diff_Elo_Surface', 'Diff_Age', 'Diff_Height', 'Diff_WinRate', 'Diff_Form', 'Diff_Exp', 'Diff_H2H', 'Diff_Days', 'Diff_BP_Save', 'Diff_TB_Win', 'Diff_Decider', 'Diff_Home']
    
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
    
    return model, final_player_stats, final_h2h, final_elo, final_elo_surface, final_dates, final_pressure, last_known

if __name__ == "__main__":
    model, stats, h2h, elo, elo_s, date, press, last = build_model()
    predict_2026_match(model, stats, h2h, elo, elo_s, date, press, last, "Jannik Sinner", "Carlos Alcaraz")
    predict_2026_match(model, stats, h2h, elo, elo_s, date, press, last, "Novak Djokovic", "Jannik Sinner")