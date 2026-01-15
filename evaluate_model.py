import tennis_bot
import pandas as pd
import re

# Raw data from Flashscore (Jan 14-15, 2026)
raw_matches_text = """
15.01.2026 | Winner: Svajda Z. | Loser: Coppejans K.
15.01.2026 | Winner: Kubler J. | Loser: Blockx A.
15.01.2026 | Winner: Zheng M. | Loser: Klein L.
15.01.2026 | Winner: Gea A. | Loser: Vallejo D.
15.01.2026 | Winner: Damm M. | Loser: Bailly G. A.
15.01.2026 | Winner: Draxl L. | Loser: McDonald M.
15.01.2026 | Winner: Maestrelli F. | Loser: Lajovic D.
15.01.2026 | Winner: Ymer E. | Loser: Wong C.
15.01.2026 | Winner: Jodar R. | Loser: van Assche L.
15.01.2026 | Winner: Wu Y. | Loser: Boyer T.
15.01.2026 | Winner: Budkov Kjaer N. | Loser: Herbert P.
15.01.2026 | Winner: Basavareddy N. | Loser: Loffhagen G.
15.01.2026 | Winner: Fery A. | Loser: Prizmic D.
15.01.2026 | Winner: Faria J. | Loser: Trungelliti M.
15.01.2026 | Winner: Sweeny D. | Loser: Travaglia S.
15.01.2026 | Winner: Sakamoto R. | Loser: Zeppieri G.
14.01.2026 | Winner: Draxl L. | Loser: Sachko V.
14.01.2026 | Winner: Kubler J. | Loser: Gaubas V.
14.01.2026 | Winner: Svajda Z. | Loser: Rodionov J.
14.01.2026 | Winner: Gea A. | Loser: Burruchaga R. A.
14.01.2026 | Winner: Klein L. | Loser: Merida Aguilar D.
14.01.2026 | Winner: McDonald M. | Loser: Grenier H.
14.01.2026 | Winner: Vallejo D. | Loser: Cina F.
14.01.2026 | Winner: Blockx A. | Loser: Molcan A.
14.01.2026 | Winner: Zheng M. | Loser: Barrios Vera T.
14.01.2026 | Winner: van Assche L. | Loser: Glinka D.
14.01.2026 | Winner: Lajovic D. | Loser: Cassone M.
14.01.2026 | Winner: Coppejans K. | Loser: Riedi L.
14.01.2026 | Winner: Boyer T. | Loser: Passaro F.
14.01.2026 | Winner: Ymer E. | Loser: Moller E.
14.01.2026 | Winner: Jodar R. | Loser: Rodesch C.
14.01.2026 | Winner: Fery A. | Loser: Tomic B.
14.01.2026 | Winner: Wu Y. | Loser: Mejia N.
14.01.2026 | Winner: Maestrelli F. | Loser: Seyboth Wild T.
14.01.2026 | Winner: Wong C. | Loser: Llamas Ruiz P.
14.01.2026 | Winner: Budkov Kjaer N. | Loser: McCabe J.
14.01.2026 | Winner: Travaglia S. | Loser: Landaluce M.
14.01.2026 | Winner: Prizmic D. | Loser: Heide G.
14.01.2026 | Winner: Trungelliti M. | Loser: Rocha H.
14.01.2026 | Winner: Herbert P. | Loser: Bueno G.
14.01.2026 | Winner: Faria J. | Loser: Hassan B.
14.01.2026 | Winner: Sweeny D. | Loser: Kym J.
14.01.2026 | Winner: Damm M. | Loser: Droguet T.
14.01.2026 | Winner: Loffhagen G. | Loser: Nishioka Y.
14.01.2026 | Winner: Sakamoto R. | Loser: Smith C.
14.01.2026 | Winner: Basavareddy N. | Loser: Ofner S.
14.01.2026 | Winner: Bailly G. A. | Loser: Carballes Baena R.
14.01.2026 | Winner: Zeppieri G. | Loser: Holt B.
"""

def parse_flashscore_name(name_str):
    """
    Input: "Svajda Z." or "Bailly G. A."
    Output: {'last': 'Svajda', 'initials': ['Z']}
    """
    parts = name_str.strip().split()
    if not parts: return None
    
    # Usually Format is "Lastname I." or "Lastname I. I."
    # We assume the last part(s) ending in dot are initials
    # But Flashscore puts initials AT THE END usually? Yes "Svajda Z."
    
    # Let's find where the initials start.
    # Usually initials are 1 letter + dot.
    
    initials = []
    lastname_parts = []
    
    for p in parts:
        if p.endswith('.') and len(p) <= 3: # "Z." or "A."
            initials.append(p.replace('.', ''))
        else:
            lastname_parts.append(p)
            
    return {
        'last': " ".join(lastname_parts).lower(),
        'initials': [i.lower() for i in initials]
    }

def find_player_full_name(fs_name_str, all_player_names):
    """
    Tries to find the best match in all_player_names (list of full names)
    for the Flashscore name "Svajda Z."
    """
    parsed = parse_flashscore_name(fs_name_str)
    if not parsed: return None
    
    target_last = parsed['last']
    target_initials = parsed['initials']
    
    candidates = []
    
    for full_name in all_player_names:
        fn_lower = str(full_name).lower()
        # simplified check: does the full name contain the target last name?
        # And does the first name start with the target initial?
        
        # Split fullname
        # Usually "firstname lastname" in tennis_bot DB
        # But data could be mixed. Let's assume standard "First Last"
        
        if target_last in fn_lower:
            # Check initials
            # Split full name
            fn_parts = fn_lower.split()
            # Assumption: Last name is at the end, First name at start
            # But "van Assche" -> parts: van, assche
            
            # Check if any part starts with the initial
            if not target_initials:
                 candidates.append(full_name)
                 continue
                 
            match_initial = False
            # Check the first part(s) that are NOT the last name
            # This is heuristics.
            
            # Better: Check if any word starts with the initial
            for part in fn_parts:
                if part.startswith(target_initials[0]):
                    match_initial = True
                    break
            
            if match_initial:
                candidates.append(full_name)
    
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Ambiguity. Try to resolve?
        # E.g. "Ymer E." might match "Elias Ymer" (Mikael Ymer starts with M)
        # prefer exact match of surname at end of string?
        # For now return first
        # Sort by length similarity?
        return candidates[0]
            
    return None

def main():
    print("Building model...")
    model, stats, h2h, elo, elo_s, date, press, last_known = tennis_bot.build_model()
    
    all_known_names = list(last_known.keys()) # These are lowercased in build_model logic? 
    # Actually last_known keys are whatever was put in.
    # In build_model: last_known[p.lower()] = ... so keys are lower.
    # But inside the dict 'name' is preserved.
    
    real_names_map = {k: v['name'] for k, v in last_known.items()}
    all_names_real = list(real_names_map.values())
    
    print(f"\nParsing {len(raw_matches_text.strip().splitlines())} matches...")
    
    parse_success = 0
    matches_to_eval = []
    
    for line in raw_matches_text.strip().splitlines():
        if "|" not in line: continue
        # Format: 15.01.2026 | Winner: Svajda Z. | Loser: Coppejans K.
        parts = line.split("|")
        date_str = parts[0].strip()
        winner_raw = parts[1].split("Winner:")[1].strip()
        loser_raw = parts[2].split("Loser:")[1].strip()
        
        # Resolve names
        w_full = find_player_full_name(winner_raw, all_names_real)
        l_full = find_player_full_name(loser_raw, all_names_real)
        
        if w_full and l_full:
            matches_to_eval.append({
                'date': date_str,
                'p1': w_full,
                'p2': l_full,
                'actual_winner': w_full,
                'raw': line
            })
            parse_success += 1
        else:
            print(f"Mapping failed for: {winner_raw} ({w_full}) vs {loser_raw} ({l_full})")

    print(f"\nSuccessfully mapped {len(matches_to_eval)} matches.")
    
    correct = 0
    total = 0
    results = []
    
    for m in matches_to_eval:
        p1 = m['p1']
        p2 = m['p2']
        
        # Predict
        prediction_data = tennis_bot.predict_2026_match_data(
            model, stats, h2h, elo, elo_s, date, press, last_known, p1, p2, match_surface='Hard', tourney_country_code='AUS'
        )
        
        if not prediction_data:
            print(f"Skipping prediction for {p1} vs {p2} (Data issue)")
            continue
            
        prob_p1 = prediction_data['prob_p1']
        predicted_winner = p1 if prob_p1 > 0.5 else p2
        confidence = prob_p1 if prob_p1 > 0.5 else 1 - prob_p1
        
        is_correct = (predicted_winner == m['actual_winner']) # Names should match as we used resolved names
        if is_correct: correct += 1
        total += 1
        
        results.append({
            'Match': f"{p1} vs {p2}",
            'Actual': m['actual_winner'],
            'Predicted': predicted_winner,
            'Confidence': f"{confidence:.2%}",
            'Correct': "YES" if is_correct else "NO"
        })
        
    print("\n--- RESULTS ---")
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        print(df_res.to_string(index=False))
        
    if total > 0:
        print(f"\nAccuracy: {correct/total:.2%} ({correct}/{total})")

if __name__ == "__main__":
    main()
