import requests
import json
import os
import datetime
from datetime import timedelta
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = "tennis_data_cache/weather_cache.json"

# --- 1. METADATA TORNEI ---
# Mappa nome torneo -> {type: 'Indoor/Outdoor', lat, lon}
# Coordinate approssimative delle città ospitanti
TOURNAMENT_META = {
    # Grand Slams
    'Australian Open': {'type': 'Outdoor', 'lat': -37.814, 'lon': 144.963}, # Melbourne
    'Roland Garros': {'type': 'Outdoor', 'lat': 48.847, 'lon': 2.246}, # Paris
    'Wimbledon': {'type': 'Outdoor', 'lat': 51.434, 'lon': -0.214}, # London
    'US Open': {'type': 'Outdoor', 'lat': 40.749, 'lon': -73.847}, # New York
    
    # Masters 1000
    'Indian Wells': {'type': 'Outdoor', 'lat': 33.720, 'lon': -116.360},
    'Miami': {'type': 'Outdoor', 'lat': 25.958, 'lon': -80.238}, # Miami Gardens
    'Monte Carlo': {'type': 'Outdoor', 'lat': 43.738, 'lon': 7.427},
    'Madrid': {'type': 'Outdoor', 'lat': 40.367, 'lon': -3.689},
    'Rome': {'type': 'Outdoor', 'lat': 41.936, 'lon': 12.454},
    'Canada': {'type': 'Outdoor', 'lat': 45.526, 'lon': -73.665}, # Montreal (alternate Toronto)
    'Montreal': {'type': 'Outdoor', 'lat': 45.526, 'lon': -73.665},
    'Toronto': {'type': 'Outdoor', 'lat': 43.771, 'lon': -79.513},
    'Cincinnati': {'type': 'Outdoor', 'lat': 39.351, 'lon': -84.269},
    'Shanghai': {'type': 'Outdoor', 'lat': 31.045, 'lon': 121.366},
    'Paris Masters': {'type': 'Indoor', 'lat': 48.840, 'lon': 2.378}, # Bercy
    'Paris': {'type': 'Indoor', 'lat': 48.840, 'lon': 2.378}, # Spesso chiamato solo 'Paris' nel dataset ATP per il Masters indoor
    
    # Finals
    'Tour Finals': {'type': 'Indoor', 'lat': 45.039, 'lon': 7.647}, # Turin (attuale)
    'Next Gen Finals': {'type': 'Indoor', 'lat': 21.614, 'lon': 39.110}, # Jeddah

    # Altri tornei comuni (Esempio)
    'Rotterdam': {'type': 'Indoor', 'lat': 51.878, 'lon': 4.475},
    'Rio de Janeiro': {'type': 'Outdoor', 'lat': -22.977, 'lon': -43.204},
    'Dubai': {'type': 'Outdoor', 'lat': 25.234, 'lon': 55.337},
    'Acapulco': {'type': 'Outdoor', 'lat': 16.786, 'lon': -99.805},
    'Vienna': {'type': 'Indoor', 'lat': 48.219, 'lon': 16.416},
    'Basel': {'type': 'Indoor', 'lat': 47.539, 'lon': 7.616},
    'Beijing': {'type': 'Outdoor', 'lat': 40.016, 'lon': 116.377},
    'Tokyo': {'type': 'Outdoor', 'lat': 35.636, 'lon': 139.790},
}

DEFAULT_META = {'type': 'Outdoor', 'lat': 40.0, 'lon': 0.0} # Outdoor default, coordinate fittizie

def get_tournament_meta(tourney_name):
    """Recupera metadati torneo, gestendo nomi parziali o non mappati."""
    # 1. Match Esatto
    if tourney_name in TOURNAMENT_META:
        return TOURNAMENT_META[tourney_name]
    
    # 2. Match Parziale (es. "Miami Masters" -> "Miami")
    for key, meta in TOURNAMENT_META.items():
        if key in tourney_name:
            return meta
            
    # 3. Default
    # Se il nome contiene 'Indoor' forziamo tipo Indoor
    meta = DEFAULT_META.copy()
    if 'Indoor' in tourney_name:
        meta['type'] = 'Indoor'
    
    return meta

# --- 2. CLIENT METEO ---

class WeatherClient:
    def __init__(self, cache_file="tennis_data_cache/weather_cache_bulk.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        if "data" not in self.cache: self.cache["data"] = {}
        if "processed_locations" not in self.cache: self.cache["processed_locations"] = []

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Errore caricamento cache meteo: {e}. Inizializzo cache vuota.")
                return {"data": {}, "processed_locations": []}
        return {"data": {}, "processed_locations": []}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f) # Minified for space
        except Exception as e:
            logging.error(f"Errore salvataggio cache meteo: {e}")

    def _get_cache_key(self, lat, lon, date_str):
        return f"{lat:.2f}_{lon:.2f}_{date_str}"
        
    def _get_loc_key(self, lat, lon):
        return f"{lat:.2f}_{lon:.2f}"

    def prefetch_location(self, lat, lon, start_year=2016, end_year=2026):
        """Scarica i dati meteo in un colpo solo per il range specificato."""
        loc_key = self._get_loc_key(lat, lon)
        if loc_key in self.cache["processed_locations"]:
            # Già scaricato
            return

        logging.info(f"Prefetch meteo per {loc_key} ({start_year}-{end_year})...")
        
        today = datetime.date.today()
        # Cap end_date at today to avoid 400 from Archive API for future dates
        target_end = datetime.date(end_year, 12, 31)
        final_date = min(target_end, today)
        
        start_date = f"{start_year}-01-01"
        end_date = final_date.strftime("%Y-%m-%d")
        
        # Usa Open-Meteo Archive
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                if "daily" in res_json and "time" in res_json["daily"]:
                    times = res_json["daily"]["time"]
                    temps = res_json["daily"]["temperature_2m_mean"]
                    hums = res_json["daily"]["relative_humidity_2m_mean"]
                    
                    for i, date_str in enumerate(times):
                        t = temps[i]
                        h = hums[i]
                        if t is not None:
                            key = self._get_cache_key(lat, lon, date_str)
                            self.cache["data"][key] = {"temp": t, "humidity": h}
                    
                    # Mark as processed
                    self.cache["processed_locations"].append(loc_key)
                    self._save_cache()
            else:
                 logging.warning(f"Errore Open-Meteo prefetch: {response.status_code}")
                 
        except Exception as e:
            logging.error(f"Eccezione prefetch: {e}")

    def get_weather(self, lat, lon, date_obj):
        """
        Recupera meteo (Tmeperatura media, Umidità media) per una data e luogo.
        Look up veloce in cache.
        """
        date_str = date_obj.strftime("%Y-%m-%d")
        cache_key = self._get_cache_key(lat, lon, date_str)
        
        # 1. Check Cache
        if cache_key in self.cache["data"]:
            return self.cache["data"][cache_key]
        
        # 2. Se è futuro, prova Forecast API (come prima, singola chiamata)
        # Ma non facciamo fallback su Historical singola perché abbiamo fatto prefetch.
        today = datetime.date.today()
        is_future = date_obj.date() > today if isinstance(date_obj, datetime.datetime) else date_obj > today
        
        if is_future:
             try:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
                    "start_date": date_str,
                    "end_date": date_str,
                    "timezone": "auto"
                }
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    res_json = response.json()
                    if "daily" in res_json and "temperature_2m_mean" in res_json["daily"]:
                        temps = res_json["daily"]["temperature_2m_mean"]
                        hums = res_json["daily"]["relative_humidity_2m_mean"]
                        if temps and temps[0] is not None:
                             return {"temp": temps[0], "humidity": hums[0]}
             except:
                 pass

        # 3. Fallback Defaults
        return {"temp": 20.0, "humidity": 50.0}

# Global instance
weather_client = WeatherClient()
