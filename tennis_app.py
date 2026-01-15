import streamlit as st
import pandas as pd
import tennis_bot
import weather_utils
from datetime import datetime

# --- CUSTOM CSS & STYLING ---
def local_css():
    st.markdown("""
    <style>
    /* Importante Font Google */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background Gradient (Dark Premium) */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f1c2e 0%, #040d12 90%);
    }
    
    /* Card Glassmorphism Effect */
    div.css-1r6slb0.e1tzin5v2, div[data-testid="stMetric"], div.stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e0e0e0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0b1219;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom Button (Tennis Green Gradient) */
    div.stButton > button {
        background: linear-gradient(135deg, #ccff00 0%, #99cc00 100%);
        color: #000;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(204, 255, 0, 0.3);
        color: #000;
    }
    
    /* Progress Bar Custom Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ccff00, #4facfe);
    }
    
    /* Value Bet Success/Error Custom */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

local_css()

# Titolo Moderno con Emojis
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0;">🎾 TENNIS AI 2026</h1>
        <p style="color: #888; letter-spacing: 1px; text-transform: uppercase; font-weight: 600;">
            Next Gen ATP Prediction Engine
        </p>
    </div>
""", unsafe_allow_html=True)

# sidebar clean
st.sidebar.markdown("### ⚙️ PARAMETERS")

# Caricamento Modello (Cached - No changes to logic)
@st.cache_resource
def get_model():
    with st.spinner('🚀 Booting Neural Engine...'):
        return tennis_bot.build_model()

# Config Sidebar Input
confidence_threshold = st.sidebar.slider("🔹 Edge Threshold (%)", 0, 20, 5)
st.sidebar.markdown("---")
st.sidebar.subheader("🏟️ Match Setup")
surface = st.sidebar.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"], index=0)
best_of = st.sidebar.radio("Format", ["Best of 3", "Best of 5"], index=1)
best_of_val = 3 if best_of == "Best of 3" else 5

# Tournament & Date
tourney_list = list(weather_utils.TOURNAMENT_META.keys())
tourney_name = st.sidebar.selectbox("🏟️ Tournament", tourney_list, index=0)
match_date = st.sidebar.date_input("📅 Match Date", datetime.today())

# Country Code map (Partial map for display/backend compatibility if needed)
# Backend uses tourney_name primarily for weather now, but Country Code for Home Adv.
# We can infer or keep UNK default. Let's try to map generic.
tourney_country = "UNK" # Simplified, backend handles Home Adv logic via name mapping internally?
# Actually tennis_bot.predict_2026_match_data takes tourney_country_code.
# Let's try to use the dictionary in bot if possible or just pass UNK if user doesn't specify.
# To keep UI simple, I won't ask for Country Code explicitly if I have Tournament Name.
# But for Home Advantage I need it.
# I will infer it from tennis_bot.TOURNEY_COUNTRY_MAP if possible.
country_code = tennis_bot.TOURNEY_COUNTRY_MAP.get(tourney_name, 'UNK')
st.sidebar.caption(f"📍 Location: {country_code}")

try:
    model, stats, h2h, elo, elo_surf, dates, pressure, players = get_model()
    st.sidebar.success("✅ System Ready")
except Exception as e:
    st.error(f"❌ System Error: {e}")
    st.stop()

# Layout Selezione Giocatori (Hero Section)
col1, col_vs, col2 = st.columns([1, 0.2, 1])

# Lista Giocatori Ordinata per Ranking
sorted_players = sorted(players.values(), key=lambda x: x['rank'])
player_names = [p['name'] for p in sorted_players]

with col1:
    st.markdown("### 👤 Player 1")
    p1_name = st.selectbox("Select Player 1", player_names, index=0, label_visibility="collapsed")
    # Quick info preview logic can go here if needed

with col_vs:
    st.markdown("<h2 style='text-align: center; color: #666; padding-top: 20px;'>VS</h2>", unsafe_allow_html=True)

with col2:
    st.markdown("### 👤 Player 2")
    default_idx = 1 if len(player_names) > 1 else 0
    p2_name = st.selectbox("Select Player 2", player_names, index=default_idx, label_visibility="collapsed")

st.write("") # Spacer

# Prediction Button Centered
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    predict_clicked = st.button("🔮 ANALYZE MATCHUP", type="primary", use_container_width=True)

if predict_clicked:
    match_data = tennis_bot.predict_2026_match_data(
        model, stats, h2h, elo, elo_surf, dates, pressure, players, p1_name, p2_name, surface, country_code,
        tourney_name=tourney_name, match_date=match_date
    )
    st.session_state['match_data'] = match_data

if 'match_data' in st.session_state and st.session_state['match_data']:
    match_data = st.session_state['match_data']
    
    prob_p1 = match_data['prob_p1']
    prob_p2 = 1 - prob_p1
    winner = p1_name if prob_p1 > 0.5 else p2_name
    win_prob = prob_p1 if prob_p1 > 0.5 else prob_p2
    
    # --- RESULT HERO CARD ---
    st.markdown("---")
    
    # Gradient Metrics
    m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
    
    with m_col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 15px; border: 1px solid #ccff00;">
            <h3 style="margin:0; color: #aaa; font-size: 1rem;">PREDICTED WINNER</h3>
            <h1 style="margin:10px 0; font-size: 3rem; color: #fff; text-shadow: 0 0 20px rgba(204,255,0,0.5);">{winner}</h1>
            <h2 style="margin:0; color: #ccff00; font-size: 2.5rem;">{win_prob*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.progress(float(prob_p1), text=f"WIN PROBABILITY DISTRIBUTION")
    st.caption(f"⬅️ {p1_name} ({prob_p1*100:.1f}%) | {p2_name} ({prob_p2*100:.1f}%) ➡️")
        
    # Weather Info
    w = match_data.get('weather', {})
    if w:
        st.markdown("### 🌤️ FORECAST CONDITIONS")
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.metric("Temperature", f"{w.get('temp', 20):.1f}°C")
        w_col2.metric("Humidity", f"{w.get('humidity', 50):.0f}%")
        env_type = "Indoor 🏟️" if w.get('is_indoor', 0) == 1 else "Outdoor ☀️"
        w_col3.metric("Court Type", env_type)
        st.markdown("---")

    # Stats Comparison with Progress Bars inside Dataframe? Streamlit dataframe is interactive.
    # Let's clean up the table visual.
    st.markdown(f"### 📊 HEAD-TO-HEAD STATISTICS ({surface})")
    
    # Prepare Data with proper Formatting
    def fmt_pct(val): return f"{val*100:.1f}%"
    
    metrics = {
        "Metric": ["🏆 ATP Rank", "⚡ Elo Rating", f"🏟️ {surface} Elo", "📈 Recent Form", "🧱 BP Save %", "🔥 Decider Win %"],
        p1_name: [
            f"#{match_data['stats_p1']['rank']}", 
            f"{int(match_data['stats_p1']['elo'])}",
            f"{int(match_data['stats_p1']['elo_surface'])}",
            f"{match_data['stats_p1']['form']:.2f}",
            fmt_pct(match_data['stats_p1']['bp_save']),
            fmt_pct(match_data['stats_p1']['decider'])
        ],
        p2_name: [
            f"#{match_data['stats_p2']['rank']}", 
            f"{int(match_data['stats_p2']['elo'])}",
            f"{int(match_data['stats_p2']['elo_surface'])}",
            f"{match_data['stats_p2']['form']:.2f}",
            fmt_pct(match_data['stats_p2']['bp_save']),
            fmt_pct(match_data['stats_p2']['decider'])
        ]
    }
    
    df_stats = pd.DataFrame(metrics).set_index("Metric")
    st.table(df_stats) # st.table handles styling better for static comparison than data_editor in specific custom cases
    
    st.markdown(f"<p style='text-align:center; color:#666;'>Historical H2H: <b style='color:#fff'>{match_data['h2h']}</b></p>", unsafe_allow_html=True)
    
    # --- VALUE BETTING SECTION ---
    st.markdown("### 💰 SMART BETTING EDGE")
    
    # Container for betting UI
    with st.container():
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            odds_p1 = st.number_input(f"Odds {p1_name}", min_value=1.01, value=1.85, step=0.01)
        with o_col2:
            odds_p2 = st.number_input(f"Odds {p2_name}", min_value=1.01, value=1.85, step=0.01)
            
        fair_odds_p1 = 1 / prob_p1 if prob_p1 > 0 else 99.0
        fair_odds_p2 = 1 / prob_p2 if prob_p2 > 0 else 99.0
        
        ana_col1, ana_col2 = st.columns(2)
        
        def render_value_card(col, name, book_odds, fair_odds, prob):
            with col:
                # ROI
                edge_pct = ((prob * book_odds) - 1) * 100
                is_value = book_odds > fair_odds * (1 + confidence_threshold/100)
                
                border_color = "#ccff00" if is_value else "#ff4444"
                title_color = "#ccff00" if is_value else "#ff4444"
                status_icon = "✅ DIAMOND BET" if is_value else "❌ NO VALUE"
                
                st.markdown(f"""
                <div style="border: 1px solid {border_color}; padding: 15px; border-radius: 10px; background: rgba(255,255,255,0.02);">
                    <h4 style="margin:0; color: #fff;">{name}</h4>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <span style="color:#888;">Fair Odds:</span>
                        <span style="color:#fff; font-weight:bold;">{fair_odds:.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color:#888;">Book Odds:</span>
                        <span style="color:#fff; font-weight:bold;">{book_odds:.2f}</span>
                    </div>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <div style="text-align: center;">
                        <h2 style="margin:0; color: {title_color};">{edge_pct:+.1f}% EDGE</h2>
                        <span style="font-size: 0.8rem; letter-spacing: 2px; color: {title_color};">{status_icon}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        render_value_card(ana_col1, p1_name, odds_p1, fair_odds_p1, prob_p1)
        render_value_card(ana_col2, p2_name, odds_p2, fair_odds_p2, prob_p2)
