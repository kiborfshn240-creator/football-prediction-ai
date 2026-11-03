import streamlit as st
import pandas as pd
import numpy as np
import requests
import math

st.set_page_config(
    page_title="Football Prediction AI v2.0",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
h1, h2, h3 { color: #38bdf8; }
div[data-testid="stMetric"] {
    background: #172033;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Prediction AI v2.0")
st.caption("Educational football statistics prototype • Not a guarantee of match results")

URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"

@st.cache_data(ttl=3600)
def load_data():
    response = requests.get(URL, timeout=20)
    response.raise_for_status()
    raw = response.json()

    rows = []
    for match in raw.get("matches", []):
        score = match.get("score", {})
        ft = score.get("ft") if isinstance(score, dict) else None

        if not ft or len(ft) != 2:
            continue

        try:
            rows.append({
                "Home": match["team1"],
                "Away": match["team2"],
                "HG": int(ft[0]),
                "AG": int(ft[1])
            })
        except (KeyError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows)

def poisson_probability(lam, goals):
    return (math.exp(-lam) * lam**goals) / math.factorial(goals)

def predict(home, away, df):
    home_games = df[df["Home"] == home]
    away_games = df[df["Away"] == away]

    if len(home_games) < 3 or len(away_games) < 3:
        return None

    home_gf = home_games["HG"].mean()
    home_ga = home_games["AG"].mean()
    away_gf = away_games["AG"].mean()
    away_ga = away_games["HG"].mean()

    league_home = df["HG"].mean()
    league_away = df["AG"].mean()

    xh = max(0.1, home_gf * away_ga / max(league_home, 0.1))
    xa = max(0.1, away_gf * home_ga / max(league_away, 0.1))

    p_home = p_draw = p_away = 0.0

    for h in range(11):
        for a in range(11):
            p = poisson_probability(xh, h) * poisson_probability(xa, a)
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    return {
        "x_home": xh,
        "x_away": xa,
        "home": p_home / total * 100,
        "draw": p_draw / total * 100,
        "away": p_away / total * 100
    }

try:
    df = load_data()
    teams = sorted(set(df["Home"]) | set(df["Away"]))

    st.success(f"Data berhasil dimuat: {len(df)} pertandingan musim 2025–26")

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Tim kandang", teams, index=0)
    with col2:
        away = st.selectbox("Tim tandang", teams, index=min(1, len(teams)-1))

    if st.button("ANALISIS PERTANDINGAN", use_container_width=True):
        if home == away:
            st.warning("Pilih dua tim yang berbeda.")
        else:
            result = predict(home, away, df)

            if result is None:
                st.warning("Data pertandingan kandang/tandang belum cukup.")
            else:
                st.subheader("Hasil analisis")
                c1, c2, c3 = st.columns(3)
                c1.metric("Menang kandang", f'{result["home"]:.1f}%')
                c2.metric("Seri", f'{result["draw"]:.1f}%')
                c3.metric("Menang tandang", f'{result["away"]:.1f}%')

                st.write(f'Perkiraan gol model: {result["x_home"]:.2f} - {result["x_away"]:.2f}')
                st.caption("Angka ini merupakan estimasi model dari data historis, bukan peluang resmi atau jaminan hasil.")

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.info("Periksa koneksi internet, lalu coba muat ulang aplikasi.")
