import streamlit as st
import pandas as pd
import joblib
import warnings
import os
import gdown
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Simulador CS:GO", layout="wide")

# -------------------------------
# HEADER
# -------------------------------
st.markdown("## 🎮 Simulador de enfrentamientos CS:GO")
st.caption("Predicción de resultados basada en Machine Learning")

st.divider()

# -------------------------------
# ESTADO
# -------------------------------
if "team1" not in st.session_state:
    st.session_state.team1 = None
if "team2" not in st.session_state:
    st.session_state.team2 = None
if "maps" not in st.session_state:
    st.session_state.maps = []
if "format" not in st.session_state:
    st.session_state.format = "BO1"

# -------------------------------
# LIMPIAR
# -------------------------------
if st.button("🧼 Limpiar selección"):
    st.session_state.team1 = None
    st.session_state.team2 = None
    st.session_state.maps = []
    st.session_state.format = "BO1"
    st.rerun()

# -------------------------------
# DATOS
# -------------------------------

# Descargar modelos si no existen
if not os.path.exists("random_forest_model.pkl"):
    gdown.download(id="12DYSlP7XVWlFfeWzd_Oxjq-147alJjb2",
                   output="random_forest_model.pkl",
                   quiet=False)

if not os.path.exists("feature_columns.pkl"):
    gdown.download(id="1EYkAbl6shjCh9oPYdipQQ8nrXg1geXMD",
                   output="feature_columns.pkl",
                   quiet=False)

model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
data = pd.read_csv("Datasets/csgo_dataset_prepared.csv")

# -------------------------------
# TEAM STATS
# -------------------------------
team1_df = data[[
    "team_1", "team1_avg_kills", "team1_avg_rating", "rank_1"
]].rename(columns={
    "team_1": "team",
    "team1_avg_kills": "avg_kills",
    "team1_avg_rating": "avg_rating",
    "rank_1": "rank"
})

team2_df = data[[
    "team_2", "team2_avg_kills", "team2_avg_rating", "rank_2"
]].rename(columns={
    "team_2": "team",
    "team2_avg_kills": "avg_kills",
    "team2_avg_rating": "avg_rating",
    "rank_2": "rank"
})

team_stats = pd.concat([team1_df, team2_df])
team_stats = team_stats.groupby("team").mean().reset_index()

teams = sorted(team_stats["team"].unique())
maps_list = sorted(data["_map"].unique())

# -------------------------------
# UI
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Equipo 1", teams, key="team1")

with col2:
    team2 = st.selectbox("Equipo 2", teams, key="team2")

if team1 == team2:
    st.error("Selecciona equipos diferentes")
    st.stop()

st.divider()

format_option = st.selectbox("Formato", ["BO1", "BO3", "BO5"], key="format")
num_maps = {"BO1": 1, "BO3": 3, "BO5": 5}[format_option]

selected_maps = st.multiselect(
    f"Selecciona {num_maps} mapas",
    maps_list,
    key="maps",
    max_selections=num_maps
)

if selected_maps:
    st.info(f"Mapas seleccionados: {', '.join(selected_maps)}")

st.divider()

# -------------------------------
# FUNCIONES
# -------------------------------
def predict_map(team1, team2, selected_map):

    t1 = team_stats[team_stats["team"] == team1].iloc[0]
    t2 = team_stats[team_stats["team"] == team2].iloc[0]

    input_data = {
        "rank_1": t1["rank"],
        "rank_2": t2["rank"],
        "team1_avg_kills": t1["avg_kills"],
        "team1_avg_rating": t1["avg_rating"],
        "team2_avg_kills": t2["avg_kills"],
        "team2_avg_rating": t2["avg_rating"],
        "rating_diff": t1["avg_rating"] - t2["avg_rating"],
        "kills_diff": t1["avg_kills"] - t2["avg_kills"],
        "rank_diff": t2["rank"] - t1["rank"]
    }

    X_input = pd.DataFrame([input_data])

    for col in feature_columns:
        if col.startswith("_map_"):
            X_input[col] = 0

    map_col = f"_map_{selected_map}"
    if map_col in X_input.columns:
        X_input[map_col] = 1

    X_input = X_input[feature_columns]

    proba = model.predict_proba(X_input)[0]

    return {
        "team1_prob": float(proba[0]),
        "team2_prob": float(proba[1])
    }


def simulate_series(team1, team2, maps):

    results = []
    team1_wins = 0
    team2_wins = 0

    team1_probs = []
    team2_probs = []

    for m in maps:
        pred = predict_map(team1, team2, m)

        winner = team1 if pred["team1_prob"] > pred["team2_prob"] else team2

        if winner == team1:
            team1_wins += 1
        else:
            team2_wins += 1

        team1_probs.append(pred["team1_prob"])
        team2_probs.append(pred["team2_prob"])

        results.append({
            "map": m,
            "team1_prob": pred["team1_prob"],
            "team2_prob": pred["team2_prob"],
            "winner": winner
        })

    final_winner = team1 if team1_wins > team2_wins else team2

    return {
        "maps": results,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "winner": final_winner,
        "team1_series_prob": sum(team1_probs) / len(team1_probs),
        "team2_series_prob": sum(team2_probs) / len(team2_probs)
    }

# -------------------------------
# SIMULACIÓN
# -------------------------------
if st.button("🎯 Ejecutar simulación"):

    if len(selected_maps) != num_maps:
        st.warning("Selecciona el número correcto de mapas")
    else:
        result = simulate_series(team1, team2, selected_maps)

        st.subheader("Resultados por mapa")

        df = pd.DataFrame(result["maps"])

        df_pretty = pd.DataFrame({
            "Mapa": df["map"],
            team1: df["team1_prob"],
            team2: df["team2_prob"],
            "Ganador": df["winner"]
        })

        # 🎨 COLORES INTELIGENTES
        def style_probs(val):
            if isinstance(val, float):
                if val > 0.5:
                    return "background-color: #d4edda; color: black"
                elif val < 0.5:
                    return "background-color: #f8d7da; color: black"
                else:
                    return "background-color: #fff3cd; color: black"
            return ""

        def style_static(col):
            return ["background-color: #f1f1f1"] * len(col)

        styled = df_pretty.style \
            .map(style_probs, subset=[team1, team2]) \
            .apply(style_static, subset=["Mapa", "Ganador"]) \
            .format({team1: "{:.1%}", team2: "{:.1%}"})

        st.dataframe(styled, use_container_width=True)

        st.divider()

        # -----------------------
        # RESULTADO FINAL PRO
        # -----------------------
        st.subheader("Resultado del enfrentamiento")

        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.metric(team1, result["team1_wins"])

        with col2:
            st.metric(team2, result["team2_wins"])

        with col3:
            st.markdown("### 🏆")
            st.markdown(f"**{result['winner']}**")

        st.divider()

        # -----------------------
        # PROBABILIDADES
        # -----------------------
        st.subheader("Probabilidad del enfrentamiento")

        st.write(f"{team1} ({result['team1_series_prob']:.2%})")
        st.progress(result["team1_series_prob"])

        st.write(f"{team2} ({result['team2_series_prob']:.2%})")
        st.progress(result["team2_series_prob"])
