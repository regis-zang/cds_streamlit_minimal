# app.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Configurações básicas
# ------------------------------------------------------------
st.set_page_config(page_title="CDs & Raios (minimal)", layout="wide")

DATA_PATH = Path("DataBase") / "points_enriched_final.parquet"

# Paleta (uma cor por cluster 0,1,2,3,4…)
PALETTE = [
    [225, 82, 79],    # 0 - vermelho
    [44, 160, 44],    # 1 - verde
    [99, 110, 250],   # 2 - azul
    [171, 99, 250],   # 3 - roxo
    [255, 207, 86],   # 4 - amarelo
    [23, 190, 207],   # 5 - ciano (sobras)
    [214, 39, 40],    # 6
    [148, 103, 189],  # 7
]

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normaliza nomes esperados
    df = df.rename(columns={"lat": "latitude", "lng": "longitude"})
    # Garantias mínimas
    for col in ("cluster", "latitude", "longitude"):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")
    # Tipos
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["latitude", "longitude", "cluster"]).copy()
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em KM entre (lat1,lon1) e (lat2,lon2)."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um df com:
      cluster, centroid_lat, centroid_lon, radius_km (p90), n_points
    """
    # centróides por mediana (mais robusto)
    cent = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .median()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
        .reset_index()
    )

    base = df.merge(cent, on="cluster", how="left")
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"], base["centroid_lat"], base["centroid_lon"]
    )

    r90 = base.groupby("cluster")["dist_km"].quantile(0.90).reset_index(name="radius_km")
    cnt = base.groupby("cluster").size().reset_index(name="n_points")

    summary = cent.merge(r90, on="cluster").merge(cnt, on="cluster")
    summary["radius_m"] = (summary["radius_km"] * 1000.0).astype(float)
    return summary


def make_view_state(df: pd.DataFrame) -> pdk.ViewState:
    lat0 = df["latitude"].mean()
    lon0 = df["longitude"].mean()
    return pdk.ViewState(latitude=float(lat0), longitude=float(lon0), zoom=6)


def color_map_for_clusters(unique_clusters: list[int]) -> dict[int, list[int]]:
    cmap = {}
    for i, c in enumerate(sorted(unique_clusters)):
        cmap[c] = PALETTE[i % len(PALETTE)]
    return cmap


def make_deck(
    df_map: pd.DataFrame,
    show_p90: bool,
    show_counts: bool,
) -> pdk.Deck:
    # Cores por cluster
    clusters = sorted(df_map["cluster"].unique().tolist())
    cmap = color_map_for_clusters(clusters)
    df_map = df_map.copy()
    df_map["rgba"] = df_map["cluster"].map(lambda c: cmap[int(c)] + [180])

    # Camada de base (TileLayer – Carto Light)
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )

    # Pontos
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        get_radius=60,  # em metros (tamanho fixo bom para zoom 5–8)
        pickable=True,
        stroked=True,
        get_line_color=[0, 0, 0, 120],
        get_line_width=1,
    )

    layers = [base_layer, pts]

    # Resumo por cluster (para círculos e contagens)
    summary = compute_cluster_summary(df_map)

    if show_p90 and not summary.empty:
        rings = pdk.Layer(
            "ScatterplotLayer",
            data=summary,
            get_position="[centroid_lon, centroid_lat]",
            get_radius="radius_m",  # metros
            stroked=True,
            filled=False,
            get_line_color=[51, 136, 255, 180],
            get_line_width=2,
        )
        layers.append(rings)

    if show_counts and not summary.empty:
        # Cor do texto segue a cor do cluster
        summary = summary.copy()
        cmap = color_map_for_clusters(summary["cluster"].tolist())
        summary["text_color"] = summary["cluster"].map(lambda c: cmap[int(c)] + [255])
        summary["label"] = summary["n_points"].astype(str)

        labels = pdk.Layer(
            "TextLayer",
            data=summary,
            get_position="[centroid_lon, centroid_lat]",
            get_text="label",
            get_color="text_color",
            get_size=18,
            get_alignment_baseline="'center'",
        )
        layers.append(labels)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=make_view_state(df_map),
        map_provider=None,   # evita tentar usar Mapbox
        map_style=None,
        tooltip={"text": "Cluster {cluster}\n({latitude}, {longitude})"},
    )
    return deck


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
df = load_data(DATA_PATH)

# Sidebar
st.sidebar.header("Filtros")
all_clusters = sorted(df["cluster"].dropna().astype(int).unique().tolist())
sel = st.sidebar.multiselect(
    "Clusters", options=all_clusters, default=all_clusters, format_func=lambda x: str(x)
)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

df_map = df[df["cluster"].isin(sel)].copy()
if df_map.empty:
    st.warning("Nenhum ponto para os filtros selecionados.")
    st.stop()

# Métricas (corrigido: agora usamos .metric de verdade)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
lat_rng = f"{df_map['latitude'].min():.5f} ~ {df_map['latitude'].max():.5f}"
lon_rng = f"{df_map['longitude'].min():.5f} ~ {df_map['longitude'].max():.5f}"
c3.metric("Lat range", lat_rng)
c4.metric("Lon range", lon_rng)

# Abas
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

# ------------------ Mapa
with tab_map:
    deck = make_deck(df_map, show_p90=show_p90, show_counts=show_counts)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(min(6, len(sel) if sel else len(all_clusters)))
        cmap = color_map_for_clusters(sel if sel else all_clusters)
        for i, c in enumerate(sorted(cmap.keys())):
            r, g, b = cmap[c][:3]
            cols[i % len(cols)].markdown(
                f"<div style='display:flex;align-items:center;'>"
                f"<div style='width:14px;height:14px;border-radius:50%;"
                f"background:rgb({r},{g},{b});margin-right:8px;'></div>"
                f"Cluster {c}</div>",
                unsafe_allow_html=True,
            )

# ------------------ CDs & Raios
with tab_charts:
    summary = compute_cluster_summary(df_map)
    st.subheader("CDs & Raios (resumo por cluster)")
    st.dataframe(
        summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]]
        .sort_values("cluster")
        .reset_index(drop=True),
        use_container_width=True,
    )

    left, right = st.columns(2)
    if not summary.empty:
        fig1 = px.bar(
            summary.sort_values("n_points"),
            x="n_points",
            y="cluster",
            orientation="h",
            title="Pontos por cluster",
            labels={"cluster": "cluster", "n_points": "pontos"},
        )
        left.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            summary.sort_values("radius_km"),
            x="radius_km",
            y="cluster",
            orientation="h",
            title="Raio p90 (km) por cluster",
            labels={"cluster": "cluster", "radius_km": "km"},
            color_discrete_sequence=["#ff7f0e"],
        )
        right.plotly_chart(fig2, use_container_width=True)

