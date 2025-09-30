# app.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px


# =============================================================================
# Config
# =============================================================================
st.set_page_config(page_title="Pontos por cluster (cores distintas)", layout="wide")

DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"

# Paleta fixa (5 clusters)
PALETTE = {
    0: (230, 57, 70, 180),     # vermelho
    1: (42, 157, 143, 180),    # verde
    2: (66, 135, 245, 180),    # azul
    3: (155, 81, 224, 180),    # roxo
    4: (241, 196, 15, 180),    # amarelo
}


# =============================================================================
# Utilitários
# =============================================================================
@st.cache_data(show_spinner=False)
def load_points() -> pd.DataFrame:
    df = pd.read_parquet(POINTS_FILE)
    # saneamento de nomes esperados
    rename_map = {
        "lat": "latitude",
        "lng": "longitude",
        "lon": "longitude",
    }
    df = df.rename(columns=rename_map)
    # garantias mínimas
    for c in ["cluster", "latitude", "longitude"]:
        if c not in df.columns:
            raise KeyError(f"Coluna obrigatória ausente: {c}")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine em km (vetorizada)."""
    R = 6371.0088
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera:
      - centroid_lat / centroid_lon (média dos pontos do cluster)
      - radius_km = p90 da distância ao centróide
      - n_points
    Funciona só com os próprios pontos filtrados (não depende de arquivo extra).
    """
    if df.empty:
        return pd.DataFrame(
            columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]
        )

    # centróides
    cent = (
        df.groupby("cluster", as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
    )

    # distâncias aos centróides
    base = df.merge(cent, on="cluster", how="left")
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"], base["centroid_lat"], base["centroid_lon"]
    )

    # p90 por cluster
    r90 = (
        base.groupby("cluster")["dist_km"]
        .quantile(0.90)
        .reset_index()
        .rename(columns={"dist_km": "radius_km"})
    )

    # contagem
    npts = (
        df.groupby("cluster")
        .size()
        .reset_index(name="n_points")
    )

    out = cent.merge(r90, on="cluster", how="left").merge(npts, on="cluster", how="left")
    out = out.sort_values("cluster").reset_index(drop=True)
    return out


def make_deck(
    df_map: pd.DataFrame,
    summary: pd.DataFrame,
    show_p90: bool,
    show_counts: bool,
    view_state: pdk.ViewState | None = None,
) -> pdk.Deck:
    """Cria o Deck com pontos coloridos por cluster, círculos p90 e contagens."""
    if view_state is None:
        # enquadra nos dados atuais
        lat_min, lat_max = df_map["latitude"].min(), df_map["latitude"].max()
        lon_min, lon_max = df_map["longitude"].min(), df_map["longitude"].max()
        lat_c = float(lat_min + (lat_max - lat_min) / 2) if not math.isnan(lat_min) else -22.5
        lon_c = float(lon_min + (lon_max - lon_min) / 2) if not math.isnan(lon_min) else -48.8
        view_state = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=6.2)

    # cor RGBA por cluster
    df_map = df_map.copy()
    df_map["rgba"] = df_map["cluster"].map(PALETTE).fillna((200, 200, 200, 160))

    # camada dos pontos
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["longitude", "latitude"],
        get_radius=60,  # pixels em metros (aprox.)
        get_fill_color="rgba",
        pickable=True,
    )

    layers = [pts]

    if show_p90 and not summary.empty:
        ring_data = summary.copy()
        ring_data["rgba"] = ring_data["cluster"].map(PALETTE).fillna((120, 120, 120, 180))
        ring_data["radius_m"] = (ring_data["radius_km"].fillna(0) * 1000).astype(float)

        rings = pdk.Layer(
            "ScatterplotLayer",
            data=ring_data,
            get_position=["centroid_lon", "centroid_lat"],
            get_radius="radius_m",
            stroked=True,
            filled=False,
            lineWidthMinPixels=2,
            get_line_color="rgba",
        )
        layers.append(rings)

    if show_counts and not summary.empty:
        txt = pdk.Layer(
            "TextLayer",
            data=summary,
            get_position=["centroid_lon", "centroid_lat"],
            get_text="n_points",
            get_color=[0, 0, 0, 220],
            get_size=12,
            get_alignment_baseline="'bottom'",
        )
        layers.append(txt)

    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "Cluster {cluster}"},
    )


# =============================================================================
# UI
# =============================================================================
df = load_points()

# --- Sidebar (filtros) ---
st.sidebar.header("Filtros")

# chips de cluster
clusters_sorted = sorted(df["cluster"].dropna().astype(int).unique().tolist())
default_sel = clusters_sorted  # todos marcados
sel = st.sidebar.multiselect(
    "Clusters",
    options=clusters_sorted,
    default=default_sel,
    format_func=lambda x: str(x),
)

show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# filtro nos dados
df_map = df[df["cluster"].isin(sel)].copy()

# métricas do topo
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", len(df))
c2.metric("Pontos no mapa (amostra)", len(df_map))
lat_min, lat_max = df_map["latitude"].min(), df_map["latitude"].max()
lon_min, lon_max = df_map["longitude"].min(), df_map["longitude"].max()
c3.metric("Lat range", f"{lat_min:.5f} ~ {lat_max:.5f}" if not np.isnan(lat_min) else "—")
c4.metric("Lon range", f"{lon_min:.5f} ~ {lon_max:.5f}" if not np.isnan(lon_min) else "—")

st.title("Pontos por cluster (cores distintas)")

# Tabs
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

with tab_map:
    # resumo para círculos/contagens
    summary = summarize_clusters(df_map)

    deck = make_deck(df_map, summary, show_p90=show_p90, show_counts=show_counts)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.subheader("Legenda (cluster → cor)")
        cols = st.columns(min(5, len(PALETTE)))
        for i, (clu, rgba) in enumerate(PALETTE.items()):
            with cols[i % len(cols)]:
                st.color_picker(f"Cluster {clu}", "#%02x%02x%02x" % rgba[:3], key=f"legend_{clu}", disabled=True)

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")

    summary = summarize_clusters(df_map)
    st.dataframe(summary, use_container_width=True, height=280)

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar resumo (CSV)", csv, file_name="clusters_summary.csv", mime="text/csv")

    if not summary.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(summary, x="cluster", y="n_points", title="Pontos por cluster")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(summary, x="cluster", y="radius_km", title="Raio p90 (km) por cluster")
            st.plotly_chart(fig, use_container_width=True)
