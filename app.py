# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# =======================
# Config / arquivos
# =======================
st.set_page_config(page_title="Pontos por cluster (cores distintas)", layout="wide")

DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"

# Carto Light (padrão)
CARTO_LIGHT = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

# Paleta fixa por cluster (RGBA)
PALETTE: Dict[int, List[int]] = {
    0: [235, 87, 87, 190],    # vermelho
    1: [46, 204, 113, 190],   # verde
    2: [66, 133, 244, 190],   # azul
    3: [155, 81, 224, 190],   # roxo
    4: [241, 196, 15, 190],   # amarelo
}
FALLBACK_COLOR = [0, 0, 0, 190]


# =======================
# Utilidades
# =======================
@st.cache_data(show_spinner=False)
def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.rename(columns={"lat": "latitude", "lon": "longitude", "lng": "longitude"})
    required = {"latitude", "longitude", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no arquivo: {missing}")
    # tipos básicos (não forço aqui para não perder dados; força no resumo)
    return df


def colorize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rgba"] = out["cluster"].map(PALETTE).apply(
        lambda c: c if isinstance(c, list) else FALLBACK_COLOR
    )
    return out


def haversine_array(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um DF com:
      cluster, centroid_lat, centroid_lon, radius_km (p90), n_points
    — Robusta a tipos errados e dados faltantes.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]
        )

    # Mantém somente colunas necessárias e força tipos
    base = df.loc[:, ["cluster", "latitude", "longitude"]].copy()

    base["cluster"] = pd.to_numeric(base["cluster"], errors="coerce")
    base["latitude"] = pd.to_numeric(base["latitude"], errors="coerce")
    base["longitude"] = pd.to_numeric(base["longitude"], errors="coerce")

    base = base.dropna(subset=["cluster", "latitude", "longitude"])
    if base.empty:
        return pd.DataFrame(
            columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]
        )

    # Garante inteiros no cluster (após remover NaN)
    base["cluster"] = base["cluster"].astype(int)

    # Centróides + contagem usando agg nomeado (evita rename)
    cents = (
        base.groupby("cluster", as_index=False)
        .agg(
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            n_points=("cluster", "size"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    # Junta de volta para medir distâncias
    joined = base.merge(
        cents[["cluster", "centroid_lat", "centroid_lon"]],
        on="cluster",
        how="left",
        validate="many_to_one",
    )

    # Distâncias haversine
    dist = haversine_array(
        joined["latitude"].values,
        joined["longitude"].values,
        joined["centroid_lat"].values,
        joined["centroid_lon"].values,
    )
    joined = joined.assign(dist_km=dist)

    # Raio p90
    r90 = (
        joined.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )

    summary = cents.merge(r90, on="cluster", how="left")
    summary["radius_km"] = pd.to_numeric(summary["radius_km"], errors="coerce").fillna(0.0)

    return summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]]


def make_deck(df_map: pd.DataFrame, show_p90: bool, show_counts: bool) -> pdk.Deck:
    # --- Basemap (TileLayer) ---
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
    )

    # --- Pontos (cores do df via coluna 'rgba') ---
    pts_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        pickable=False,
        stroked=False,
        filled=True,
        radius_units="pixels",
        get_radius=6,
    )

    layers = [base_layer, pts_layer]

    # --- Resumo por cluster (para círculos e contagens) ---
    summary = compute_cluster_summary(df_map)

    # ====== AQUI está a mudança ======
    if show_p90 and not summary.empty:
        circ = summary.assign(radius_m=(summary["radius_km"] * 1000.0).clip(lower=0))

        # desenha apenas o contorno (sem preencher) para não “tingir” o mapa
        rings = pdk.Layer(
            "ScatterplotLayer",
            data=circ,
            get_position="[centroid_lon, centroid_lat]",
            get_radius="radius_m",
            radius_units="meters",
            stroked=True,
            filled=False,                      # <<< SEM preenchimento
            get_line_color=[0, 149, 255, 180], # contorno azul
            line_width_min_pixels=2,
            pickable=False,
        )
        layers.append(rings)

        # Se preferir manter preenchimento, troque por:
        # filled=True, get_fill_color=[0,149,255,10]  # opacidade bem baixa

    if show_counts and not summary.empty:
        labels = summary.rename(
            columns={"centroid_lat": "lat", "centroid_lon": "lon", "n_points": "text"}
        )[["lat", "lon", "text"]]
        text_layer = pdk.Layer(
            "TextLayer",
            data=labels,
            get_position="[lon, lat]",
            get_text="text",
            get_color=[64, 64, 64, 240],
            get_size=16,
            get_angle=0,
            get_alignment_baseline="'bottom'",
        )
        layers.append(text_layer)

    # Enquadramento inicial
    lat_center = float(df_map["latitude"].mean()) if not df_map.empty else -23.5
    lon_center = float(df_map["longitude"].mean()) if not df_map.empty else -46.6
    view_state = pdk.ViewState(
        latitude=lat_center, longitude=lon_center,
        zoom=7, min_zoom=3, max_zoom=18, bearing=0, pitch=0
    )

    # map_style=None evita Mapbox e usa apenas o TileLayer
    return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None)


# =======================
# UI
# =======================
st.title("Pontos por cluster (cores distintas)")

# Carregamento
try:
    df = load_points(POINTS_FILE)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Sidebar — filtros / opções
st.sidebar.header("Filtros")
clusters = sorted(pd.to_numeric(df["cluster"], errors="coerce").dropna().astype(int).unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", clusters, default=clusters)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# Filtragem + cores
df_map = df[df["cluster"].isin(sel_clusters)].copy()
df_map = colorize(df_map)

# Cabeçalho de métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
lat_min, lat_max = pd.to_numeric(df["latitude"], errors="coerce").min(), pd.to_numeric(df["latitude"], errors="coerce").max()
lon_min, lon_max = pd.to_numeric(df["longitude"], errors="coerce").min(), pd.to_numeric(df["longitude"], errors="coerce").max()
c3.metric("Lat range", f"{lat_min:.5f} ~ {lat_max:.5f}")
c4.metric("Lon range", f"{lon_min:.5f} ~ {lon_max:.5f}")

# Tabs
tab_map, tab_charts = st.tabs(["🗺️ Mapa", "📊 CDs & Raios"])

with tab_map:
    st.subheader("Mapa por cluster")
    if df_map.empty:
        st.info("Nenhum ponto para os filtros selecionados.")
    else:
        deck = make_deck(df_map, show_p90=show_p90, show_counts=show_counts)
        st.pydeck_chart(deck, use_container_width=True)

        if show_legend:
            st.markdown("**Legenda (cluster → cor)**")
            cols = st.columns(len(PALETTE))
            for i, (cl, rgba) in enumerate(sorted(PALETTE.items())):
                with cols[i]:
                    st.write(
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]/255});"
                        f"border-radius:50%;margin-right:6px;'></span>"
                        f"Cluster {cl}",
                        unsafe_allow_html=True,
                    )

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    summary = compute_cluster_summary(df_map)
    st.dataframe(summary, use_container_width=True)

    if not summary.empty:
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar resumo (CSV)", data=csv, file_name="cds_raios.csv")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                summary.sort_values("n_points", ascending=False),
                x="cluster",
                y="n_points",
                title="Pontos por cluster",
                labels={"cluster": "cluster", "n_points": "pontos"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(
                summary.sort_values("radius_km", ascending=False),
                x="cluster",
                y="radius_km",
                title="Raio p90 (km) por cluster",
                labels={"cluster": "cluster", "radius_km": "km"},
            )
            st.plotly_chart(fig2, use_container_width=True)
