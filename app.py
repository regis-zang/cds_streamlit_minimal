from __future__ import annotations

import io
import os
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st


# =========================
# Configurações gerais
# =========================
st.set_page_config(page_title="CDs - Mapa & Raios", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "DataBase")
PARQUET_FILE = os.path.join(DATA_DIR, "points_enriched_final.parquet")

# Paleta de cores (cicla se tiver mais clusters)
BASE_PALETTE = [
    (239, 83, 80),   # vermelho
    (76, 175, 80),   # verde
    (66, 133, 244),  # azul
    (186, 104, 200), # roxo
    (255, 235, 59),  # amarelo
    (255, 152, 0),   # laranja
    (0, 188, 212),   # ciano
    (121, 85, 72),   # marrom
]


# =========================
# Funções auxiliares
# =========================
@st.cache_data(show_spinner=False)
def load_points(path: str) -> pd.DataFrame:
    """Lê o Parquet e padroniza nomes + tipos."""
    df = pd.read_parquet(path)

    # normalização de nomes
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("lat", "latitude"):
            col_map[c] = "latitude"
        elif lc in ("lon", "lng", "longitude"):
            col_map[c] = "longitude"
        elif lc == "cluster":
            col_map[c] = "cluster"
    if col_map:
        df = df.rename(columns=col_map)

    required = {"latitude", "longitude", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltam colunas no Parquet: {missing}. Esperado: {sorted(list(required))}")

    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["latitude", "longitude", "cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    # padroniza tipos numéricos
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    return df.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine em KM (vetorizado)."""
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p) * np.cos(lat2*p) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _ensure_centroid_columns_after_merge(base: pd.DataFrame) -> pd.DataFrame:
    """
    Após o merge, garante que existam:
      - latitude, longitude (ponto)
      - centroid_lat, centroid_lon (centróide)
    Normaliza sufixos como _x/_y, _centroid, etc.
    """
    b = base.copy()

    # ponto (lado esquerdo)
    if "latitude" not in b.columns and "latitude_x" in b.columns:
        b = b.rename(columns={"latitude_x": "latitude"})
    if "longitude" not in b.columns and "longitude_x" in b.columns:
        b = b.rename(columns={"longitude_x": "longitude"})

    # centróide
    if "centroid_lat" not in b.columns:
        if "latitude_y" in b.columns:
            b = b.rename(columns={"latitude_y": "centroid_lat"})
        elif "latitude_centroid" in b.columns:
            b = b.rename(columns={"latitude_centroid": "centroid_lat"})

    if "centroid_lon" not in b.columns:
        if "longitude_y" in b.columns:
            b = b.rename(columns={"longitude_y": "centroid_lon"})
        elif "longitude_centroid" in b.columns:
            b = b.rename(columns={"longitude_centroid": "centroid_lon"})

    # fallback final: primeira coluna que contenha 'lat' (exceto 'latitude')
    if "centroid_lat" not in b.columns:
        lat_cands = [c for c in b.columns if "lat" in c.lower() and c != "latitude"]
        if lat_cands:
            b = b.rename(columns={lat_cands[0]: "centroid_lat"})
    if "centroid_lon" not in b.columns:
        lon_cands = [c for c in b.columns if "lon" in c.lower() and c != "longitude"]
        if lon_cands:
            b = b.rename(columns={lon_cands[0]: "centroid_lon"})

    needed = {"latitude", "longitude", "centroid_lat", "centroid_lon"}
    missing = needed - set(b.columns)
    if missing:
        raise KeyError(
            f"Ainda faltam colunas após o merge: {missing}. "
            f"Colunas disponíveis: {list(b.columns)}"
        )

    return b


@st.cache_data(show_spinner=False)
def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centróides, raio p90 e contagem por cluster — robusto a sufixos.
    Usa cache (chave baseada em hash do subconjunto relevante).
    """
    # cria assinatura leve do DF para cache
    sig = (
        pd.util.hash_pandas_object(df[["cluster", "latitude", "longitude"]], index=False)
        .sum()
    )
    _ = sig  # só para explicitar uso no cache

    cent = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .mean()
        .reset_index()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
    )

    base = df.merge(cent, on="cluster", how="left", suffixes=("", "_centroid"))
    base = _ensure_centroid_columns_after_merge(base)

    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"],
        base["centroid_lat"], base["centroid_lon"],
    )

    r90 = (
        base.groupby("cluster")["dist_km"].quantile(0.90)
        .reset_index()
        .rename(columns={"dist_km": "radius_km"})
    )
    n_points = base.groupby("cluster").size().reset_index(name="n_points")

    out = (
        cent.merge(r90, on="cluster")
            .merge(n_points, on="cluster")
            .sort_values("cluster")
            .reset_index(drop=True)
    )
    return out


def color_for_cluster(cluster: int) -> tuple[int, int, int]:
    return BASE_PALETTE[cluster % len(BASE_PALETTE)]


def add_rgba(df: pd.DataFrame, alpha: int = 185) -> pd.DataFrame:
    df = df.copy()
    df["rgba"] = df["cluster"].apply(lambda c: list(color_for_cluster(c)) + [alpha])
    return df


def make_deck(
    df_points: pd.DataFrame,
    show_r90: bool,
    show_counts: bool,
    view_state: pdk.ViewState,
) -> pdk.Deck:
    layers = []

    # Basemap claro (Carto)
    layers.append(
        pdk.Layer(
            "TileLayer",
            data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
        )
    )

    # Pontos
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_points,
            get_position=["longitude", "latitude"],
            get_fill_color="rgba",
            get_radius=60,
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 160],
            stroked=False,
            opacity=0.9,
            parameters={"depthTest": False},
        )
    )

    # Círculos (p90)
    if show_r90:
        summary = compute_cluster_summary(df_points)
        circ = summary.copy()
        circ["radius_m"] = (circ["radius_km"] * 1000.0).astype(float)
        circ["rgba_line"] = circ["cluster"].apply(lambda c: list(color_for_cluster(c)) + [200])

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=circ,
                get_position=["centroid_lon", "centroid_lat"],
                get_fill_color=[0, 0, 0, 0],
                get_line_color="rgba_line",
                stroked=True,
                line_width_min_pixels=2,
                get_radius="radius_m",
                radius_units="meters",
                opacity=0.6,
            )
        )

    # Contagem no centróide
    if show_counts:
        summary = compute_cluster_summary(df_points)
        summary["txt"] = summary["n_points"].astype(str)
        summary["text_color"] = summary["cluster"].apply(lambda c: list(color_for_cluster(c)))

        layers.append(
            pdk.Layer(
                "TextLayer",
                data=summary,
                get_position=["centroid_lon", "centroid_lat"],
                get_text="txt",
                get_color="text_color",
                get_size=18,
                get_angle=0,
                get_alignment_baseline="'center'",
            )
        )

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br>"
                "<b>Lat:</b> {latitude}<br>"
                "<b>Lon:</b> {longitude}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    return pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip,
        map_provider=None,
        map_style=None,
    )


def legend_markdown(df_points: pd.DataFrame) -> str:
    clusters = sorted(df_points["cluster"].unique().tolist())
    parts = []
    for c in clusters:
        r, g, b = color_for_cluster(c)
        parts.append(
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"background: rgb({r},{g},{b});border-radius:50%;margin-right:6px;'></span>"
            f"Cluster {c}"
        )
    return " &nbsp; ".join(parts)


def deck_to_html_bytes(deck: pdk.Deck) -> bytes:
    """Gera um HTML (self-contained) do mapa para download."""
    html = deck.to_html(as_string=True, css_background_color="white")
    return html.encode("utf-8")


# =========================
# App
# =========================
df = load_points(PARQUET_FILE)

# Sidebar
st.sidebar.header("Filtros")
all_clusters = sorted(df["cluster"].unique().tolist())
sel_clusters = st.sidebar.multiselect("Clusters", options=all_clusters, default=all_clusters)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_r90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# Filtro
if sel_clusters:
    df = df[df["cluster"].isin(sel_clusters)].copy()
else:
    st.warning("Selecione pelo menos um cluster para visualizar.")
    st.stop()

# Cores
df = add_rgba(df, alpha=185)

# Métricas
summary_for_metrics = compute_cluster_summary(df)
total_area_km2 = float((np.pi * (summary_for_metrics["radius_km"] ** 2)).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metr
