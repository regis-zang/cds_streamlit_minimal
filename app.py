# app.py
# ------------------------------------------------------------
# Dashboard minimalista de clusters + CDs & raios (p90)
# ------------------------------------------------------------

import math
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st
from pathlib import Path

# -------------------- Config --------------------
st.set_page_config(page_title="CDs & Clusters", layout="wide")

DATA_FILE = Path("DataBase/points_enriched_final.parquet")

# Paleta de cores por cluster (consistente entre mapa e legendas)
CLUSTER_COLORS = {
    0: "#e74c3c",  # vermelho
    1: "#2ecc71",  # verde
    2: "#3498db",  # azul
    3: "#9b59b6",  # roxo
    4: "#f1c40f",  # amarelo
}

# -------------------- Utils --------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # nomes esperados: latitude, longitude, cluster
    # se vierem como strings, converte
    for c in ("latitude", "longitude"):
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # cluster como int
    if df["cluster"].dtype != "int64" and df["cluster"].dtype != "int32":
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64").fillna(-1).astype(int)
    return df.dropna(subset=["latitude", "longitude", "cluster"])

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distância Haversine em km."""
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada cluster:
      - centroide (médias)
      - raio p90 (km)
      - contagem (n_points)
    """
    if df.empty:
        return pd.DataFrame(columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"])

    base = (
        df.groupby("cluster", as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
    )
    # distância de cada ponto ao centróide do seu cluster
    tmp = df.merge(base, on="cluster", how="left")
    tmp["dist_km"] = tmp.apply(
        lambda r: haversine_km(r["latitude"], r["longitude"], r["centroid_lat"], r["centroid_lon"]),
        axis=1,
    )
    r90 = tmp.groupby("cluster", as_index=False)["dist_km"].quantile(0.90).rename(columns={"dist_km": "radius_km"})
    out = base.merge(r90, on="cluster", how="left")
    out["n_points"] = df.groupby("cluster")["cluster"].transform("count").drop_duplicates().values
    return out.sort_values("cluster").reset_index(drop=True)

def make_deck(
    df_map: pd.DataFrame,
    show_p90: bool,
    show_counts: bool,
    view_state=None,
) -> pdk.Deck:
    """PyDeck Deck com TileLayer + pontos (Scatterplot).
       Opcionalmente adiciona círculos p90 e contagem no centróide (TextLayer)."""

    # Base map (carto light) – compatível com Streamlit Cloud
    base = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
    )

    # Mapeia cluster -> cor RGBA
    def cluster_rgba(c):
        hex_color = CLUSTER_COLORS.get(int(c), "#95a5a6")  # cinza para desconhecido
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return [r, g, b, 200]  # alfa ~78%

    df_map = df_map.copy()
    df_map["rgba"] = df_map["cluster"].map(cluster_rgba)

    # Pontos
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_radius=60,  # em metros
        get_fill_color="rgba",
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=12,
        stroked=False,
    )

    layers = [base, pts]

    # Círculos p90 + contagem
    if show_p90 or show_counts:
        summary = compute_cluster_summary(df_map)
        if not summary.empty:
            if show_p90:
                # Círculos p90 (em pixels ~ aproximação para escala única; simples e rápido)
                # Se quiser em metros reais, use `H3HexagonLayer` ou projete com `GreatCircleLayer`.
                circ = summary.copy()
                circ["rgba"] = circ["cluster"].map(lambda c: cluster_rgba(c)[:-1] + [60])  # alfa fraco
                circles = pdk.Layer(
                    "ScatterplotLayer",
                    data=circ,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius="radius_km * 1000",  # metros aproximados
                    get_fill_color="rgba",
                    stroked=True,
                    get_line_width=2,
                    get_line_color="[30, 136, 229, 150]",
                    pickable=False,
                )
                layers.append(circles)

            if show_counts:
                # Texto com contagem no centróide
                txt = summary[["centroid_lat", "centroid_lon", "n_points"]].copy()
                txt["text"] = txt["n_points"].astype(str)
                text_layer = pdk.Layer(
                    "TextLayer",
                    data=txt,
                    get_position="[centroid_lon, centroid_lat]",
                    get_text="text",
                    get_color="[30, 136, 229, 255]",
                    get_size=16,
                    get_alignment_baseline="'bottom'",
                )
                layers.append(text_layer)

    if view_state is None:
        # Centraliza pelos dados filtrados
        if not df_map.empty:
            view_state = pdk.ViewState(
                latitude=float(df_map["latitude"].mean()),
                longitude=float(df_map["longitude"].mean()),
                zoom=6.0,
                pitch=0,
            )
        else:
            view_state = pdk.ViewState(latitude=-23.5, longitude=-46.6, zoom=5.5, pitch=0)

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider=None,  # usando TileLayer
        tooltip={"html": "<b>Cluster:</b> {cluster}<br><b>Lat:</b> {latitude}<br><b>Lon:</b> {longitude}", "style": {"color": "white"}},
    )

# -------------------- App --------------------
df = load_data(DATA_FILE)

# Sidebar – filtros
st.sidebar.header("Filtros")
clusters_sorted = sorted(df["cluster"].unique().tolist())
sel_clusters = st.sidebar.multiselect(
    "Clusters", options=clusters_sorted, default=clusters_sorted, format_func=lambda x: str(x)
)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# Aplica filtros
df_f = df[df["cluster"].isin(sel_clusters)].copy()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_f):,}".replace(",", "."))
if not df_f.empty:
    lat_rng = f"{df_f['latitude'].min():.3f} ~ {df_f['latitude'].max():.3f}"
    lon_rng = f"{df_f['longitude'].min():.3f} ~ {df_f['longitude'].max():.3f}"
else:
    lat_rng = "–"
    lon_rng = "–"
c3.metric("Lat range", lat_rng)
c4.metric("Lon range", lon_rng)

# Tabs
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

with tab_map:
    st.subheader("Mapa por cluster")
    deck = make_deck(df_f, show_p90=show_p90, show_counts=show_counts)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(len(CLUSTER_COLORS))
        for i, (cl, color) in enumerate(CLUSTER_COLORS.items()):
            with cols[i]:
                st.color_picker(f"Cluster {cl}", color, key=f"color_{cl}", disabled=True, label_visibility="visible")

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    summary = compute_cluster_summary(df_f)
    st.dataframe(summary, use_container_width=True)

    if not summary.empty:
        # Gráficos
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(summary, x="n_points", y="cluster", orientation="h", title="Pontos por cluster",
                         color="cluster", color_discrete_map=CLUSTER_COLORS, height=420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(summary, x="radius_km", y="cluster", orientation="h", title="Raio p90 (km) por cluster",
                          color="cluster", color_discrete_map=CLUSTER_COLORS, height=420)
            st.plotly_chart(fig2, use_container_width=True)

        # Download CSV
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar resumo (CSV)", csv, file_name="cds_raios_resumo.csv", mime="text/csv")
    else:
        st.info("Sem dados para os filtros atuais.")
