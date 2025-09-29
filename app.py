# app.py — CDS minimal (corrigido sem id no Deck)
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


# ===================== CONFIG =====================
st.set_page_config(page_title="CDs - Mapa e Raios", layout="wide")

DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"

MAP_SAMPLE_CAP = 8000          # limite de pontos renderizados no mapa
MARKER_PX = 6                  # tamanho do marcador
BASEMAP = "Carto Light"        # fixo conforme pedido

PALETTE = {
    0: [230, 57, 70, 200],     # vermelho
    1: [67, 170, 139, 200],    # verde
    2: [76, 114, 176, 200],    # azul
    3: [170, 90, 160, 200],    # roxo
    4: [246, 189, 22, 200],    # amarelo
}

# ===================== UTILS =====================
@st.cache_data(show_spinner=False)
def load_points(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    # Normaliza nomes de colunas de latitude/longitude
    lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("lon", "lng", "longitude")), None)
    if not lat_col or not lon_col:
        raise ValueError("Não encontrei colunas de latitude/longitude no parquet.")

    df = df.rename(columns={lat_col: "lat", lon_col: "lon"})
    # garante tipo de cluster
    if "cluster" not in df.columns:
        raise ValueError("Coluna 'cluster' não encontrada no parquet.")
    df["cluster"] = df["cluster"].astype(int)

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_cluster_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (summary, df_map)
      summary: por cluster -> centroid_lat, centroid_lon, radius_km (p90), n_points
      df_map:  df com colunas lat, lon, cluster e cores (rgba) para o mapa
    """
    # Centróides
    if {"centroid_lat", "centroid_lon"}.issubset(df.columns):
        centroids = (
            df[["cluster", "centroid_lat", "centroid_lon"]]
            .drop_duplicates(subset=["cluster"])
            .copy()
        )
    else:
        centroids = (
            df.groupby("cluster")[["lat", "lon"]]
            .mean()
            .rename(columns={"lat": "centroid_lat", "lon": "centroid_lon"})
            .reset_index()
        )

    tmp = df.merge(centroids, on="cluster", how="left")
    tmp["dist_km"] = haversine_km(tmp["lat"], tmp["lon"], tmp["centroid_lat"], tmp["centroid_lon"])

    r90 = (
        tmp.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )
    counts = tmp.groupby("cluster").size().reset_index(name="n_points")

    summary = (
        centroids.merge(r90, on="cluster").merge(counts, on="cluster").sort_values("cluster")
    )

    # Cores para o mapa
    tmp["rgba"] = tmp["cluster"].map(PALETTE).apply(lambda c: c if isinstance(c, list) else [0, 0, 0, 160])

    return summary, tmp


def make_deck(df_map: pd.DataFrame, summary: pd.DataFrame, show_circles: bool, show_counts: bool) -> pdk.Deck:
    """Cria o objeto pdk.Deck com TileLayer + pontos + círculos (opcional) + rótulos (opcional)."""
    layers = []

    # 1) Basemap como TileLayer (compatível com pydeck no Streamlit Cloud)
    if BASEMAP == "Carto Light":
        tile_url = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    else:
        tile_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

    layers.append(
        pdk.Layer(
            "TileLayer",
            data=tile_url,
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=1.0,
            id="base",
        )
    )

    # 2) Pontos
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position="[lon, lat]",
            get_fill_color="rgba",
            pickable=True,
            radius_units="pixels",
            get_radius=MARKER_PX,
            stroked=False,
            opacity=0.9,
            id="pts",
        )
    )

    # 3) Círculos p90 por CD (como discos)
    if show_circles:
        circ = summary.copy()
        circ["radius_m"] = (circ["radius_km"].fillna(0) * 1000).clip(lower=0)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=circ,
                get_position="[centroid_lon, centroid_lat]",
                get_fill_color="[64, 128, 255, 45]",
                get_radius="radius_m",
                radius_units="meters",
                stroked=True,
                get_line_color="[64, 128, 255, 160]",
                line_width_min_pixels=1,
                id="circles",
            )
        )

    # 4) Contagem no centróide
    if show_counts:
        labels = summary[["cluster", "centroid_lat", "centroid_lon", "n_points"]].copy()
        labels["txt"] = labels["n_points"].astype(str)
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=labels,
                get_position="[centroid_lon, centroid_lat]",
                get_text="txt",
                get_size=16,
                size_units="pixels",
                get_color="[255, 255, 255, 230]",
                get_text_anchor='"middle"',
                get_alignment_baseline='"center"',
                id="labels",
            )
        )

    # View inicial centralizada nos dados
    lat0 = float(df_map["lat"].mean())
    lon0 = float(df_map["lon"].mean())
    view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6.0, bearing=0, pitch=0)

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {lat}<br/><b>Lon:</b> {lon}",
        "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
    }

    # ⚠️ Sem `id=` no Deck (para evitar o erro). IDs estão nas camadas.
    deck = pdk.Deck(
        initial_view_state=view,
        layers=layers,
        map_style=None,  # porque estamos usando TileLayer
        parameters={"clearColor": [0.97, 0.97, 0.97, 1.0]},
        tooltip=tooltip,
    )
    return deck


# ===================== APP =====================
df_raw = load_points(POINTS_FILE)

# -------- Sidebar --------
st.sidebar.header("Filtros")
clusters_sorted = sorted(df_raw["cluster"].unique())
sel_clusters = st.sidebar.multiselect(
    "Clusters", clusters_sorted, default=clusters_sorted, label_visibility="collapsed"
)

show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_circles = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# Filtra por cluster
df = df_raw[df_raw["cluster"].isin(sel_clusters)].copy()

# Amostra só para o mapa
if len(df) > MAP_SAMPLE_CAP:
    df_map_sample = df.sample(MAP_SAMPLE_CAP, random_state=42).copy()
else:
    df_map_sample = df.copy()

summary, df_map = compute_cluster_summary(df_map_sample)

# -------- Métricas de topo --------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df_raw):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map_sample):,}".replace(",", "."))
lat_min, lat_max = df["lat"].min(), df["lat"].max()
lon_min, lon_max = df["lon"].min(), df["lon"].max()
c3.metric("Lat range", f"{lat_min:.5f} ~ {lat_max:.5f}")
c4.metric("Lon range", f"{lon_min:.5f} ~ {lon_max:.5f}")

# -------- Tabs --------
tab_map, tab_charts = st.tabs(["🗺️ Mapa", "📊 CDs & Raios"])

with tab_map:
    st.subheader("Mapa por cluster")

    deck = make_deck(df_map, summary, show_circles=show_circles, show_counts=show_counts)
    # sem `key` aqui; usamos IDs nas camadas
    st.pydeck_chart(deck, use_container_width=True, height=650)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(min(5, len(clusters_sorted)))
        for i, cl in enumerate(sorted(df["cluster"].unique())):
            color = PALETTE.get(int(cl), [120, 120, 120, 200])
            rgba_css = f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255:.2f})"
            with cols[i % len(cols)]:
                st.markdown(
                    f"""
                    <div style='display:flex;align-items:center;gap:8px;margin:6px 0;'>
                        <span style='width:14px;height:14px;border-radius:50%;background:{rgba_css};display:inline-block;'></span>
                        <span>Cluster {cl}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    st.dataframe(
        summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]],
        use_container_width=True,
        height=260,
    )

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar resumo (CSV)", csv, "resumo_cds.csv", "text/csv")

    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.bar(
            summary.sort_values("n_points", ascending=True),
            x="n_points",
            y="cluster",
            orientation="h",
            title="Pontos por cluster",
            labels={"n_points": "n pontos", "cluster": "cluster"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.bar(
            summary.sort_values("radius_km", ascending=True),
            x="radius_km",
            y="cluster",
            orientation="h",
            title="Raio p90 (km) por cluster",
            labels={"radius_km": "raio p90 (km)", "cluster": "cluster"},
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
        st.plotly_chart(fig2, use_container_width=True)
