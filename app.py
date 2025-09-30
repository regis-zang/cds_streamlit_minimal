# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

# ------------------------------------------------------
# Config
# ------------------------------------------------------
st.set_page_config(page_title="Pontos por cluster (cores distintas)", layout="wide")
DATA_FILE = "DataBase/points_enriched_final.parquet"

CLUSTERS_ALL = [0, 1, 2, 3, 4]
PALETTE = {
    0: [234, 67, 53],    # vermelho
    1: [52, 168, 83],    # verde
    2: [66, 133, 244],   # azul
    3: [156, 39, 176],   # roxo
    4: [251, 188, 5],    # amarelo
}

# ------------------------------------------------------
# Utilitários
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_points(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # fallback para execução local
        path = os.path.basename(path)
    df = pd.read_parquet(path)
    # Normaliza nomes esperados
    cols = {c.lower(): c for c in df.columns}
    for col_needed in ["latitude", "longitude", "cluster"]:
        if col_needed not in df.columns:
            # tenta achar equivalente ignorando caixa
            for k, v in cols.items():
                if k == col_needed:
                    df.rename(columns={v: col_needed}, inplace=True)
                    break
    # Tipos
    df["cluster"] = df["cluster"].astype(int)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # raio médio da terra
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat/2.0)**2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    )
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna: cluster, centroid_lat, centroid_lon, radius_km (p90) e n_points.
    Garante colunas 'centroid_lat' e 'centroid_lon' (rename após groupby).
    """
    if df.empty:
        return pd.DataFrame(
            columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]
        )

    # centróides por média
    cent = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .mean()
        .reset_index()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
    )

    # junta para calcular distância ao centróide
    base = df.merge(cent, on="cluster", how="left")

    # distância em km de cada ponto ao centróide do seu cluster
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"], base["centroid_lat"], base["centroid_lon"]
    )

    # p90 da distância por cluster
    r90 = (
        base.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )

    # contagem por cluster
    npts = (
        df.groupby("cluster", as_index=False)
        .size()
        .rename(columns={"size": "n_points"})
    )

    # consolida
    summary = (
        cent.merge(r90, on="cluster", how="left")
        .merge(npts, on="cluster", how="left")
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    # segurança contra NaN
    for c in ["radius_km", "n_points"]:
        if c in summary:
            summary[c] = summary[c].fillna(0)

    return summary

def make_deck(df_map: pd.DataFrame, show_p90: bool, show_counts: bool) -> pdk.Deck:
    """
    Monta a visualização (TileLayer + ScatterplotLayer + opcional CircleLayer e TextLayer).
    """
    if df_map.empty:
        # view default no sudeste
        view_state = pdk.ViewState(latitude=-22.5, longitude=-47, zoom=6)
        return pdk.Deck(layers=[], initial_view_state=view_state, map_provider=None)

    df_map = df_map.copy()
    df_map["rgba"] = df_map["cluster"].map(PALETTE)

    # View inicial
    lat_c = df_map["latitude"].mean()
    lon_c = df_map["longitude"].mean()
    view_state = pdk.ViewState(latitude=float(lat_c), longitude=float(lon_c), zoom=6)

    layers = []

    # Base map (Carto Light)
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
    )
    layers.append(base_layer)

    # Pontos
    pts_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["longitude", "latitude"],
        get_fill_color="rgba",
        get_radius=60,
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=12,
        stroked=True,
        get_line_color=[240, 240, 240],
        line_width_min_pixels=0.5,
    )
    layers.append(pts_layer)

    # Resumo para p90 / contagens
    summary = compute_cluster_summary(df_map)

    # Círculos p90
    if show_p90 and not summary.empty:
        # CircleLayer usa metros
        summary["_radius_m"] = summary["radius_km"] * 1000.0
        rings = pdk.Layer(
            "CircleLayer",
            data=summary,
            get_position=["centroid_lon", "centroid_lat"],
            get_radius="_radius_m",
            get_fill_color=[33, 150, 243, 40],
            get_line_color=[33, 150, 243, 180],
            line_width_min_pixels=1.5,
            stroked=True,
            pickable=False,
        )
        layers.append(rings)

    # Contagem no centróide
    if show_counts and not summary.empty:
        summary["_label"] = summary["n_points"].astype(int).astype(str)
        txt = pdk.Layer(
            "TextLayer",
            data=summary,
            get_position=["centroid_lon", "centroid_lat"],
            get_text="_label",
            get_color=[20, 20, 20],
            get_size=16,
            size_units="meters",
            size_scale=20,
            size_min_pixels=10,
            get_alignment_baseline="'center'",
        )
        layers.append(txt)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider=None,
        tooltip={"text": "Cluster {cluster}\nlat {latitude}\nlon {longitude}"},
    )
    return deck

# ------------------------------------------------------
# UI – Sidebar (filtros)
# ------------------------------------------------------
with st.sidebar:
    st.header("Filtros")
    df = load_points(DATA_FILE)

    # chips de clusters
    uniq_clusters = sorted(df["cluster"].unique().tolist())
    # mostra todos por padrão
    cols = st.columns(len(uniq_clusters))
    selected = []
    for i, c in enumerate(uniq_clusters):
        checked = cols[i].checkbox(str(c), value=True)
        if checked:
            selected.append(c)

    # toggles
    st.markdown("---")
    show_legend = st.checkbox("Mostrar legenda de cores", value=True)
    show_p90 = st.checkbox("Mostrar círculos p90 por CD", value=True)
    show_counts = st.checkbox("Mostrar contagem no centróide", value=True)

# ------------------------------------------------------
# Filtragem e métricas
# ------------------------------------------------------
df_map = df[df["cluster"].isin(selected)].copy()

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
if not df_map.empty:
    c3.metric("Lat range", f"{df_map['latitude'].min():.5f} ~ {df_map['latitude'].max():.5f}")
    c4.metric("Lon range", f"{df_map['longitude'].min():.5f} ~ {df_map['longitude'].max():.5f}")
else:
    c3.metric("Lat range", "-")
    c4.metric("Lon range", "-")

# ------------------------------------------------------
# Abas
# ------------------------------------------------------
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

with tab_map:
    st.subheader("Mapa por cluster")
    deck = make_deck(df_map, show_p90=show_p90, show_counts=show_counts)
    # use_container_width removido para evitar warnings futuros do Streamlit
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(len(PALETTE))
        for i, c in enumerate(sorted(PALETTE)):
            color = PALETTE[c]
            cols[i].markdown(
                f"<div style='display:inline-block;width:12px;height:12px;"
                f"background:rgba({color[0]},{color[1]},{color[2]},1);"
                f"border-radius:50%;margin-right:6px;'></div> Cluster {c}",
                unsafe_allow_html=True,
            )

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    summary = compute_cluster_summary(df_map)

    st.dataframe(
        summary.rename(columns={
            "centroid_lat": "centroid_lat",
            "centroid_lon": "centroid_lon",
            "radius_km": "radius_km",
            "n_points": "n_points"
        }),
        use_container_width=True,
        height=270
    )

    # Gráficos
    cc1, cc2 = st.columns(2)
    if not summary.empty:
        fig1 = px.bar(
            summary.sort_values("n_points", ascending=True),
            x="n_points", y="cluster", orientation="h",
            title="Pontos por cluster",
            labels={"n_points": "nº pontos", "cluster": "cluster"},
        )
        cc1.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(
            summary.sort_values("radius_km", ascending=True),
            x="radius_km", y="cluster", orientation="h",
            title="Raio p90 (km) por cluster",
            labels={"radius_km": "km (p90)", "cluster": "cluster"},
        )
        cc2.plotly_chart(fig2, use_container_width=True)
    else:
        cc1.info("Sem dados para o filtro atual.")
        cc2.info("Sem dados para o filtro atual.")
