# app.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

# ==============================
# Configuração da página
# ==============================
st.set_page_config(
    page_title="CDs – Mapa e Raios (p90)",
    layout="wide",
    page_icon="🗺️",
)

# ==============================
# Parâmetros / Constantes
# ==============================
DATA_FILE = os.getenv("POINTS_PARQUET", "DataBase/points_enriched_final.parquet")
POINT_MARKER_PX = 6  # tamanho do marcador (px)
MAP_HEIGHT = 620

# Paleta (RGB) por cluster
COLOR_PALETTE = {
    0: [230, 57, 70],   # vermelho
    1: [33, 158, 188],  # teal
    2: [59, 130, 246],  # azul
    3: [168, 85, 247],  # púrpura
    4: [234, 179, 8],   # amarelo
}
DEFAULT_RGB = [180, 180, 180]


# ==============================
# Funções Utilitárias
# ==============================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normaliza nomes esperados
    df = df.rename(columns={
        "lat": "latitude",
        "lon": "longitude",
        "long": "longitude"
    })
    required = {"latitude", "longitude", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Colunas ausentes no parquet: {missing}")
    return df


def make_palette(df: pd.DataFrame) -> dict:
    """Devolve dicionário cluster -> rgba (4 canais)."""
    uniq = sorted(df["cluster"].astype(int).unique().tolist())
    palette = {}
    for c in uniq:
        rgb = COLOR_PALETTE.get(int(c), DEFAULT_RGB)
        palette[int(c)] = rgb + [190]  # alpha ~ 0.75
    return palette


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine (km). Entende pandas Series/arrays."""
    R = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saída:
      cluster, centroid_lat, centroid_lon, radius_km (p90), n_points, radius_m
    Evita KeyError renomeando colunas já no groupby.
    """
    # centróides (mediana) com nomes finais
    cent = (
        df.groupby("cluster", as_index=False)
          .agg(centroid_lat=("latitude", "median"),
               centroid_lon=("longitude", "median"))
    )

    # calcula distância de cada ponto ao seu centróide
    base = df.merge(cent, on="cluster", how="left")
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"], base["centroid_lat"], base["centroid_lon"]
    )

    # raio p90 por cluster + contagem
    r90 = base.groupby("cluster", as_index=False)["dist_km"].quantile(0.90)
    r90 = r90.rename(columns={"dist_km": "radius_km"})
    cnt = base.groupby("cluster", as_index=False).size().rename(columns={"size": "n_points"})

    summary = (cent.merge(r90, on="cluster")
                    .merge(cnt, on="cluster"))
    summary["radius_m"] = (summary["radius_km"] * 1000.0).astype(float)
    return summary


def calc_view_state(df: pd.DataFrame) -> pdk.ViewState:
    lat_min, lat_max = float(df["latitude"].min()), float(df["latitude"].max())
    lon_min, lon_max = float(df["longitude"].min()), float(df["longitude"].max())
    lat_c = (lat_min + lat_max) / 2.0
    lon_c = (lon_min + lon_max) / 2.0
    # heurística simples de zoom
    span = max(lat_max - lat_min, lon_max - lon_min, 0.5)
    zoom = float(np.clip(9.0 - np.log2(span), 2, 12))
    return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, bearing=0, pitch=0)


def make_deck(
    df_map: pd.DataFrame,
    show_p90: bool,
    show_counts: bool,
) -> pdk.Deck:
    """
    Constrói o objeto Deck com:
      - TileLayer base
      - Scatterplot dos pontos
      - (opcional) círculos p90 por cluster
      - (opcional) contagem no centróide (TextLayer)
    """
    palette = make_palette(df_map)
    df_map = df_map.copy()
    df_map["rgba"] = df_map["cluster"].astype(int).map(palette)
    # Deck espera lista [r,g,b,a]; se houver cluster fora, cai no default.
    df_map["rgba"] = df_map["rgba"].apply(lambda v: v if isinstance(v, list) else DEFAULT_RGB + [190])

    # Basemap compatível (Carto Light)
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
    )

    # Pontos
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        get_radius=POINT_MARKER_PX,
        pickable=True,
        stroked=False,
        radius_units="pixels",
    )

    layers = [base_layer, pts]

    # Resumo (p/ círculos p90 e contagem)
    summary = compute_cluster_summary(df_map)

    if show_p90 and not summary.empty:
        # círculos p90 (borda)
        circ = summary.copy()
        circ["rgba_ring"] = circ["cluster"].astype(int).map(
            {k: v[:3] + [140] for k, v in palette.items()}
        )
        rings = pdk.Layer(
            "ScatterplotLayer",
            data=circ,
            get_position="[centroid_lon, centroid_lat]",
            get_radius="radius_m",
            radius_units="meters",
            stroked=True,
            filled=False,
            get_line_color="rgba_ring",
            line_width_min_pixels=2,
        )
        layers.append(rings)

    if show_counts and not summary.empty:
        txt = summary.copy()
        txt["label"] = txt["n_points"].astype(str)
        txt_layer = pdk.Layer(
            "TextLayer",
            data=txt,
            get_position="[centroid_lon, centroid_lat]",
            get_text="label",
            get_color=[30, 30, 30, 220],
            get_size=14,
            size_units="pixels",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
        )
        layers.append(txt_layer)

    deck = pdk.Deck(
        map_style=None,  # usando TileLayer
        layers=layers,
        initial_view_state=calc_view_state(df_map),
        tooltip={
            "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
            "style": {"backgroundColor": "white", "color": "black"},
        },
    )
    return deck


def legend_html(palette: dict) -> str:
    items = []
    for k in sorted(palette.keys()):
        r, g, b, *_ = palette[k]
        items.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:14px'>"
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:7px;"
            f"background: rgb({r},{g},{b});margin-right:6px;border:1px solid rgba(0,0,0,.25)'></span>"
            f"Cluster {k}</span>"
        )
    return "<div style='padding-top:6px'>" + "".join(items) + "</div>"


# ==============================
# App
# ==============================
st.title("Pontos por cluster (cores distintas)")

try:
    df_all = load_data(DATA_FILE)
except Exception as e:
    st.error(f"Falha ao carregar o parquet '{DATA_FILE}'. Detalhes: {e}")
    st.stop()

# Filtros
with st.sidebar:
    st.header("Filtros")
    clusters = sorted(df_all["cluster"].astype(int).unique().tolist())
    # picker de clusters (chips)
    cols = st.columns(6)
    selected = st.session_state.get("sel_clusters", set(clusters))
    # desenha os chips
    new_selected = set(selected)
    for i, c in enumerate(clusters):
        if cols[i % 6].checkbox(f"{c}", value=(c in selected)):
            new_selected.add(c)
        else:
            new_selected.discard(c)
    selected = sorted(new_selected)
    st.session_state["sel_clusters"] = set(selected)

    st.markdown("---")
    show_legend = st.checkbox("Mostrar legenda de cores", value=True)
    show_p90 = st.checkbox("Mostrar círculos p90 por CD", value=True)
    show_counts = st.checkbox("Mostrar contagem no centróide", value=True)

# Aplica filtro
df_map = df_all[df_all["cluster"].astype(int).isin(selected)].copy()
df_map.reset_index(drop=True, inplace=True)

# Métricas do topo
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df_all):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
if not df_map.empty:
    lat_rng = f"{df_map['latitude'].min():.5f} ~ {df_map['latitude'].max():.5f}"
    lon_rng = f"{df_map['longitude'].min():.5f} ~ {df_map['longitude'].max():.5f}"
else:
    lat_rng = lon_rng = "–"
c3.metric("Lat range", lat_rng)
c4.metric("Lon range", lon_rng)

# Tabs
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

# ------------------ Mapa ------------------
with tab_map:
    st.subheader("Mapa por cluster")

    if df_map.empty:
        st.info("Nenhum ponto para os clusters selecionados.")
    else:
        deck = make_deck(df_map, show_p90=show_p90, show_counts=show_counts)
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)

        if show_legend:
            pal = make_palette(df_map)
            st.markdown(legend_html(pal), unsafe_allow_html=True)

# ------------- Tabela & Gráficos ----------
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")

    if df_map.empty:
        st.info("Nenhum ponto para resumir.")
    else:
        summary = compute_cluster_summary(df_map).sort_values("cluster").reset_index(drop=True)
        st.dataframe(
            summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]],
            use_container_width=True,
            height=260,
        )

        cA, cB = st.columns(2)
        with cA:
            fig1 = px.bar(
                summary.sort_values("n_points"),
                x="n_points", y="cluster", orientation="h",
                title="Pontos por cluster",
                labels={"n_points": "Pontos", "cluster": "cluster"},
                height=360,
            )
            fig1.update_layout(yaxis=dict(type="category"))
            st.plotly_chart(fig1, use_container_width=True, height=360)

        with cB:
            fig2 = px.bar(
                summary.sort_values("radius_km"),
                x="radius_km", y="cluster", orientation="h",
                title="Raio p90 (km) por cluster",
                labels={"radius_km": "Raio p90 (km)", "cluster": "cluster"},
                height=360,
            )
            fig2.update_layout(yaxis=dict(type="category"))
            st.plotly_chart(fig2, use_container_width=True, height=360)

        # Download do resumo
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resumo (CSV)",
            data=csv,
            file_name="cds_resumo_clusters.csv",
            mime="text/csv",
            use_container_width=False,
        )
