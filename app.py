# app.py
from __future__ import annotations

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

# Paleta de cores (cicla caso existam mais clusters)
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

    return df.reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância Haversine em KM."""
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p) * np.cos(lat2*p) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _ensure_centroid_columns_after_merge(base: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que, após o merge, existam as colunas:
      - centroid_lat
      - centroid_lon
    e que as colunas de ponto sejam:
      - latitude
      - longitude

    Faz mapeamentos robustos caso o Pandas tenha criado sufixos
    como latitude_x / latitude_y, latitude_centroid etc.
    """
    b = base.copy()

    # 1) Normaliza latitude/longitude do ponto (lado esquerdo)
    #    Preferências: 'latitude'/'longitude' -> 'latitude_x'/'longitude_x'
    if "latitude" not in b.columns and "latitude_x" in b.columns:
        b = b.rename(columns={"latitude_x": "latitude"})
    if "longitude" not in b.columns and "longitude_x" in b.columns:
        b = b.rename(columns={"longitude_x": "longitude"})

    # 2) Cria centroid_lat/centroid_lon a partir das alternativas existentes
    if "centroid_lat" not in b.columns:
        if "latitude_y" in b.columns:  # caso o rename não tenha acontecido
            b = b.rename(columns={"latitude_y": "centroid_lat"})
        elif "latitude_centroid" in b.columns:
            b = b.rename(columns={"latitude_centroid": "centroid_lat"})
    if "centroid_lon" not in b.columns:
        if "longitude_y" in b.columns:
            b = b.rename(columns={"longitude_y": "centroid_lon"})
        elif "longitude_centroid" in b.columns:
            b = b.rename(columns={"longitude_centroid": "centroid_lon"})

    # 3) Se ainda assim não existirem, tenta detectar colunas candidatas por sufixo
    #    (último fallback)
    if "centroid_lat" not in b.columns or "centroid_lon" not in b.columns:
        # Procura por qualquer coluna que tenha "lat" e não seja "latitude"
        lat_cands = [c for c in b.columns if "lat" in c.lower() and c != "latitude"]
        lon_cands = [c for c in b.columns if "lon" in c.lower() and c != "longitude"]
        if lat_cands and "centroid_lat" not in b.columns:
            b = b.rename(columns={lat_cands[0]: "centroid_lat"})
        if lon_cands and "centroid_lon" not in b.columns:
            b = b.rename(columns={lon_cands[0]: "centroid_lon"})

    # 4) Confere se ficou tudo certo
    needed = {"latitude", "longitude", "centroid_lat", "centroid_lon"}
    missing = needed - set(b.columns)
    if missing:
        raise KeyError(
            f"Após o merge, ainda faltam colunas: {missing}. "
            f"Colunas disponíveis: {list(b.columns)}"
        )

    return b


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula centróides, raio p90 e contagem por cluster (robusto a sufixos)."""
    # Centróide (média)
    cent = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    # Garante nomes-friendly no centróide
    cent = cent.rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})

    # Merge com robustez a sufixos
    base = df.merge(cent, on="cluster", how="left", suffixes=("", "_centroid"))
    base = _ensure_centroid_columns_after_merge(base)

    # Distâncias ao centróide
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"],
        base["centroid_lat"], base["centroid_lon"],
    )

    # p90 e contagem
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

    # Basemap (CARTO Light)
    layers.append(
        pdk.Layer(
            "TileLayer",
            data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=1.0,
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
            stroked=False,
            opacity=0.9,
        )
    )

    # Círculos p90
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
        map_style=None,  # evita fundo preto
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
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df):,}".replace(",", "."))
c3.metric("Lat range", f"{df['latitude'].min():.3f} ~ {df['latitude'].max():.3f}")
c4.metric("Lon range", f"{df['longitude'].min():.3f} ~ {df['longitude'].max():.3f}")

tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

# ====== Mapa
with tab_map:
    st.subheader("Mapa por cluster")
    view = pdk.ViewState(
        latitude=float(df["latitude"].mean()),
        longitude=float(df["longitude"].mean()),
        zoom=6, pitch=0, bearing=0,
    )
    deck = make_deck(df, show_r90=show_r90, show_counts=show_counts, view_state=view)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        st.markdown(legend_markdown(df), unsafe_allow_html=True)

# ====== Tabelas/Gráficos
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    summary = compute_cluster_summary(df)
    st.dataframe(summary, use_container_width=True)

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar resumo (CSV)", data=csv_bytes, file_name="cds_raios_resumo.csv", mime="text/csv")

    g1, g2 = st.columns(2)
    with g1:
        fig = px.bar(
            summary.sort_values("n_points", ascending=True),
            x="n_points", y="cluster", orientation="h",
            color="cluster",
            color_discrete_sequence=[f"rgb{color_for_cluster(c)}" for c in summary["cluster"]],
            title="Pontos por cluster",
        )
        fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        fig2 = px.bar(
            summary.sort_values("radius_km", ascending=True),
            x="radius_km", y="cluster", orientation="h",
            color="cluster",
            color_discrete_sequence=[f"rgb{color_for_cluster(c)}" for c in summary["cluster"]],
            title="Raio p90 (km) por cluster",
        )
        fig2.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)
