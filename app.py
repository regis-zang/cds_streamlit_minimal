# app.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

# =========================================
# Config
# =========================================
st.set_page_config(page_title="CDs - Mapa e Raios", layout="wide")
st.markdown(
    """
    <style>
    /* tira excesso de padding do cabeçalho de abas */
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# caminhos/colunas
DATA_DIR = Path("DataBase")
PARQUET_FILE = DATA_DIR / "points_enriched_final.parquet"

COL_LAT = "latitude"
COL_LON = "longitude"
COL_CLUSTER = "cluster"

# paleta para os clusters (pode estender)
PALETTE = {
    0: [230, 57, 70],     # vermelho
    1: [53, 183, 41],     # verde
    2: [66, 135, 245],    # azul
    3: [162, 99, 235],    # roxo
    4: [241, 196, 15],    # amarelo
    5: [255, 127, 80],
    6: [0, 191, 255],
    7: [255, 99, 132],
}

# =========================================
# Helpers
# =========================================
@st.cache_data(show_spinner=False)
def load_points() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_FILE)
    # garante tipos
    df = df.rename(columns={COL_LAT: "latitude", COL_LON: "longitude", COL_CLUSTER: "cluster"})
    df["cluster"] = df["cluster"].astype(int)
    return df[["latitude", "longitude", "cluster"]].copy()

def haversine_km(lat1, lon1, lat2, lon2):
    """distância haversine em km."""
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """centroide + raio p90 + contagem por cluster."""
    base = df.groupby("cluster")[["latitude", "longitude"]].mean().reset_index()
    base = base.rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})

    # distância de cada ponto ao centróide do seu cluster
    tmp = df.merge(base, on="cluster", how="left")
    tmp["dist_km"] = haversine_km(
        tmp["latitude"].to_numpy(),
        tmp["longitude"].to_numpy(),
        tmp["centroid_lat"].to_numpy(),
        tmp["centroid_lon"].to_numpy(),
    )

    r90 = tmp.groupby("cluster", as_index=False)["dist_km"].quantile(0.90)
    r90 = r90.rename(columns={"dist_km": "radius_km"})

    sizes = df.groupby("cluster").size().rename("n_points").reset_index()

    resumo = (
        base.merge(r90, on="cluster", how="left")
            .merge(sizes, on="cluster", how="left")
            .sort_values("cluster")
            .reset_index(drop=True)
    )
    return resumo[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]]

def color_for_cluster(c: int):
    return PALETTE.get(int(c), [120, 120, 120])

def legend_html(selected_clusters):
    chips = []
    for c in selected_clusters:
        r, g, b = color_for_cluster(c)
        chips.append(
            f"""<span style="display:inline-flex;align-items:center;margin-right:.8rem">
                 <span style="width:12px;height:12px;border-radius:50%;background:rgb({r},{g},{b});display:inline-block;margin-right:.4rem"></span>
                 Cluster {c}
               </span>"""
        )
    return " ".join(chips)

def next_map_key():
    """chave única para o componente de mapa (evita erro ao trocar de aba)."""
    k = st.session_state.get("_map_key", 0) + 1
    st.session_state["_map_key"] = k
    return f"map_{k}"

# =========================================
# Dados
# =========================================
if not PARQUET_FILE.exists():
    st.error(f"Arquivo não encontrado: `{PARQUET_FILE}`")
    st.stop()

df_all = load_points()

# =========================================
# Sidebar (filtros)
# =========================================
st.sidebar.header("Filtros")
all_clusters = sorted(df_all["cluster"].unique().tolist())
selected = st.sidebar.multiselect(
    "Clusters", all_clusters, default=all_clusters, format_func=lambda x: str(x)
)

show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_circles = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# =========================================
# Filtragem + KPIs
# =========================================
df = df_all[df_all["cluster"].isin(selected)].copy()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total de pontos (dados)", f"{len(df_all):,}".replace(",", "."))
col_b.metric("Pontos no mapa (amostra)", f"{len(df):,}".replace(",", "."))

if len(df) > 0:
    lat_rng = (df["latitude"].min(), df["latitude"].max())
    lon_rng = (df["longitude"].min(), df["longitude"].max())
    col_c.metric("Lat range", f"{lat_rng[0]:.3f} ~ {lat_rng[1]:.3f}")
    col_d.metric("Lon range", f"{lon_rng[0]:.3f} ~ {lon_rng[1]:.3f}")
else:
    col_c.metric("Lat range", "-")
    col_d.metric("Lon range", "-")

# =========================================
# Abas
# =========================================
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

# =========================================
# ABA 1 - MAPA
# =========================================
with tab_map:
    st.subheader("Mapa por cluster")

    if len(df) == 0:
        st.info("Nenhum cluster selecionado.")
        st.stop()

    # Camada base (TileLayer)
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",  # fundo claro
        min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
    )

    # PONTOS
    df["rgba"] = df["cluster"].map(color_for_cluster)
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        get_radius=80,                      # pixels -> usaremos meters, ver abaixo
        radius_units="pixels",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=0.5,
    )

    # RESUMO PARA CÍRCULOS E CONTAGENS
    resumo = compute_cluster_summary(df)
    resumo["rgba"] = resumo["cluster"].map(color_for_cluster)

    # CÍRCULOS p90 (em metros)
    layers = [base_layer, points_layer]
    if show_circles:
        circles = pdk.Layer(
            "ScatterplotLayer",
            data=resumo,
            get_position="[centroid_lon, centroid_lat]",
            get_radius="radius_km * 1000",
            radius_units="meters",
            stroked=True,
            filled=False,
            get_line_color="rgba",
            line_width_min_pixels=2,
        )
        layers.append(circles)

    # CONTAGEM (texto) no centróide
    if show_counts:
        text = pdk.Layer(
            "TextLayer",
            data=resumo.assign(label=resumo["n_points"].astype(str)),
            get_position="[centroid_lon, centroid_lat]",
            get_text="label",
            get_color=[255, 255, 255],
            get_size=16,
            size_units="pixels",
            get_alignment_baseline="'bottom'",
            background=True,
            background_color=[0, 0, 0, 160],
        )
        layers.append(text)

    # View: centraliza sobre os dados
    view = pdk.ViewState(
        latitude=float(df["latitude"].mean()),
        longitude=float(df["longitude"].mean()),
        zoom=7,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
        "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"}
    }

    deck = pdk.Deck(
        initial_view_state=view,
        layers=layers,
        map_style=None,  # <-- sem estilo Mapbox (evita fundo preto com TileLayer)
        parameters={"clearColor": [0.97, 0.97, 0.97, 1.0]},  # fundo clarinho
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True, key=next_map_key())

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        st.markdown(legend_html(sorted(resumo["cluster"].tolist())), unsafe_allow_html=True)

# =========================================
# ABA 2 - CDs & RAIOS
# =========================================
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    resumo = compute_cluster_summary(df)

    # tabela
    st.dataframe(
        resumo.style.format({"centroid_lat": "{:.4f}", "centroid_lon": "{:.4f}", "radius_km": "{:.4f}"}),
        use_container_width=True, height=250,
    )

    # download
    csv = resumo.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar resumo (CSV)",
        data=csv,
        file_name="cds_resumo_clusters.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # gráficos
    left, right = st.columns(2)
    with left:
        st.subheader("Pontos por cluster")
        chart1 = (
            alt.Chart(resumo)
            .mark_bar()
            .encode(
                x=alt.X("n_points:Q", title="n pontos"),
                y=alt.Y("cluster:O", sort="x", title="cluster"),
                color=alt.Color("cluster:N", legend=None),
                tooltip=["cluster", "n_points"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart1, use_container_width=True)

    with right:
        st.subheader("Raio p90 (km) por cluster")
        chart2 = (
            alt.Chart(resumo)
            .mark_bar(color="#f2994a")
            .encode(
                x=alt.X("radius_km:Q", title="raio p90 (km)"),
                y=alt.Y("cluster:O", sort="x", title="cluster"),
                tooltip=["cluster", alt.Tooltip("radius_km:Q", format=".2f")]
            )
            .properties(height=320)
        )
        st.altair_chart(chart2, use_container_width=True)
