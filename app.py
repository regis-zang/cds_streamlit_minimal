# app.py
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Pontos por cluster (cores distintas)",
    layout="wide",
    page_icon="🗺️",
)

DATA_FILE = "DataBase/points_enriched_final.parquet"

# Paleta fixa (RGBA) para 5 clusters: 0..4
PALETTE = {
    0: [230, 57, 70, 220],   # vermelho
    1: [46, 204, 113, 220],  # verde
    2: [66, 133, 244, 220],  # azul
    3: [155, 89, 182, 220],  # roxo
    4: [241, 196, 15, 220],  # amarelo
}
FALLBACK = [80, 80, 80, 180]  # caso apareça cluster fora do esperado


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_points(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # normaliza nomes esperados
    df = df.rename(columns={
        "lat": "latitude",
        "lng": "longitude",
        "lon": "longitude",
    })
    return df


def colorize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas [r,g,b,a] com base em 'cluster' para o ScatterplotLayer.
    """
    out = df.copy()
    cl = pd.to_numeric(out["cluster"], errors="coerce").astype("Int64")
    rgba = cl.map(PALETTE).fillna([FALLBACK]).astype(object)
    out[["r", "g", "b", "a"]] = pd.DataFrame(rgba.tolist(), index=out.index)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Distância Haversine em KM entre (lat1,lon1) e (lat2,lon2) – vetorizado.
    """
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2) - np.radians(lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, por cluster:
      - centróide (média de lat/lon)
      - raio p90 (km)
      - contagem de pontos
    """
    # centróides simples (média)
    cent = (
        df.groupby("cluster", as_index=False)
        .agg(centroid_lat=("latitude", "mean"),
             centroid_lon=("longitude", "mean"),
             n_points=("cluster", "size"))
    )

    # distância de cada ponto ao centróide do seu cluster
    base = df.merge(cent, on="cluster", how="left")
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"],
        base["centroid_lat"], base["centroid_lon"]
    )

    r90 = (
        base.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )

    out = cent.merge(r90, on="cluster", how="left")
    return out.sort_values("cluster").reset_index(drop=True)


def build_deck(
    df_points: pd.DataFrame,
    summary: pd.DataFrame,
    show_p90: bool,
    show_counts: bool,
) -> pdk.Deck:
    """
    Monta o mapa com:
      - TileLayer (base map)
      - pontos por cluster (cores RGBA)
      - círculos p90 (opcional)
      - contagem no centróide (opcional)
    """

    # --- camada de base (Carto Light) ---
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )

    # --- pontos coloridos por cluster (usa [r,g,b,a]) ---
    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_points,
        get_position='[longitude, latitude]',
        get_fill_color='[r, g, b, a]',
        get_radius=80,                 # em metros (aprox.)
        radius_min_pixels=3,
        radius_max_pixels=12,
        stroked=False,
        pickable=True,
    )

    layers = [base_layer, pts]

    # --- círculos p90 ---
    if show_p90 and not summary.empty:
        s = summary.copy()
        s["radius_m"] = s["radius_km"].fillna(0) * 1000.0
        rings = pdk.Layer(
            "ScatterplotLayer",
            data=s,
            get_position='[centroid_lon, centroid_lat]',
            get_radius="radius_m",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[63, 136, 248, 180],
            stroked=True,
            filled=False,
            line_width_min_pixels=2,
            pickable=False,
        )
        layers.append(rings)

    # --- contagem no centróide ---
    if show_counts and not summary.empty:
        labels = pdk.Layer(
            "TextLayer",
            data=summary,
            get_position='[centroid_lon, centroid_lat]',
            get_text="n_points",
            get_color=[40, 40, 40, 230],
            get_size=14,
            get_alignment_baseline="'center'",
            pickable=False,
        )
        layers.append(labels)

    # viewport: ajusta pro conjunto atual
    if len(df_points) > 0:
        v = pdk.data_utils.compute_view(
            df_points[["longitude", "latitude"]].rename(columns={
                "longitude": "lon", "latitude": "lat"
            })
        )
        v.pitch = 0
        v.bearing = 0
    else:
        v = pdk.ViewState(latitude=-22.0, longitude=-47.0, zoom=6)

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=v,
        map_provider=None,   # estamos usando TileLayer (não usar Mapbox)
        map_style=None,
        tooltip=tooltip,
    )
    return deck


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
st.title("Pontos por cluster (cores distintas)")

# Carrega dados
df_all = load_points(DATA_FILE)

# Filtros à esquerda
with st.sidebar:
    st.subheader("Filtros")
    clusters = sorted(pd.to_numeric(df_all["cluster"], errors="coerce").dropna().astype(int).unique().tolist())
    sel = st.multiselect("Clusters", clusters, default=clusters, label_visibility="collapsed")
    st.divider()
    show_legend = st.checkbox("Mostrar legenda de cores", value=True)
    show_p90 = st.checkbox("Mostrar círculos p90 por CD", value=True)
    show_counts = st.checkbox("Mostrar contagem no centróide", value=True)

# Aplica filtro
df_map = df_all[df_all["cluster"].isin(sel)].copy()

# Métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df_all):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
if len(df_map):
    c3.metric("Lat range", f"{df_map['latitude'].min():.5f} ~ {df_map['latitude'].max():.5f}")
    c4.metric("Lon range", f"{df_map['longitude'].min():.5f} ~ {df_map['longitude'].max():.5f}")
else:
    c3.metric("Lat range", "-")
    c4.metric("Lon range", "-")

# Abas
tab_map, tab_charts = st.tabs(["🗺️ Mapa", "📊 CDs & Raios"])

# ------------------------ Mapa ------------------------
with tab_map:
    st.subheader("Mapa por cluster")

    # cria colunas de cor (r,g,b,a)
    df_map = colorize(df_map)

    # resumo para p90/labels
    summary = compute_cluster_summary(df_map) if len(df_map) else pd.DataFrame(
        columns=["cluster", "centroid_lat", "centroid_lon", "n_points", "radius_km"]
    )

    deck = build_deck(
        df_points=df_map,
        summary=summary,
        show_p90=show_p90,
        show_counts=show_counts,
    )
    st.pydeck_chart(deck, use_container_width=True)

    # legenda simples
    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(len(PALETTE))
        for i, cl in enumerate(sorted(PALETTE.keys())):
            rgba = PALETTE[cl]
            swatch = f"background-color: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]/255:.2f}); width:14px; height:14px; display:inline-block; border-radius:50%; margin-right:6px;"
            with cols[i]:
                st.markdown(f"<span style='{swatch}'></span> Cluster {cl}", unsafe_allow_html=True)

# -------------------- Tabela & Gráficos --------------------
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")

    summary = compute_cluster_summary(df_map) if len(df_map) else pd.DataFrame(
        columns=["cluster", "centroid_lat", "centroid_lon", "n_points", "radius_km"]
    )

    st.dataframe(
        summary.rename(columns={
            "cluster": "cluster",
            "centroid_lat": "centroid_lat",
            "centroid_lon": "centroid_lon",
            "radius_km": "radius_km",
            "n_points": "n_points",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if len(summary):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                summary.sort_values("n_points"),
                x="n_points", y="cluster", orientation="h",
                labels={"n_points": "Pontos", "cluster": "Cluster"},
                title="Pontos por cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                summary.sort_values("radius_km"),
                x="radius_km", y="cluster", orientation="h",
                labels={"radius_km": "Raio p90 (km)", "cluster": "Cluster"},
                title="Raio p90 (km) por cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum cluster selecionado.")
