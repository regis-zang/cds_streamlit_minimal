# app.py
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt
import streamlit.components.v1 as components


# =========================
# Config
# =========================
st.set_page_config(page_title="CDs • Mapa & Raios", layout="wide", page_icon="🗺️")

DATA_DIR = Path("DataBase")
POINTS_FILE = DATA_DIR / "points_enriched_final.parquet"

# Parâmetros fixos (removemos sliders/selects)
MAX_POINTS_DEFAULT = 8000
POINT_SIZE_DEFAULT = 6
BASE_MAP_DEFAULT = "Carto Light"  # "Carto Light" | "OSM Standard"


# =========================
# Utilitários
# =========================
@st.cache_data(show_spinner=False)
def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normaliza nomes mais comuns
    cols = {c.lower(): c for c in df.columns}
    for req in ["latitude", "longitude", "cluster"]:
        if req not in cols:
            # tenta procurar variações
            cand = [c for c in df.columns if c.lower().startswith(req)]
            if cand:
                df = df.rename(columns={cand[0]: req})
            else:
                raise ValueError(f"Coluna obrigatória ausente: {req}")
    # força tipos
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # cluster como int quando possível
    try:
        df["cluster"] = df["cluster"].astype("Int64")
    except Exception:
        pass
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # km
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = p2 - p1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_view(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=-23.5, longitude=-46.6, zoom=5.0, pitch=0)
    lat = float(df["latitude"].mean())
    lon = float(df["longitude"].mean())
    # zoom aproximado pela dispersão
    lat_span = max(0.1, float(df["latitude"].max() - df["latitude"].min()))
    lon_span = max(0.1, float(df["longitude"].max() - df["longitude"].min()))
    span = max(lat_span, lon_span)
    # aproximação de zoom
    zoom = max(3, min(12, 8 - math.log(span + 1e-9, 2)))
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0)


def make_palette(ordered_clusters):
    # cores estáveis por cluster (0..9)
    base = [
        [239, 83, 80, 220],    # 0 - red
        [76, 175, 80, 220],    # 1 - green
        [66, 165, 245, 220],   # 2 - blue
        [171, 71, 188, 220],   # 3 - purple
        [255, 235, 59, 220],   # 4 - yellow
        [0, 188, 212, 220],    # 5 - cyan
        [255, 152, 0, 220],    # 6 - orange
        [121, 85, 72, 220],    # 7 - brown
        [158, 158, 158, 220],  # 8 - grey
        [0, 150, 136, 220],    # 9 - teal
    ]
    pal = {}
    for i, c in enumerate(sorted(ordered_clusters)):
        pal[int(c)] = base[i % len(base)]
    return pal


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centróide (média), raio p90 (km) e nº de pontos por cluster.
    Robusta a colisões de nomes (centroid_* já existentes no Parquet).
    """
    empty_cols = ["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points", "radius_m"]
    if "cluster" not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    base = df.dropna(subset=["cluster"]).copy()
    if base.empty:
        return pd.DataFrame(columns=empty_cols)

    base["cluster"] = base["cluster"].astype(int)

    # remove colunas antigas para evitar sufixos no merge
    drop_cols = [c for c in ["centroid_lat", "centroid_lon", "radius_km", "radius_m", "n_points", "dist_km"] if c in base.columns]
    if drop_cols:
        base = base.drop(columns=drop_cols)

    cent = (base.groupby("cluster", as_index=False)
                 .agg(centroid_lat=("latitude", "mean"),
                      centroid_lon=("longitude", "mean")))

    tmp = base.merge(cent, on="cluster", how="left")

    # identifica nomes reais pós-merge (se houver sufixos)
    def pick_col(prefix: str) -> str:
        if prefix in tmp.columns:
            return prefix
        cand = [c for c in tmp.columns if c.startswith(prefix)]
        if cand:
            return cand[0]
        raise KeyError(prefix)

    latc = pick_col("centroid_lat")
    lonc = pick_col("centroid_lon")

    # distância até o centróide
    tmp["dist_km"] = haversine_km(tmp["latitude"], tmp["longitude"], tmp[latc], tmp[lonc])

    r90 = (tmp.groupby("cluster", as_index=False)["dist_km"]
             .quantile(0.90)
             .rename(columns={"dist_km": "radius_km"}))

    npts = base.groupby("cluster", as_index=False).size().rename(columns={"size": "n_points"})

    out = cent.merge(r90, on="cluster").merge(npts, on="cluster")
    out["radius_m"] = out["radius_km"] * 1000.0
    return out.sort_values("cluster")


def color_dot_html(color_rgba):
    r, g, b, a = color_rgba
    return f"""
    <span style="
        display:inline-block;width:14px;height:14px;
        border-radius:50%;margin-right:6px;
        background: rgba({r},{g},{b},{max(0,min(255,a))});
        border: 1px solid rgba(0,0,0,0.25);
    "></span>
    """


# =========================
# Load & Sidebar
# =========================
df = load_points(POINTS_FILE)

# Sidebar — filtros simples
st.sidebar.header("Filtros")
clusters_sorted = sorted(pd.Series(df["cluster"].dropna().unique()).astype(int).tolist())
selected_clusters = st.sidebar.multiselect("Clusters", clusters_sorted, default=clusters_sorted)

show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_areas = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_labels = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# parâmetros fixos (sem controles)
max_points = MAX_POINTS_DEFAULT
point_size = POINT_SIZE_DEFAULT
base_choice = BASE_MAP_DEFAULT


# =========================
# Filtro e métricas
# =========================
df_map = df.copy()
if selected_clusters:
    df_map = df_map[df_map["cluster"].isin(selected_clusters)]

# amostra para o mapa
if len(df_map) > max_points:
    df_map = df_map.sample(n=max_points, random_state=42).reset_index(drop=True)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))

if not df_map.empty:
    lat_range = f'{df_map["latitude"].min():.3f} ~ {df_map["latitude"].max():.3f}'
    lon_range = f'{df_map["longitude"].min():.3f} ~ {df_map["longitude"].max():.3f}'
else:
    lat_range = "–"
    lon_range = "–"
c3.metric("Lat range", lat_range)
c4.metric("Lon range", lon_range)

# resumo de CDs/raios
summary = compute_cluster_summary(df_map)


# =========================
# Abas
# =========================
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])


# ---------- Mapa ----------
with tab_map:
    st.subheader("Mapa por cluster")

    # Paleta e dados imutáveis para o PyDeck
    palette = {}
    legend_items = []
    df_map_local = df_map.copy()

    if "cluster" in df_map_local.columns and df_map_local["cluster"].notna().any():
        ordered = sorted(df_map_local["cluster"].dropna().astype(int).unique().tolist())
        palette = make_palette(ordered)
        default_color = [120, 120, 120, 180]
        df_map_local["rgba"] = df_map_local["cluster"].map(
            lambda c: palette.get(int(c) if pd.notna(c) else None, default_color)
        )
        legend_items = [(c, palette[c]) for c in ordered]
    else:
        df_map_local["rgba"] = [[30, 144, 255, 200]] * len(df_map_local)

    points_records = (
        df_map_local[["longitude", "latitude", "cluster", "rgba"]]
        .astype({"longitude": float, "latitude": float})
        .to_dict(orient="records")
    )

    view = compute_view(df_map_local)

    # Basemap com TileLayer (compatível)
    if base_choice == "Carto Light":
        base_layer = pdk.Layer(
            "TileLayer",
            data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
        )
    else:  # OSM
        base_layer = pdk.Layer(
            "TileLayer",
            data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0,
        )

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_records,
        get_position='[longitude, latitude]',
        get_fill_color='rgba',
        get_radius=1,
        radius_min_pixels=int(point_size),
        pickable=True,
        stroked=True,
        get_line_color=[0, 0, 0, 100],
        line_width_min_pixels=1,
    )

    layers = [base_layer, points_layer]

    # Círculos p90 por CD
    if show_areas and not summary.empty:
        areas = summary.copy()

        def col(c):
            base = palette.get(int(c), [80, 120, 255, 160]) if palette else [80, 120, 255, 160]
            return [base[0], base[1], base[2], 80]

        areas["rgba"] = areas["cluster"].apply(col)

        areas_records = (
            areas[["centroid_lon", "centroid_lat", "radius_m", "rgba", "cluster", "n_points"]]
            .astype({"centroid_lon": float, "centroid_lat": float, "radius_m": float})
            .to_dict(orient="records")
        )

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=areas_records,
            get_position='[centroid_lon, centroid_lat]',
            get_radius="radius_m",
            radius_units="meters",
            pickable=True,
            filled=True,
            get_fill_color='rgba',
            stroked=True,
            get_line_color=[30, 87, 255, 220],
            line_width_min_pixels=2,
        ))

    # Rótulos com a contagem
    if show_labels and not summary.empty:
        labels_records = (
            summary[["centroid_lon", "centroid_lat", "n_points"]]
            .astype({"centroid_lon": float, "centroid_lat": float})
            .to_dict(orient="records")
        )

        layers.append(pdk.Layer(
            "TextLayer",
            data=labels_records,
            get_position='[centroid_lon, centroid_lat]',
            get_text='n_points',
            get_size=16,
            get_color=[40, 40, 40, 230],
            get_alignment_baseline='"bottom"',
        ))

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
        "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}
    }

    deck = pdk.Deck(initial_view_state=view, layers=layers, tooltip=tooltip)

    # Render robusto: tenta pydeck_chart; fallback para HTML se necessário
    try:
        st.pydeck_chart(deck, use_container_width=True)
    except Exception:
        components.html(deck.to_html(as_string=True, notebook_display=False),
                        height=650, scrolling=False)

    # Legenda
    if show_legend and legend_items:
        st.markdown("#### Legenda (cluster → cor)")
        cols = st.columns(min(6, len(legend_items)))
        for i, (c, rgba) in enumerate(legend_items):
            with cols[i % len(cols)]:
                st.markdown(
                    f"{color_dot_html(rgba)} **Cluster {c}**",
                    unsafe_allow_html=True,
                )

# ---------- Tabelas & Gráficos ----------
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")

    if summary.empty:
        st.info("Sem dados para os clusters selecionados.")
    else:
        # Tabela
        st.dataframe(
            summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]],
            use_container_width=True
        )

        # Download
        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar resumo (CSV)", data=csv, file_name="cds_raios_resumo.csv", mime="text/csv")

        # Gráficos (Altair)
        st.markdown("### Pontos por cluster")
        chart_pts = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                y=alt.Y("cluster:O", sort="-x", title="cluster"),
                x=alt.X("n_points:Q", title="nº de pontos"),
                tooltip=["cluster", "n_points"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_pts, use_container_width=True)

        st.markdown("### Raio p90 (km) por cluster")
        chart_r = (
            alt.Chart(summary)
            .mark_bar(color="#FF9800")
            .encode(
                y=alt.Y("cluster:O", sort="-x", title="cluster"),
                x=alt.X("radius_km:Q", title="raio p90 (km)"),
                tooltip=["cluster", alt.Tooltip("radius_km:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_r, use_container_width=True)
