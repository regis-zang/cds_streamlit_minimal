# app.py
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Pontos por cluster (cores distintas)", layout="wide", page_icon="🗺️")

DATA_FILE = Path("DataBase/points_enriched_final.parquet")


# =========================
# Utils
# =========================
@st.cache_data(show_spinner=False)
def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"latitude", "longitude", "cluster"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Colunas faltando no parquet: {sorted(miss)}")

    df = df.dropna(subset=["latitude", "longitude", "cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    return df


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    R = 6371.0088
    p = math.pi / 180.0
    lat1, lon1, lat2, lon2 = lat1 * p, lon1 * p, lat2 * p, lon2 * p
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        (pd.Series(dlat).apply(math.sin) ** 2)
        + pd.Series(lat1).apply(math.cos)
        * pd.Series(lat2).apply(math.cos)
        * (pd.Series(dlon).apply(math.sin) ** 2)
    )
    c = 2 * a.apply(math.sqrt).apply(math.asin)
    return R * c


def compute_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna: cluster, centroid_lat, centroid_lon, radius_km, n_points, radius_m.
    Robusta a MultiIndex e a sufixos _x/_y após merge.
    """
    # 1) centróides com nomes finais (sem MultiIndex)
    cent = (
        df.groupby("cluster")[["latitude", "longitude"]]
        .median()
        .reset_index()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
    )

    # 2) merge e padronização de nomes (se vierem _x/_y)
    base = df.merge(cent, on="cluster", how="left")

    # Caso o merge tenha criado latitude_x/latitude_y, padroniza:
    # - latitude/longitude => pontos
    # - centroid_lat/centroid_lon => centróides
    ren = {}
    if "latitude_x" in base.columns and "latitude" not in base.columns:
        ren["latitude_x"] = "latitude"
    if "longitude_x" in base.columns and "longitude" not in base.columns:
        ren["longitude_x"] = "longitude"
    if "latitude_y" in base.columns and "centroid_lat" not in base.columns:
        ren["latitude_y"] = "centroid_lat"
    if "longitude_y" in base.columns and "centroid_lon" not in base.columns:
        ren["longitude_y"] = "centroid_lon"
    if ren:
        base = base.rename(columns=ren)

    # Garante presença das colunas esperadas
    needed = {"latitude", "longitude", "centroid_lat", "centroid_lon"}
    miss = needed - set(base.columns)
    if miss:
        # Último fallback: se ainda faltou, recalcula centróide e faz segundo merge
        cent2 = (
            df.groupby("cluster", as_index=False)
            .agg(centroid_lat=("latitude", "median"), centroid_lon=("longitude", "median"))
        )
        base = df.merge(cent2, on="cluster", how="left")

    # 3) distância e raio p90
    base["dist_km"] = haversine_km(
        base["latitude"], base["longitude"], base["centroid_lat"], base["centroid_lon"]
    )
    r90 = (
        base.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )
    cnt = (
        base.groupby("cluster", as_index=False)
        .size()
        .rename(columns={"size": "n_points"})
    )

    summary = (
        base[["cluster", "centroid_lat", "centroid_lon"]]
        .drop_duplicates("cluster")
        .merge(r90, on="cluster")
        .merge(cnt, on="cluster")
    )
    summary["radius_m"] = (summary["radius_km"] * 1000.0).astype(float)
    return summary


def palette_for_clusters(values: list[int]) -> dict[int, list[int]]:
    base = {
        0: [231, 76, 60, 200],
        1: [46, 204, 113, 200],
        2: [52, 152, 219, 200],
        3: [155, 89, 182, 200],
        4: [241, 196, 15, 200],
        5: [230, 126, 34, 200],
        6: [26, 188, 156, 200],
        7: [127, 140, 141, 200],
        8: [52, 73, 94, 200],
        9: [243, 156, 18, 200],
    }
    return {c: base.get(c, [200, 200, 200, 200]) for c in values}


def make_deck(df_map: pd.DataFrame, summary: pd.DataFrame, show_p90: bool, show_counts: bool) -> pdk.Deck:
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )

    pts = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        pickable=True,
        radius_min_pixels=2,
        get_radius=60,
        stroked=False,
        filled=True,
    )

    layers = [base_layer, pts]

    if show_p90 and not summary.empty:
        rings = pdk.Layer(
            "ScatterplotLayer",
            data=summary,
            get_position="[centroid_lon, centroid_lat]",
            get_radius="radius_m",
            filled=False,
            stroked=True,
            line_width_min_pixels=2,
            get_line_color=[30, 144, 255, 160],
        )
        layers.append(rings)

    if show_counts and not summary.empty:
        txt = pdk.Layer(
            "TextLayer",
            data=summary.assign(txt=lambda d: d["n_points"].astype(str)),
            get_position="[centroid_lon, centroid_lat]",
            get_text="txt",
            get_color=[30, 144, 255, 230],
            get_size=14,
            sizeUnits="meters",
            sizeScale=4,
            sizeMinPixels=10,
            get_alignment_baseline="'bottom'",
        )
        layers.append(txt)

    if len(df_map) > 0:
        lat0 = float(df_map["latitude"].mean())
        lon0 = float(df_map["longitude"].mean())
    else:
        lat0, lon0 = -22.0, -47.0
    view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6.5)

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
        "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view, tooltip=tooltip)


# =========================
# App
# =========================
st.title("Pontos por cluster (cores distintas)")

df_all = load_points(DATA_FILE)

st.sidebar.header("Filtros")
clusters_sorted = sorted(df_all["cluster"].unique().tolist())
selected = st.sidebar.multiselect("Clusters", options=clusters_sorted, default=clusters_sorted)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90 = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

df_map = df_all[df_all["cluster"].isin(selected)].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df_all):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
if len(df_map):
    c3.metric("Lat range", f"{df_map['latitude'].min():.5f} ~ {df_map['latitude'].max():.5f}")
    c4.metric("Lon range", f"{df_map['longitude'].min():.5f} ~ {df_map['longitude'].max():.5f}")
else:
    c3.metric("Lat range", "-")
    c4.metric("Lon range", "-")

pal = palette_for_clusters(clusters_sorted)
df_map["rgba"] = df_map["cluster"].map(pal)

tab_map, tab_charts = st.tabs(["🗺️ Mapa", "📊 CDs & Raios"])

with tab_map:
    summary = compute_cluster_summary(df_map) if len(df_map) else pd.DataFrame(
        columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points", "radius_m"]
    )
    deck = make_deck(df_map, summary, show_p90=show_p90, show_counts=show_counts)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend and selected:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(min(5, len(selected)))
        for i, c in enumerate(sorted(selected)):
            r, g, b, _ = pal[c]
            with cols[i % len(cols)]:
                st.markdown(
                    f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                    f"background:rgba({r},{g},{b},1);margin-right:8px;vertical-align:middle'></span>"
                    f"Cluster {c}",
                    unsafe_allow_html=True,
                )

with tab_charts:
    summary = compute_cluster_summary(df_map) if len(df_map) else pd.DataFrame(
        columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points", "radius_m"]
    )

    st.subheader("CDs & Raios (resumo por cluster)")
    st.dataframe(
        summary[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]],
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if len(summary):
            fig = px.bar(
                summary.sort_values("n_points", ascending=False),
                x="n_points",
                y="cluster",
                orientation="h",
                title="Pontos por cluster",
                labels={"n_points": "Pontos", "cluster": "Cluster"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para exibir o gráfico de contagem.")

    with c2:
        if len(summary):
            fig = px.bar(
                summary.sort_values("radius_km", ascending=False),
                x="radius_km",
                y="cluster",
                orientation="h",
                title="Raio p90 (km) por cluster",
                labels={"radius_km": "Raio p90 (km)", "cluster": "Cluster"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem dados para exibir o gráfico de raios.")
