# app.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path

# -----------------------------
# Config e cache
# -----------------------------
st.set_page_config(page_title="CDs - Mapa e Sugestões", layout="wide")

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    parquet_path = Path("DataBase/points_enriched_final.parquet")
    df = pd.read_parquet(parquet_path)

    # Normalização suave de nomes
    df = df.rename(columns={
        "lat": "latitude",
        "lon": "longitude",
        "long": "longitude",
    })
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)
    return df

# Paleta fixa
PALETTE = {
    0: [231, 76, 60],    # vermelho
    1: [39, 174, 96],    # verde
    2: [52, 152, 219],   # azul
    3: [155, 89, 182],   # roxo
    4: [241, 196, 15],   # amarelo
}
FALLBACK = [60, 60, 60]

def colorize(df: pd.DataFrame, alpha=180) -> pd.DataFrame:
    if "cluster" not in df.columns:
        df = df.copy()
        df["cluster"] = -1
    return df.assign(
        rgba=df["cluster"]
        .map(PALETTE)
        .apply(lambda x: (x if isinstance(x, list) else FALLBACK) + [alpha])
    )

# -----------------------------
# Distância haversine
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

# -----------------------------
# Resumo por cluster (robusto)
# -----------------------------
def _pick_col(cols, *names):
    """retorna o primeiro nome existente em cols."""
    for n in names:
        if n in cols:
            return n
    return None

def compute_cluster_summary(df_map: pd.DataFrame) -> pd.DataFrame:
    """Centroides, raio p90 e contagem por cluster (robusto a colisão de nomes)."""
    if df_map.empty:
        return pd.DataFrame(
            columns=["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]
        )

    # base com centroides e contagem
    base = (
        df_map
        .groupby("cluster", as_index=False)
        .agg(centroid_lat=("latitude", "mean"),
             centroid_lon=("longitude", "mean"),
             n_points=("cluster", "size"))
    )

    # merge com sufixo controlado (se houver colisão de nomes)
    tmp = df_map.merge(base, on="cluster", how="left", suffixes=("", "_b"))

    # escolhe os nomes corretos pós-merge
    c_lat = _pick_col(tmp.columns, "centroid_lat", "centroid_lat_b", "centroid_lat_y", "centroid_lat_x")
    c_lon = _pick_col(tmp.columns, "centroid_lon", "centroid_lon_b", "centroid_lon_y", "centroid_lon_x")
    if c_lat is None or c_lon is None:
        # algo muito atípico; devolve só contagem
        out = base.rename(columns={"centroid_lat": "centroid_lat",
                                   "centroid_lon": "centroid_lon"})
        out["radius_km"] = np.nan
        return out[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]]

    tmp["dist_km"] = haversine_km(
        tmp["latitude"], tmp["longitude"], tmp[c_lat], tmp[c_lon]
    )

    r90 = (
        tmp.groupby("cluster", as_index=False)["dist_km"]
        .quantile(0.90)
        .rename(columns={"dist_km": "radius_km"})
    )

    out = base.merge(r90, on="cluster", how="left")
    return out[["cluster", "centroid_lat", "centroid_lon", "radius_km", "n_points"]]

# -----------------------------
# Sugestão de CDs
# -----------------------------
def build_cd_suggestions(df_map: pd.DataFrame) -> pd.DataFrame:
    summary = compute_cluster_summary(df_map).copy()
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "sug_cluster", "sug_tipo", "sug_nome",
                "sug_codigo", "sug_alias", "sug_citycode",
                "sug_municipio", "sug_uf",
                "sug_lat", "sug_lon",
                "sug_raio_km", "sug_qtd_pontos"
            ]
        )

    rows = []
    for _, row in summary.iterrows():
        c = int(row["cluster"])
        sub = df_map[df_map["cluster"] == c].copy()

        if not sub.empty:
            d = haversine_km(sub["latitude"], sub["longitude"],
                             row["centroid_lat"], row["centroid_lon"])
            sub = sub.assign(_d=d)
            best = sub.loc[sub["_d"].idxmin()]
        else:
            best = pd.Series({})

        municipio = best.get("cidade", np.nan)
        uf        = best.get("uf", np.nan)
        city_code = best.get("city_code", np.nan)

        tipo = "CD" if int(row["n_points"]) >= 120 else "RDC"

        if pd.notna(municipio) and pd.notna(uf):
            nome  = f"{tipo} Sugerido – {municipio}/{uf}"
            alias = f"Polo {municipio}"
        else:
            nome  = f"{tipo} Sugerido – Cluster {c}"
            alias = f"Polo C{c}"

        codigo = f"SUG-{str(city_code) if pd.notna(city_code) else str(c).zfill(3)}"

        rows.append({
            "sug_cluster": c,
            "sug_tipo": tipo,
            "sug_nome": nome,
            "sug_codigo": codigo,
            "sug_alias": alias,
            "sug_citycode": city_code,
            "sug_municipio": municipio,
            "sug_uf": uf,
            "sug_lat": float(row["centroid_lat"]),
            "sug_lon": float(row["centroid_lon"]),
            "sug_raio_km": float(row["radius_km"]),
            "sug_qtd_pontos": int(row["n_points"]),
        })

    df_sug = pd.DataFrame(rows).sort_values("sug_cluster").reset_index(drop=True)
    return df_sug

# -----------------------------
# Deck (Mapa)
# -----------------------------
def make_deck(df_map: pd.DataFrame,
              show_p90: bool = True,
              show_counts: bool = True) -> pdk.Deck:
    if df_map.empty:
        view = pdk.ViewState(latitude=-22.9, longitude=-47.0, zoom=5.5)
    else:
        view = pdk.ViewState(
            latitude=float(df_map["latitude"].mean()),
            longitude=float(df_map["longitude"].mean()),
            zoom=6.2
        )

    # 1) base
    base_layer = pdk.Layer(
        "TileLayer",
        data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=1.0,
    )

    # 2) pontos coloridos
    pts = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        pickable=True,
        get_position="[longitude, latitude]",
        get_fill_color="rgba",
        get_radius=70,
        radius_min_pixels=2,
        radius_max_pixels=8,
        stroked=False,
    )

    layers = [base_layer, pts]

    # 3) círculos p90
    if show_p90:
        summary = compute_cluster_summary(df_map)
        if not summary.empty:
            summary = summary.assign(radius_m=summary["radius_km"] * 1000.0)
            rings = pdk.Layer(
                "ScatterplotLayer",
                summary,
                pickable=False,
                get_position="[centroid_lon, centroid_lat]",
                get_radius="radius_m",
                stroked=True,
                filled=False,
                get_line_color=[30, 90, 200, 60],
                line_width_min_pixels=2,
            )
            layers.append(rings)

    # 4) contagem no centróide
    if show_counts:
        summary = compute_cluster_summary(df_map)
        if not summary.empty:
            summary = summary.assign(label=summary["n_points"].astype(str))
            txt = pdk.Layer(
                "TextLayer",
                summary,
                get_position="[centroid_lon, centroid_lat]",
                get_text="label",
                get_size=12,
                get_color=[70, 70, 70, 240],
                get_alignment_baseline="'bottom'",
            )
            layers.append(txt)

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=None,  # evita overlay azul
        tooltip={
            "html": "<b>Cluster:</b> {cluster}<br>"
                    "<b>Lat:</b> {latitude}<br>"
                    "<b>Lon:</b> {longitude}",
            "style": {"font-size": "11px"}
        }
    )

# -----------------------------
# UI
# -----------------------------
df = load_data()

st.sidebar.header("Filtros")
all_clusters = sorted(df["cluster"].dropna().unique().astype(int).tolist())
selected = st.sidebar.multiselect(
    "Clusters",
    options=all_clusters,
    default=all_clusters,
    format_func=lambda x: str(x),
)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_p90     = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_counts  = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

df_map = df[df["cluster"].isin(selected)].copy()
df_map = colorize(df_map, alpha=180)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", f"{len(df):,}".replace(",", "."))
c2.metric("Pontos no mapa (amostra)", f"{len(df_map):,}".replace(",", "."))
if not df_map.empty:
    c3.metric("Lat range", f"{df_map['latitude'].min():.6f} ~ {df_map['latitude'].max():.6f}")
    c4.metric("Lon range", f"{df_map['longitude'].min():.6f} ~ {df_map['longitude'].max():.6f}")
else:
    c3.metric("Lat range", "–")
    c4.metric("Lon range", "–")

tab_map, tab_charts, tab_sug = st.tabs(["🗺️ Mapa", "📊 CDs & Raios", "💡 Sugestão de CDs"])

# ----- Mapa
with tab_map:
    st.subheader("Mapa por cluster")
    deck = make_deck(df_map, show_p90=show_p90, show_counts=show_counts)
    st.pydeck_chart(deck, use_container_width=True)

    if show_legend:
        st.markdown("**Legenda (cluster → cor)**")
        cols = st.columns(min(6, len(PALETTE)))
        keys = sorted(PALETTE.keys())
        for i, c in enumerate(keys):
            rgb = PALETTE[c]
            html = f"""
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="width:12px;height:12px;border-radius:50%;
                             background: rgba({rgb[0]},{rgb[1]},{rgb[2]},1.0);
                             display:inline-block;"></span>
                Cluster {c}
            </div>
            """
            cols[i % len(cols)].markdown(html, unsafe_allow_html=True)

# ----- CDs & Raios
with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    summary = compute_cluster_summary(df_map)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    csv = summary.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar resumo (CSV)", csv, file_name="resumo_cds_raios.csv", mime="text/csv")

# ----- Sugestão de CDs
with tab_sug:
    st.subheader("Sugestão de CDs (uma linha por cluster)")
    df_sug = build_cd_suggestions(df_map)
    st.dataframe(df_sug, use_container_width=True, hide_index=True)
    csv2 = df_sug.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar sugestões (CSV)", csv2, file_name="sugestao_cds.csv", mime="text/csv")
    if not df_sug.empty:
        st.caption(
            f"{len(df_sug)} clusters sugeridos • "
            f"média de raio: {df_sug['sug_raio_km'].mean():.1f} km • "
            f"total de pontos cobertos: {df_sug['sug_qtd_pontos'].sum()}"
        )
