# app.py
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="CDs • Mapa e Raios", layout="wide")

DATA_DIR = Path("DataBase")
FILE = DATA_DIR / "points_enriched_final.parquet"

# ========================= Utils =========================
@st.cache_data(show_spinner=False)
def load_parquet_safe(p: Path) -> pd.DataFrame:
    """Lê o parquet com fallback para fastparquet."""
    if not p.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {p}")
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")


def clean_geo(df: pd.DataFrame, lon: str, lat: str) -> pd.DataFrame:
    g = df.copy()
    g[lon] = pd.to_numeric(g[lon], errors="coerce")
    g[lat] = pd.to_numeric(g[lat], errors="coerce")
    g = g.dropna(subset=[lon, lat])
    return g[g[lon].between(-180, 180) & g[lat].between(-90, 90)]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


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

    # Se vieram do Parquet colunas antigas de centróide / métricas, removo para não gerar sufixos.
    drop_cols = [c for c in ["centroid_lat", "centroid_lon", "radius_km", "radius_m", "n_points", "dist_km"] if c in base.columns]
    if drop_cols:
        base = base.drop(columns=drop_cols)

    # centróides com nomes definidos
    cent = (base.groupby("cluster", as_index=False)
                 .agg(centroid_lat=("latitude", "mean"),
                      centroid_lon=("longitude", "mean")))

    # merge (se por qualquer motivo ainda houver colisão, o Pandas pode sufixar)
    tmp = base.merge(cent, on="cluster", how="left")

    # identifica os nomes reais das colunas de centróide pós-merge
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


def make_palette(values: list[int]) -> dict[int, list]:
    """Cores RGBA por cluster (estáveis)."""
    n = max(1, len(values))
    pal = {}
    for i, c in enumerate(values):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.65, 0.95)
        pal[int(c)] = [int(r*255), int(g*255), int(b*255), 210]
    return pal


def compute_view(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5)
    lat_c, lon_c = float(df["latitude"].mean()), float(df["longitude"].mean())
    spread = float(df["latitude"].max() - df["latitude"].min())
    zoom = 3.5 if spread > 20 else 5 if spread > 8 else 6.5 if spread > 3 else 8
    return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom)


# ========================= Load =========================
df = load_parquet_safe(FILE)
df.columns = [c.strip().lower() for c in df.columns]
# padroniza nomes de lat/lon
ren = {}
if "lng" in df.columns:  ren["lng"] = "longitude"
if "lon" in df.columns:  ren["lon"] = "longitude"
if "long" in df.columns: ren["long"] = "longitude"
if "lat" in df.columns:  ren["lat"] = "latitude"
df = df.rename(columns=ren)

if not {"latitude","longitude"}.issubset(df.columns):
    st.error("Colunas latitude/longitude não encontradas.")
    st.stop()

# cluster como inteiro (se existir)
if "cluster" in df.columns:
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")

df = clean_geo(df, "longitude", "latitude")

# ========================= Sidebar =========================
st.sidebar.header("Filtros")

if "cluster" in df.columns and df["cluster"].notna().any():
    clusters_all = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    sel_clusters = st.sidebar.multiselect("Clusters", clusters_all, default=clusters_all)
    if sel_clusters:
        df = df[df["cluster"].isin(sel_clusters)]
else:
    st.sidebar.info("Coluna 'cluster' não encontrada — mostrando todos os pontos.")
    sel_clusters = []

# ---- Parâmetros fixos (sem mostrar controles) ----
MAX_POINTS_DEFAULT = 8000
POINT_SIZE_DEFAULT = 6
BASE_MAP_DEFAULT   = "Carto Light"   # ou "OSM Standard" se preferir

max_points  = MAX_POINTS_DEFAULT
point_size  = POINT_SIZE_DEFAULT
base_choice = BASE_MAP_DEFAULT

show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)
show_areas  = st.sidebar.checkbox("Mostrar círculos p90 por CD", value=True)
show_labels = st.sidebar.checkbox("Mostrar contagem no centróide", value=True)

# Amostra para o mapa
df_map = df if len(df) <= max_points else df.sample(max_points, random_state=42)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", len(df))
c2.metric("Pontos no mapa (amostra)", len(df_map))
c3.metric("Lat range", f"{df['latitude'].min():.3f} ~ {df['latitude'].max():.3f}")
c4.metric("Lon range", f"{df['longitude'].min():.3f} ~ {df['longitude'].max():.3f}")

# Resumo por cluster (CDs)
summary = compute_cluster_summary(df)

# ========================= Tabs =========================
tab_map, tab_charts = st.tabs(["🗺️  Mapa", "📊  CDs & Raios"])

with tab_map:
    st.subheader("Mapa por cluster")

    # cores por cluster
    palette = {}
    legend_items = []
    if "cluster" in df_map.columns and df_map["cluster"].notna().any():
        ordered = sorted(df_map["cluster"].dropna().astype(int).unique().tolist())
        palette = make_palette(ordered)
        default_color = [120,120,120,180]
        df_map = df_map.copy()
        df_map["rgba"] = df_map["cluster"].map(
            lambda c: palette.get(int(c) if pd.notna(c) else None, default_color)
        )
        legend_items = [(c, palette[c]) for c in ordered]
    else:
        df_map = df_map.copy()
        df_map["rgba"] = [[30,144,255,200]] * len(df_map)

    view = compute_view(df_map)

    # camada de pontos (tamanho em pixels)
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[longitude, latitude]',
        get_fill_color='rgba',
        get_radius=1,                 # metros (irrelevante com radius_min_pixels)
        radius_min_pixels=point_size, # garante pixels
        pickable=True,
        stroked=True,
        get_line_color=[0,0,0,100],
        line_width_min_pixels=1,
    )

    layers = []

    # fundo OSM (como na referência) ou Carto
    if base_choice == "OSM Standard":
        layers.append(pdk.Layer(
            "TileLayer",
            data="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0, max_zoom=19, tile_size=256, opacity=1.0
        ))

    layers.append(points_layer)

    # círculos p90 por cluster
    if show_areas and not summary.empty:
        areas = summary.copy()
        def col(c):
            base = palette.get(int(c), [80,120,255,160]) if palette else [80,120,255,160]
            return [base[0], base[1], base[2], 80]
        areas["rgba"] = areas["cluster"].apply(col)

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=areas,
            get_position='[centroid_lon, centroid_lat]',
            get_radius="radius_m",     # metros (aqui sim é real)
            radius_units="meters",
            pickable=True,
            filled=True,
            get_fill_color='rgba',
            stroked=True,
            get_line_color=[30, 87, 255, 220],
            line_width_min_pixels=2,
        ))

    # rótulos (n_points) no centro
    if show_labels and not summary.empty:
        layers.append(pdk.Layer(
            "TextLayer",
            data=summary,
            get_position='[centroid_lon, centroid_lat]',
            get_text='n_points',
            get_size=16,
            get_color=[40,40,40,230],
            get_alignment_baseline='"bottom"',
        ))

    tooltip = {
        "html": "<b>Cluster:</b> {cluster}<br/><b>Lat:</b> {latitude}<br/><b>Lon:</b> {longitude}",
        "style": {"backgroundColor":"rgba(0,0,0,0.85)","color":"white"}
    }

    if base_choice == "Carto Light":
        deck = pdk.Deck(map_provider="carto", map_style="light",
                        initial_view_state=view, layers=layers, tooltip=tooltip)
    else:
        deck = pdk.Deck(initial_view_state=view, layers=layers, tooltip=tooltip)

    st.pydeck_chart(deck, use_container_width=True)

    if show_legend and legend_items:
        st.markdown("#### Legenda (cluster → cor)")
        html = "<div style='display:flex;flex-wrap:wrap;gap:10px'>"
        for c, col in legend_items:
            r,g,b,a = col
            html += (
                f"<div style='display:flex;align-items:center;gap:6px;"
                f"border:1px solid #ddd;border-radius:6px;padding:4px 8px;'>"
                f"<span style='width:14px;height:14px;background:rgba({r},{g},{b},{a/255});"
                f"border:1px solid #666;display:inline-block;border-radius:3px'></span>"
                f"<span>Cluster {c}</span></div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

with tab_charts:
    st.subheader("CDs & Raios (resumo por cluster)")
    if summary.empty:
        st.info("Não foi possível calcular o resumo (verifique a coluna 'cluster').")
    else:
        st.dataframe(
            summary[['cluster','centroid_lat','centroid_lon','radius_km','n_points']],
            use_container_width=True
        )

        csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar resumo (CSV)", data=csv,
                           file_name="cds_resumo.csv", mime="text/csv")

        c1, c2 = st.columns(2)

        bar_pts = (alt.Chart(summary)
                   .mark_bar(color="#4e79a7")
                   .encode(x=alt.X('n_points:Q', title='nº de pontos'),
                           y=alt.Y('cluster:N', title='cluster', sort='-x'))
                   .properties(height=400, title="Pontos por cluster"))

        bar_rad = (alt.Chart(summary)
                   .mark_bar(color="#f28e2c")
                   .encode(x=alt.X('radius_km:Q', title='raio p90 (km)'),
                           y=alt.Y('cluster:N', title='cluster', sort='-x'))
                   .properties(height=400, title="Raio p90 (km) por cluster"))

        c1.altair_chart(bar_pts, use_container_width=True)
        c2.altair_chart(bar_rad, use_container_width=True)

# ========================= Prévia =========================
st.subheader("Prévia dos dados filtrados")
view_cols = [c for c in ["cluster","latitude","longitude"] if c in df.columns]
st.dataframe(df[view_cols].head(1000), use_container_width=True)
