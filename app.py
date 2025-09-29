# app.py
from pathlib import Path
import colorsys
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -------------------- CONFIG STREAMLIT --------------------
st.set_page_config(page_title="CDs • Minimal (clusters coloridos)", layout="wide")
st.title("📍 Pontos por cluster (cores distintas)")

DATA_DIR = Path("DataBase")
FILE = DATA_DIR / "points_enriched_final.parquet"

# -------------------- UTILS --------------------
@st.cache_data(show_spinner=False)
def load_parquet_safe(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {p}\n"
            f"Coloque o Parquet em {DATA_DIR.resolve()}"
        )
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")

def clean_geo(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    out = df.copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out = out.dropna(subset=[lon_col, lat_col])
    out = out[out[lon_col].between(-180, 180) & out[lat_col].between(-90, 90)]
    return out

def make_palette(values: list[int]) -> dict[int, list]:
    """
    Gera uma paleta RGBA (0-255) estável para a lista ordenada de clusters.
    Usa HSV com espaçamento uniforme (s=0.65, v=0.95).
    """
    n = max(1, len(values))
    palette = {}
    for i, c in enumerate(values):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        rgba = [int(r * 255), int(g * 255), int(b * 255), 210]
        palette[int(c)] = rgba
    return palette

def compute_view(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5)  # Brasil
    lat_c = float(df["latitude"].mean())
    lon_c = float(df["longitude"].mean())
    # zoom heurístico pelo spread lat (bem simples, mas eficaz)
    spread = float(df["latitude"].max() - df["latitude"].min())
    zoom = 3.5 if spread > 20 else 5 if spread > 8 else 6.5 if spread > 3 else 8
    return pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom)

# -------------------- LOAD --------------------
df = load_parquet_safe(FILE)
df.columns = [c.strip().lower() for c in df.columns]

# padroniza nomes comuns
ren = {}
if "lng" in df.columns:  ren["lng"] = "longitude"
if "lon" in df.columns:  ren["lon"] = "longitude"
if "long" in df.columns: ren["long"] = "longitude"
if "lat" in df.columns:  ren["lat"] = "latitude"
df = df.rename(columns=ren)

required = {"latitude", "longitude"}
if not required.issubset(df.columns):
    st.error(f"Colunas necessárias não encontradas: {required - set(df.columns)}")
    st.stop()

# cluster em inteiro se existir
if "cluster" in df.columns:
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")

# limpeza geo
df = clean_geo(df, "longitude", "latitude")

# -------------------- SIDEBAR --------------------
st.sidebar.header("Filtros")
if "cluster" in df.columns and df["cluster"].notna().any():
    clusters_all = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    selected = st.sidebar.multiselect("Clusters", clusters_all, default=clusters_all)
    if selected:
        df = df[df["cluster"].isin(selected)]
else:
    st.sidebar.info("Coluna 'cluster' não encontrada — mostrando todos os pontos.")

max_points = st.sidebar.slider("Máx. de pontos no mapa", 500, 30000, 8000, step=500)
point_size = st.sidebar.slider("Tamanho do marcador (px)", 2, 12, 6)
show_legend = st.sidebar.checkbox("Mostrar legenda de cores", value=True)

# Amostragem só para o mapa
df_map = df if len(df) <= max_points else df.sample(max_points, random_state=42)

# -------------------- KPIs --------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", len(df))
c2.metric("Pontos no mapa (amostra)", len(df_map))
lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
c3.metric("Lat range", f"{lat_min:.3f} ~ {lat_max:.3f}")
c4.metric("Lon range", f"{lon_min:.3f} ~ {lon_max:.3f}")

# -------------------- CORES POR CLUSTER --------------------
if "cluster" in df_map.columns and df_map["cluster"].notna().any():
    ordered = sorted(df_map["cluster"].dropna().astype(int).unique().tolist())
    palette = make_palette(ordered)
    default_color = [120, 120, 120, 180]

    df_map = df_map.copy()
    # mapeia cor por cluster com fallback seguro
    df_map["rgba"] = df_map["cluster"].map(
        lambda c: palette.get(int(c) if pd.notna(c) else None, default_color)
    )

    legend_items = [(c, palette[c]) for c in ordered]
else:
    # Sem cluster: cor única
    df_map = df_map.copy()
    df_map["rgba"] = [[30, 144, 255, 200]] * len(df_map)
    legend_items = []

# -------------------- MAPA COM PYDECK --------------------
view = compute_view(df_map)

tooltip = {
    "html": (
        "<b>Cluster:</b> {cluster}<br/>"
        "<b>Lat:</b> {latitude}<br/>"
        "<b>Lon:</b> {longitude}"
    ),
    "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"},
}

# --- CAMADA DE PONTOS (substitua a atual) ---
points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[longitude, latitude]',
    get_fill_color='rgba',
    get_radius=1,                 # metros (valor qualquer, vai ser "clampado")
    radius_min_pixels=point_size, # garante tamanho em PIXELS (visível em qualquer zoom)
    pickable=True,
    stroked=True,
    get_line_color=[0, 0, 0, 100],
    line_width_min_pixels=1.5,
)

# Base Carto (não precisa de token)
try:
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=view,
        layers=[points_layer],
        tooltip=tooltip,
    )
except Exception:
    # fallback sem basemap
    deck = pdk.Deck(initial_view_state=view, layers=[points_layer], tooltip=tooltip)

st.subheader("Mapa por cluster (cores distintas)")
st.pydeck_chart(deck, use_container_width=True)

# -------------------- LEGENDA --------------------
if show_legend and legend_items:
    st.markdown("#### Legenda (cluster → cor)")
    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:10px'>"
    for c, col in legend_items:
        r,g,b,a = col
        legend_html += (
            f"<div style='display:flex;align-items:center;gap:6px;"
            f"border:1px solid #ddd;border-radius:6px;padding:4px 8px;'>"
            f"<span style='width:14px;height:14px;background:rgba({r},{g},{b},{a/255});"
            f"border:1px solid #666;display:inline-block;border-radius:3px'></span>"
            f"<span>Cluster {c}</span></div>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

# -------------------- TABELA + DOWNLOAD --------------------
st.subheader("Prévia dos dados filtrados")
view_cols = [c for c in ["cluster", "latitude", "longitude"] if c in df.columns]
st.dataframe(df[view_cols].head(1000), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV filtrado", data=csv,
                   file_name="pontos_filtrado.csv", mime="text/csv")

st.caption("Agora cada cluster tem sua cor — simples, leve e objetivo. 🚀")
