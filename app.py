# app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ===== Streamlit =====
st.set_page_config(page_title="CDs • Minimal", layout="wide")
st.title("📍 Visualização mínima de pontos (Parquet)")

DATA_DIR = Path("DataBase")
FILE = DATA_DIR / "points_enriched_final.parquet"

# ---------- Utils ----------
@st.cache_data(show_spinner=False)
def load_parquet_safe(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {p}\n"
            f"Coloque o Parquet em {DATA_DIR.resolve()}"
        )
    # Tenta engine padrão e cai para fastparquet
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

# ---------- Load ----------
df = load_parquet_safe(FILE)
df.columns = [c.strip().lower() for c in df.columns]

# padroniza nomes comuns
renames = {}
if "lng" in df.columns: renames["lng"] = "longitude"
if "lon" in df.columns: renames["lon"] = "longitude"
if "long" in df.columns: renames["long"] = "longitude"
if "lat" in df.columns: renames["lat"] = "latitude"
df = df.rename(columns=renames)

# valida colunas de geo
required = {"latitude", "longitude"}
if not required.issubset(df.columns):
    st.error(f"Colunas necessárias não encontradas: {required - set(df.columns)}")
    st.stop()

# limpeza básica
df = clean_geo(df, "longitude", "latitude")

# ---------- Sidebar / filtros ----------
st.sidebar.header("Filtros")
if "cluster" in df.columns:
    clusters = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    selected = st.sidebar.multiselect("Clusters", clusters, default=clusters)
    if selected:
        df = df[df["cluster"].astype(int).isin(selected)]
else:
    st.sidebar.info("Coluna 'cluster' não encontrada — mostrando todos os pontos.")

max_points = st.sidebar.slider("Máx. de pontos no mapa", 500, 20000, 5000, step=500)

# Amostragem só para o mapa (pra não travar)
df_map = df if len(df) <= max_points else df.sample(max_points, random_state=42)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de pontos (dados)", len(df))
c2.metric("Pontos no mapa (amostra)", len(df_map))
lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
c3.metric("Lat range", f"{lat_min:.3f} ~ {lat_max:.3f}")
c4.metric("Lon range", f"{lon_min:.3f} ~ {lon_max:.3f}")

# ---------- Mapa (ultra simples) ----------
# st.map aceita 'latitude'/'longitude' (ou 'lat'/'lon')
st.subheader("Mapa")
st.map(df_map[["latitude", "longitude"]], use_container_width=True)

# ---------- Tabela + download ----------
st.subheader("Prévia dos dados filtrados")
view_cols = [c for c in ["cluster", "latitude", "longitude"] if c in df.columns]
st.dataframe(df[view_cols].head(1000), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV filtrado", data=csv,
                   file_name="pontos_filtrado.csv", mime="text/csv")

st.caption("Pronto. Simples e direto. ✨")
