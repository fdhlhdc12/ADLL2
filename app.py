import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(page_title="Stock Clustering Analysis", layout="wide", initial_sidebar_state="collapsed")

# ============================================
# DATA SAHAM
# ============================================
STOCKS_DATA = {
    'ACES.JK': 'Ace Hardware Indonesia', 'MAPA.JK': 'Map Aktif Adiperkasa',
    'MAPI.JK': 'Mitra Adiperkasa', 'ERAA.JK': 'Erajaya Swasembada',
    'BBCA.JK': 'Bank Central Asia', 'BBNI.JK': 'Bank Negara Indonesia',
    'BBRI.JK': 'Bank Rakyat Indonesia', 'BMRI.JK': 'Bank Mandiri',
    'BSDE.JK': 'Bumi Serpong Damai', 'CTRA.JK': 'Ciputra Development',
    'KIJA.JK': 'Kawasan Industri Jababeka', 'PWON.JK': 'Pakuwon Jati',
    'BUKA.JK': 'Bukalapak', 'EMTK.JK': 'Elang Mahkota Teknologi',
    'EXCL.JK': 'XL Axiata', 'GOTO.JK': 'GoTo Gojek Tokopedia',
    'TLKM.JK': 'Telkom Indonesia', 'SCMA.JK': 'Surya Citra Media',
    'CMRY.JK': 'Cisarua Mountain Dairy', 'SIDO.JK': 'Industri Jamu & Farmasi Sido',
    'UNVR.JK': 'Unilever Indonesia', 'JSMR.JK': 'Jasa Marga',
    'PGEO.JK': 'Pertamina Geothermal Energy', 'TOWR.JK': 'Sarana Menara Nusantara',
    'MIKA.JK': 'Mitra Keluarga Karyasehat', 'SMGR.JK': 'Semen Indonesia',
    'TPIA.JK': 'Chandra Asri Pacific'
}

# ============================================
# FUNGSI CACHED (AGAR TIDAK REKOMPUTASI SETIAP INTERAKSI)
# ============================================
@st.cache_data(ttl=3600)
def load_data_and_compute():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data, successful = {}, []
    with st.status("📥 Mengunduh data harga dari Yahoo Finance...", expanded=False) as status:
        for ticker, name in STOCKS_DATA.items():
            try:
                stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not stock.empty and len(stock) > 100:
                    close = stock['Close']
                    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
                    data[ticker] = close.squeeze()
                    successful.append(ticker)
            except Exception:
                pass

        if len(successful) < 10:
            st.warning("⚠️ Data Yahoo Finance kurang dari 10 saham. Menggunakan data simulasi.")
            np.random.seed(42)
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='B')
            for ticker in STOCKS_DATA:
                base, trend, vol = np.random.uniform(500, 20000), np.random.normal(0.05, 0.20)/252, np.random.uniform(0.02, 0.06)
                data[ticker] = pd.Series(base * np.exp(np.cumsum(np.random.normal(trend, vol, len(dates)))), index=dates)
            successful = list(STOCKS_DATA.keys())
        status.update(label=f"✅ {len(successful)} saham berhasil dimuat", state="complete")

    short_names = {t: t.replace('.JK', '') for t in successful}
    series_dict = {}
    for t in successful:
        s = pd.Series(data[t].values, index=pd.to_datetime(data[t].index), name=short_names[t])
        series_dict[short_names[t]] = s
    df_prices = pd.concat(series_dict, axis=1).dropna(how='all')

    # Feature Engineering
    df_ret = np.log(df_prices / df_prices.shift(1)).dropna()
    feats = pd.DataFrame(index=df_ret.columns)
    feats['daily_return'] = df_ret.mean()
    feats['annual_return'] = feats['daily_return'] * 252
    feats['volatility'] = df_ret.std() * np.sqrt(252)
    feats['sharpe_ratio'] = (feats['annual_return'] - 0.05) / feats['volatility']
    feats['max_drawdown'] = [(df_prices[s] - df_prices[s].expanding().max()).min() / df_prices[s].expanding().max() for s in df_prices.columns]
    feats['skewness'] = df_ret.skew()
    feats['kurtosis'] = df_ret.kurtosis()
    mkt = df_ret.mean(axis=1)
    feats['beta'] = [df_ret[s].cov(mkt) / mkt.var() for s in df_ret.columns]
    feats['positive_days'] = (df_ret > 0).mean()

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    return feats, feats_scaled, df_prices, successful

# ============================================
# MAIN APP
# ============================================
st.title("📊 Perbandingan 3 Algoritma Clustering Saham Indonesia")
st.markdown("Analisis **27 saham** dari berbagai sektor menggunakan **HDBSCAN**, **DBSCAN**, dan **GMM**.")

# Run cached pipeline
features, features_scaled, df_prices, successful_tickers = load_data_and_compute()

# --- CLUSTERING ---
with st.spinner("🔍 Menjalankan HDBSCAN, DBSCAN, dan GMM..."):
    # HDBSCAN
    best_h_labels, best_h_n, noise_h = None, 0, 0
    for min_cs in [2, 3, 4]:
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cs, min_samples=1, metric='euclidean', cluster_selection_method='eom').fit_predict(features_scaled)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        if n > best_h_n: best_h_labels, best_h_n, noise_h = labels, n, (labels == -1).sum()
    features['hdbscan'] = best_h_labels

    # DBSCAN
    best_d_labels, best_d_n, noise_d, best_eps = None, 0, 0, 1.5
    for eps in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(features_scaled)
        n = len(set(labels)) - (1 if -1 in labels else 0)
        if 2 <= n <= 7 and n > best_d_n: best_d_labels, best_d_n, noise_d, best_eps = labels, n, (labels == -1).sum(), eps
    if best_d_labels is None: best_d_labels = DBSCAN(eps=1.5, min_samples=2).fit_predict(features_scaled)
    features['dbscan'] = best_d_labels
    noise_d = (best_d_labels == -1).sum()

    # GMM
    best_g_labels, best_g_n, best_bic = None, 2, float('inf')
    for n in range(2, min(8, len(features))):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42, n_init=5)
        gmm.fit(features_scaled)
        bic = gmm.bic(features_scaled)
        if bic < best_bic: best_g_labels, best_g_n, best_bic = gmm.predict(features_scaled), n, bic
    features['gmm'] = best_g_labels

# --- VISUALISASI ---
st.header("📉 Visualisasi PCA (2D)")
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(features_scaled)
var_exp = pca.explained_variance_ratio_
PALETTE = ['#E63946', '#457B9D', '#2DC653', '#F4A261', '#9B5DE5', '#F72585', '#4CC9F0', '#FB8500']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor('#0D1117')
for ax in axes: ax.set_facecolor('#161B22')

def plot_cluster(ax, labels, title):
    for c in sorted(set(labels)):
        mask = labels == c
        color = '#555' if c == -1 else PALETTE[c % len(PALETTE)]
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=color, label=f'{"Noise" if c==-1 else f"Cluster {c}"}', s=100, alpha=0.85, edgecolors='white', linewidths=0.5)
    ax.set_title(title, color='white', fontsize=13, pad=10, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', color='#8B949E')
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', color='#8B949E')
    ax.legend(facecolor='#21262D', edgecolor='#30363D', labelcolor='white', fontsize=8)

plot_cluster(axes[0], best_h_labels, f'HDBSCAN | {best_h_n} Cluster | {noise_h} Noise')
plot_cluster(axes[1], best_d_labels, f'DBSCAN (ε={best_eps}) | {best_d_n} Cluster | {noise_d} Noise')
plot_cluster(axes[2], best_g_labels, f'GMM | K={best_g_n} | 0 Noise')
plt.tight_layout()
st.pyplot(fig)

# --- METRIK & TABEL ---
st.header("📊 Perbandingan Metrik Evaluasi")
def safe_sil(labels):
    u = set(labels)
    if -1 in u: mask = labels != -1; return silhouette_score(features_scaled[mask], labels[mask]) if mask.sum()>1 and len(set(labels[mask]))>1 else None
    return silhouette_score(features_scaled, labels) if len(u)>1 else None
def safe_dbi(labels):
    u = set(labels)
    if -1 in u: mask = labels != -1; return davies_bouldin_score(features_scaled[mask], labels[mask]) if mask.sum()>1 and len(set(labels[mask]))>1 else None
    return davies_bouldin_score(features_scaled, labels) if len(u)>1 else None

sil_h, sil_d, sil_g = safe_sil(best_h_labels), safe_sil(best_d_labels), silhouette_score(features_scaled, best_g_labels)
dbi_h, dbi_d, dbi_g = safe_dbi(best_h_labels), safe_dbi(best_d_labels), davies_bouldin_score(features_scaled, best_g_labels)

df_cmp = pd.DataFrame([
    ['HDBSCAN', best_h_n, noise_h, f"{sil_h:.3f}" if sil_h else "N/A", f"{dbi_h:.3f}" if dbi_h else "N/A"],
    ['DBSCAN',  best_d_n, noise_d, f"{sil_d:.3f}" if sil_d else "N/A", f"{dbi_d:.3f}" if dbi_d else "N/A"],
    ['GMM',     best_g_n, 0,       f"{sil_g:.3f}",                      f"{dbi_g:.3f}"]
], columns=['Algoritma', 'Cluster', 'Noise', 'Silhouette ↑', 'DBI ↓'])
st.dataframe(df_cmp, use_container_width=True)
st.caption("Silhouette: makin tinggi makin baik | DBI: makin rendah makin baik")

# --- ANALISIS CLUSTER (HDBSCAN) ---
st.header("📈 Analisis Return per Cluster (HDBSCAN)")
for c in sorted(set(best_h_labels)):
    grp = features[features['hdbscan'] == c]
    if c == -1: st.markdown(f"⚠️ **NOISE** ({len(grp)} saham)")
    else: st.markdown(f"🔷 **CLUSTER {c}** ({len(grp)} saham) | Avg Return: {grp['annual_return'].mean():.2%} | Avg Vol: {grp['volatility'].mean():.2%} | Avg Sharpe: {grp['sharpe_ratio'].mean():.2f}")
    for stock, row in grp.iterrows():
        st.markdown(f"`{stock}` R={row['annual_return']:.2%} | σ={row['volatility']:.2%} | Sharpe={row['sharpe_ratio']:.2f} | β={row['beta']:.2f}")

# --- REKOMENDASI ---
st.header("💡 Rekomendasi Investasi (Berdasarkan Sharpe Ratio)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🏆 TOP 5 Terbaik")
    st.dataframe(features.nlargest(5, 'sharpe_ratio')[['annual_return','volatility','sharpe_ratio','beta']], use_container_width=True)
with col2:
    st.subheader("⚠️ BOTTOM 5 Terburuk")
    st.dataframe(features.nsmallest(5, 'sharpe_ratio')[['annual_return','volatility','sharpe_ratio','beta']], use_container_width=True)

# --- EXPORT ---
st.header("📥 Export Data")
csv_data = features.copy()
csv_data.index.name = 'Saham'
st.download_button(
    label="⬇️ Download hasil_clustering.csv",
    data=csv_data.to_csv().encode('utf-8'),
    file_name='hasil_clustering.csv',
    mime='text/csv',
    use_container_width=True
)
st.success("✅ Analisis selesai. Data siap diekspor atau digunakan untuk strategi portfolio.")
