import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="E-Commerce Olist Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #f5f7fa; }

    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2563eb;
    }
    [data-testid="metric-container"] label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #111827;
        font-size: 26px;
        font-weight: 700;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }
    .section-sub {
        font-size: 13px;
        color: #6b7280;
        margin-top: -4px;
        margin-bottom: 16px;
        font-style: italic;
    }
    .insight-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 14px;
        color: #1e3a8a;
        margin-top: 8px;
    }
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── WARNA ──────────────────────────────────────────────────────────────────
C_BLUE   = '#2563eb'
C_RED    = '#ef4444'
C_GREEN  = '#10b981'
C_ORANGE = '#f59e0b'
C_PURPLE = '#8b5cf6'
C_GRAY   = '#94a3b8'
C_BG     = '#f8fafc'

WARNA_SEGMEN = {
    'Champions':          '#10b981',
    'Loyal Customers':    '#2563eb',
    'Potential Loyalist': '#f59e0b',
    'At Risk':            '#f97316',
    'Lost':               '#ef4444',
    'New Customers':      '#8b5cf6',
}

def style_ax(ax, bg=C_BG):
    ax.set_facecolor(bg)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.xaxis.label.set_color('#374151')
    ax.yaxis.label.set_color('#374151')
    ax.title.set_color('#111827')
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7, alpha=0.7)


# ── LOAD DATA ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Cari file di folder dashboard/ atau direktori saat ini
    base = os.path.dirname(__file__)

    def path(fname):
        p1 = os.path.join(base, fname)
        p2 = fname
        return p1 if os.path.exists(p1) else p2

    df_orders    = pd.read_csv(path("../data/orders_dataset.csv"))
    df_items     = pd.read_csv(path("../data/order_items_dataset.csv"))
    df_payments  = pd.read_csv(path("../data/order_payments_dataset.csv"))
    df_reviews   = pd.read_csv(path("../data/order_reviews_dataset.csv"))
    df_customers = pd.read_csv(path("../data/customers_dataset.csv"))
    df_products  = pd.read_csv(path("../data/products_dataset.csv"))
    df_category  = pd.read_csv(path("../data/product_category_name_translation.csv"))

    for kol in ['order_purchase_timestamp', 'order_approved_at',
                'order_delivered_carrier_date', 'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        df_orders[kol] = pd.to_datetime(df_orders[kol], errors='coerce')

    df_products['product_category_name'].fillna(
        df_products['product_category_name'].mode()[0], inplace=True)
    df_products['product_weight_g'].fillna(
        df_products['product_weight_g'].median(), inplace=True)

    df_prod  = df_products.merge(df_category, on='product_category_name', how='left')
    df_items = df_items.merge(
        df_prod[['product_id', 'product_category_name_english']],
        on='product_id', how='left')

    df = df_orders.merge(df_items, on='order_id', how='inner')
    df = df.merge(df_reviews[['order_id', 'review_score']], on='order_id', how='left')
    df = df.merge(
        df_customers[['customer_id', 'customer_state', 'customer_unique_id']],
        on='customer_id', how='left')
    df = df.merge(
        df_payments[['order_id', 'payment_type', 'payment_value']],
        on='order_id', how='left')

    df['revenue']     = df['price'] + df['freight_value']
    df['order_month'] = df['order_purchase_timestamp'].dt.strftime('%Y-%m')
    df['year']        = df['order_purchase_timestamp'].dt.year

    return df


df_all = load_data()
df     = df_all[df_all['order_status'] == 'delivered'].copy()


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Olist Dashboard")
    st.markdown("**E-Commerce Analytics**")
    st.markdown("---")
    st.markdown("### 🔽 Filter Data")

    tahun_list  = sorted(df['year'].dropna().unique().astype(int).tolist())
    tahun_pilih = st.multiselect("Tahun", tahun_list, default=tahun_list)

    state_list  = sorted(df['customer_state'].dropna().unique().tolist())
    state_pilih = st.multiselect("Negara Bagian", state_list, default=state_list)

    st.markdown("---")
    st.markdown("**Dataset:** Brazilian E-Commerce  \nPublic Dataset by Olist")
    st.markdown("**Periode:** Sep 2016 – Ags 2018")
    st.markdown("---")
    st.markdown("Regina Arija Zuhra")

if tahun_pilih and state_pilih:
    df_f = df[df['year'].isin(tahun_pilih) & df['customer_state'].isin(state_pilih)].copy()
else:
    df_f = df.copy()


# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("# 🛒 E-Commerce Olist — Dashboard Analisis")
st.markdown("Analisis tren pendapatan, kepuasan pelanggan, segmentasi RFM, dan distribusi geografis platform Olist Brasil.")
st.markdown("---")


# ── KPI METRICS ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Pendapatan",  f"R$ {df_f['revenue'].sum() / 1e6:.2f}M")
c2.metric("📦 Total Pesanan",     f"{df_f['order_id'].nunique():,}")
c3.metric("⭐ Rata-rata Ulasan",  f"{df_f['review_score'].mean():.2f} / 5")
c4.metric("👤 Total Pelanggan",   f"{df_f['customer_id'].nunique():,}")

st.markdown("---")


# ── VIZ 1: TREN PENDAPATAN ─────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 Pertanyaan 1 — Tren Pendapatan Bulanan</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Bagaimana tren total pendapatan bulanan Olist dari September 2016 hingga Agustus 2018?</p>', unsafe_allow_html=True)

monthly = (
    df_f
    .groupby('order_month')
    .agg(total_revenue=('revenue', 'sum'), total_orders=('order_id', 'nunique'))
    .reset_index()
    .sort_values('order_month')
)
monthly['growth'] = monthly['total_revenue'].pct_change() * 100

col_viz1, col_info1 = st.columns([3, 1])

with col_viz1:
    if len(monthly) > 0:
        idx_max = monthly['total_revenue'].idxmax()
        warna   = [C_RED if i == idx_max else C_BLUE for i in range(len(monthly))]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
        fig.patch.set_facecolor(C_BG)

        # panel atas
        axes[0].bar(monthly['order_month'], monthly['total_revenue'] / 1e6,
                    color=warna, width=0.6, alpha=0.85, zorder=3)
        axes[0].plot(monthly['order_month'], monthly['total_revenue'] / 1e6,
                     color='#1e293b', linewidth=1.8, marker='o', markersize=5, zorder=4)

        rata = monthly['total_revenue'].mean() / 1e6
        axes[0].axhline(rata, color=C_GREEN, linestyle='--', linewidth=1.3,
                        label=f'Rata-rata: R$ {rata:.2f}M', zorder=2)

        np_val = monthly.loc[idx_max, 'total_revenue'] / 1e6
        np_lbl = monthly.loc[idx_max, 'order_month']
        axes[0].annotate(
            f'Puncak\nR$ {np_val:.2f}M',
            xy=(idx_max, np_val),
            xytext=(max(0, idx_max - 3), np_val + 0.08),
            fontsize=9, fontweight='bold', color=C_RED,
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.3)
        )

        axes[0].set_ylabel('Total Pendapatan (Juta R$)', fontsize=10)
        axes[0].set_title('Total Pendapatan Bulanan — E-Commerce Olist (Sep 2016 – Ags 2018)',
                          fontsize=13, fontweight='bold', pad=12)
        axes[0].set_xticks(range(len(monthly)))
        axes[0].set_xticklabels(monthly['order_month'], rotation=45, ha='right', fontsize=8)
        axes[0].legend(fontsize=9)
        style_ax(axes[0])

        # panel bawah — growth
        g_colors = [C_GREEN if v >= 0 else C_RED for v in monthly['growth'].fillna(0)]
        axes[1].bar(range(len(monthly)), monthly['growth'].fillna(0),
                    color=g_colors, width=0.6, alpha=0.8)
        axes[1].axhline(0, color='#374151', linewidth=0.8)
        axes[1].set_ylabel('Growth (%)', fontsize=9)
        axes[1].set_xticks(range(len(monthly)))
        axes[1].set_xticklabels(monthly['order_month'], rotation=45, ha='right', fontsize=8)
        style_ax(axes[1])
        axes[1].grid(axis='y', color='#e5e7eb', linewidth=0.7, alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col_info1:
    st.markdown("#### 📊 Statistik")
    if len(monthly) > 0:
        st.metric("Pendapatan Tertinggi", f"R$ {monthly['total_revenue'].max()/1e6:.2f}M",
                  delta=monthly.loc[idx_max, 'order_month'])
        st.metric("Pendapatan Terendah",  f"R$ {monthly['total_revenue'].min()/1e6:.2f}M")
        st.metric("Rata-rata / Bulan",    f"R$ {rata:.2f}M")
        st.metric("Total Bulan Aktif",    f"{len(monthly)} bulan")
        st.markdown("---")
        st.markdown("#### 📋 Data Bulanan")
        tabel = monthly[['order_month', 'total_revenue', 'total_orders']].copy()
        tabel.columns = ['Bulan', 'Pendapatan (R$)', 'Pesanan']
        tabel = tabel.set_index('Bulan')
        st.dataframe(tabel, height=280)

if len(monthly) > 0:
    st.markdown(
        f'<div class="insight-box">💡 <b>Insight:</b> Pendapatan tertinggi terjadi pada <b>{np_lbl}</b> '
        f'sebesar <b>R$ {np_val:.2f}M</b>, diduga kuat dipicu oleh event Black Friday. '
        f'Tren pendapatan tumbuh konsisten sepanjang 2017, dan stabil di kisaran R$ 1M per bulan sejak awal 2018.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ── VIZ 2: REVIEW KATEGORI ─────────────────────────────────────────────────
st.markdown('<p class="section-title">⭐ Pertanyaan 2 — Skor Ulasan per Kategori Produk</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Kategori produk mana yang mendapat ulasan tertinggi dan terendah? Apakah harga berpengaruh terhadap kepuasan pelanggan?</p>', unsafe_allow_html=True)

df_kat   = df_f.dropna(subset=['product_category_name_english', 'review_score'])
kat_stats = (
    df_kat
    .groupby('product_category_name_english')
    .agg(
        avg_review=('review_score', 'mean'),
        count_review=('review_score', 'count'),
        avg_price=('price', 'mean')
    )
    .reset_index()
)
kat_stats = kat_stats[kat_stats['count_review'] >= 20].sort_values('avg_review', ascending=False)

n_show = st.slider("Tampilkan N kategori terbaik + terburuk:", 3, 10, 5)
top_n  = kat_stats.head(n_show)
bot_n  = kat_stats.tail(n_show)
df_v2  = pd.concat([top_n, bot_n]).drop_duplicates().sort_values('avg_review', ascending=True)

if len(df_v2) > 0:
    col_v2a, col_v2b = st.columns(2)

    with col_v2a:
        fig2, ax2 = plt.subplots(figsize=(7, max(5, len(df_v2) * 0.45)))
        fig2.patch.set_facecolor(C_BG)

        median_val = kat_stats['avg_review'].median()
        warna2     = [C_RED if v < median_val else C_BLUE for v in df_v2['avg_review']]

        bars = ax2.barh(df_v2['product_category_name_english'], df_v2['avg_review'],
                        color=warna2, height=0.65, alpha=0.88)
        ax2.axvline(median_val, color=C_GRAY, linestyle='--', linewidth=1.2,
                    label=f'Median: {median_val:.2f}')

        for bar, val in zip(bars, df_v2['avg_review']):
            ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=8, fontweight='bold')

        patch_baik  = mpatches.Patch(color=C_BLUE, label='Di atas median')
        patch_buruk = mpatches.Patch(color=C_RED,  label='Di bawah median')
        ax2.legend(handles=[patch_baik, patch_buruk], fontsize=8, loc='lower right')

        ax2.set_xlabel('Rata-rata Skor Ulasan (1–5)', fontsize=10)
        ax2.set_title('Rata-rata Skor Ulasan per Kategori Produk\n(merah = di bawah median, biru = di atas median)',
                      fontsize=12, fontweight='bold')
        ax2.set_xlim(3, 5.4)
        style_ax(ax2)
        ax2.grid(axis='x', color='#e5e7eb', linewidth=0.7)
        ax2.grid(axis='y', visible=False)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_v2b:
        fig3, ax3 = plt.subplots(figsize=(7, max(5, len(df_v2) * 0.45)))
        fig3.patch.set_facecolor(C_BG)

        sc = ax3.scatter(
            kat_stats['avg_price'], kat_stats['avg_review'],
            s=kat_stats['count_review'] * 0.5,
            c=kat_stats['avg_review'],
            cmap='RdYlGn', alpha=0.75,
            edgecolors='#94a3b8', linewidth=0.5,
            vmin=3.5, vmax=5.0
        )
        plt.colorbar(sc, ax=ax3, label='Skor Ulasan')

        for _, row in kat_stats.iterrows():
            ax3.annotate(
                row['product_category_name_english'][:14],
                (row['avg_price'], row['avg_review']),
                fontsize=6.5, xytext=(4, 3), textcoords='offset points', color='#374151'
            )

        z      = np.polyfit(kat_stats['avg_price'], kat_stats['avg_review'], 1)
        p      = np.poly1d(z)
        x_line = np.linspace(kat_stats['avg_price'].min(), kat_stats['avg_price'].max(), 100)
        ax3.plot(x_line, p(x_line), color=C_RED, linestyle='--', linewidth=1.3,
                 label='Garis tren', alpha=0.8)
        ax3.legend(fontsize=9)

        ax3.set_xlabel('Rata-rata Harga Produk (R$)', fontsize=10)
        ax3.set_ylabel('Rata-rata Skor Ulasan', fontsize=10)
        ax3.set_title('Hubungan Harga vs Skor Ulasan per Kategori\n(ukuran titik = jumlah ulasan)',
                      fontsize=12, fontweight='bold')
        style_ax(ax3)
        ax3.grid(color='#e5e7eb', linewidth=0.7)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    best  = kat_stats.iloc[0]
    worst = kat_stats.iloc[-1]
    kor   = kat_stats[['avg_review', 'avg_price']].corr().iloc[0, 1]

    st.markdown(
        f'<div class="insight-box">💡 <b>Insight:</b> Kategori <b>{best["product_category_name_english"]}</b> '
        f'mendapat ulasan tertinggi ({best["avg_review"]:.2f}), sedangkan '
        f'<b>{worst["product_category_name_english"]}</b> terendah ({worst["avg_review"]:.2f}). '
        f'Korelasi harga vs ulasan hanya <b>{kor:.3f}</b> — harga bukan penentu utama kepuasan pelanggan.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# ── RFM ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🎯 Pertanyaan 3 — Segmentasi Pelanggan (RFM Analysis)</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Bagaimana segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary selama Sep 2016 – Ags 2018?</p>', unsafe_allow_html=True)


@st.cache_data
def hitung_rfm(df_input):
    tgl_ref = df_input['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = (
        df_input
        .groupby('customer_unique_id')
        .agg(
            Recency=('order_purchase_timestamp', lambda x: (tgl_ref - x.max()).days),
            Frequency=('order_id', 'nunique'),
            Monetary=('revenue', 'sum')
        )
        .reset_index()
    )
    rfm['R'] = pd.qcut(rfm['Recency'],  q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    rfm['M'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)

    def segmen(row):
        r, f, m = row['R'], row['F'], row['M']
        if r >= 4 and f >= 4 and m >= 4:         return 'Champions'
        elif r >= 4 and f >= 3:                   return 'Loyal Customers'
        elif r >= 3 and f >= 3 and m >= 3:        return 'Potential Loyalist'
        elif r <= 2 and f >= 3:                   return 'At Risk'
        elif r <= 2 and f <= 2 and m <= 2:        return 'Lost'
        else:                                     return 'New Customers'

    rfm['segment'] = rfm.apply(segmen, axis=1)
    return rfm


df_rfm = hitung_rfm(df)

col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    seg_count = df_rfm['segment'].value_counts()
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    fig4.patch.set_facecolor(C_BG)
    warna_seg = [WARNA_SEGMEN.get(s, C_BLUE) for s in seg_count.index]
    bars4 = ax4.bar(seg_count.index, seg_count.values, color=warna_seg, alpha=0.88)
    for bar, val in zip(bars4, seg_count.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{val:,}', ha='center', fontsize=8, fontweight='bold', color='#374151')
    ax4.set_title('Distribusi Segmen Pelanggan', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Jumlah Pelanggan')
    ax4.tick_params(axis='x', rotation=25)
    style_ax(ax4)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

with col_r2:
    avg_m = df_rfm.groupby('segment')['Monetary'].mean().sort_values(ascending=True)
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    fig5.patch.set_facecolor(C_BG)
    warna_m = [WARNA_SEGMEN.get(s, C_BLUE) for s in avg_m.index]
    ax5.barh(avg_m.index, avg_m.values, color=warna_m, alpha=0.88)
    for i, val in enumerate(avg_m.values):
        ax5.text(val + 1, i, f'R${val:,.0f}', va='center', fontsize=8)
    ax5.set_title('Rata-rata Belanja per Segmen', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Rata-rata Belanja (R$)')
    style_ax(ax5)
    ax5.grid(axis='x', color='#e5e7eb', linewidth=0.7)
    ax5.grid(axis='y', visible=False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

with col_r3:
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    fig6.patch.set_facecolor(C_BG)
    for segmen_name, warna in WARNA_SEGMEN.items():
        sub = df_rfm[df_rfm['segment'] == segmen_name]
        if len(sub) > 0:
            ax6.scatter(sub['Recency'], sub['Monetary'],
                        c=warna, alpha=0.45, s=12, label=segmen_name)
    ax6.set_xlabel('Recency (hari)', fontsize=9)
    ax6.set_ylabel('Monetary (R$)', fontsize=9)
    ax6.set_title('Peta Segmen: Recency vs Monetary', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=7, loc='upper right', framealpha=0.8)
    style_ax(ax6)
    ax6.grid(color='#e5e7eb', linewidth=0.7)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

rfm_summary = (
    df_rfm.groupby('segment')
    .agg(
        Jumlah_Pelanggan=('customer_unique_id', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean')
    )
    .reset_index()
    .sort_values('Avg_Monetary', ascending=False)
    .rename(columns={'segment': 'Segmen'})
)
rfm_summary['Avg_Recency']   = rfm_summary['Avg_Recency'].round(0).astype(int).astype(str) + ' hari'
rfm_summary['Avg_Frequency'] = rfm_summary['Avg_Frequency'].round(1)
rfm_summary['Avg_Monetary']  = rfm_summary['Avg_Monetary'].round(2).apply(lambda x: f'R$ {x:,.2f}')
st.dataframe(rfm_summary.set_index('Segmen'), use_container_width=True)

st.markdown(
    '<div class="insight-box">💡 <b>Insight:</b> Mayoritas pelanggan (93.358 unik) hanya melakukan '
    '1 kali pembelian. Segmen terbesar adalah <b>New Customers</b> dan <b>At Risk</b>, '
    'mengindikasikan rendahnya tingkat retensi. <b>Champions</b> hanya ~6.9% dari total pelanggan '
    'namun menjadi kontributor pendapatan terbesar.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ── GEOSPATIAL ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🗺️ Pertanyaan 4 — Distribusi Geografis Pesanan</p>', unsafe_allow_html=True)
st.markdown('<p class="section-sub">Wilayah mana yang menghasilkan pendapatan tertinggi dan memiliki jumlah pelanggan terbanyak selama Sep 2016 – Ags 2018?</p>', unsafe_allow_html=True)

coords = {
    'SP': (-23.5, -46.6), 'RJ': (-22.9, -43.2), 'MG': (-19.9, -43.9),
    'RS': (-30.0, -51.2), 'PR': (-25.4, -49.3), 'SC': (-27.6, -48.5),
    'BA': (-12.9, -38.5), 'GO': (-16.7, -49.3), 'PE': (-8.1,  -34.9),
    'CE': (-3.7,  -38.5), 'ES': (-19.2, -40.3), 'MT': (-12.6, -55.9),
    'MS': (-20.4, -54.6), 'PB': (-7.2,  -36.8), 'RN': (-5.8,  -36.5),
    'AL': (-9.7,  -36.6), 'MA': (-5.4,  -45.4), 'PI': (-8.0,  -43.0),
    'PA': (-3.4,  -52.0), 'AM': (-4.4,  -63.6), 'DF': (-15.8, -47.9)
}

geo_stats = (
    df_f
    .groupby('customer_state')
    .agg(total_orders=('order_id', 'nunique'), total_revenue=('revenue', 'sum'))
    .reset_index()
    .sort_values('total_orders', ascending=False)
)
geo_stats['lat'] = geo_stats['customer_state'].map(lambda x: coords.get(x, (-15, -50))[0])
geo_stats['lon'] = geo_stats['customer_state'].map(lambda x: coords.get(x, (-15, -50))[1])

col_g1, col_g2 = st.columns([3, 2])

with col_g1:
    ukuran = geo_stats['total_orders'] / geo_stats['total_orders'].max() * 2000
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    fig7.patch.set_facecolor('#0f172a')
    ax7.set_facecolor('#0f172a')

    sc7 = ax7.scatter(
        geo_stats['lon'], geo_stats['lat'],
        s=ukuran, c=geo_stats['total_orders'],
        cmap='YlOrRd', alpha=0.85,
        edgecolors='white', linewidth=0.6
    )
    cb7 = plt.colorbar(sc7, ax=ax7, shrink=0.7)
    cb7.set_label('Jumlah Pesanan', color='white', fontsize=9)
    cb7.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cb7.ax.axes, 'yticklabels'), color='white')

    for _, row in geo_stats.iterrows():
        ax7.text(row['lon'], row['lat'], row['customer_state'],
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold')

    ax7.set_xlim(-75, -30)
    ax7.set_ylim(-35, 5)
    ax7.set_title('Peta Distribusi Pesanan per Negara Bagian',
                  fontsize=12, fontweight='bold', color='white', pad=12)
    ax7.set_xlabel('Longitude', color='#94a3b8', fontsize=9)
    ax7.set_ylabel('Latitude',  color='#94a3b8', fontsize=9)
    ax7.tick_params(colors='#64748b')
    ax7.grid(color='#1e3a5f', linewidth=0.5, alpha=0.6)
    for spine in ax7.spines.values():
        spine.set_edgecolor('#1e3a5f')

    plt.tight_layout()
    st.pyplot(fig7)
    plt.close()

with col_g2:
    top10 = geo_stats.head(10)

    fig8, axes8 = plt.subplots(2, 1, figsize=(6, 7))
    fig8.patch.set_facecolor(C_BG)

    warna_geo = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))[::-1]
    axes8[0].barh(top10['customer_state'], top10['total_orders'],
                  color=warna_geo, alpha=0.9)
    axes8[0].set_title('Top 10 — Jumlah Pesanan', fontsize=10, fontweight='bold')
    axes8[0].invert_yaxis()
    for i, val in enumerate(top10['total_orders']):
        axes8[0].text(val + 5, i, f'{val:,}', va='center', fontsize=8)
    style_ax(axes8[0])
    axes8[0].grid(axis='x', color='#e5e7eb', linewidth=0.7)
    axes8[0].grid(axis='y', visible=False)

    warna_geo2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top10)))[::-1]
    axes8[1].barh(top10['customer_state'], top10['total_revenue'] / 1e3,
                  color=warna_geo2, alpha=0.9)
    axes8[1].set_title('Top 10 — Total Pendapatan (Ribu R$)', fontsize=10, fontweight='bold')
    axes8[1].set_xlabel('Pendapatan (Ribu R$)')
    axes8[1].invert_yaxis()
    for i, val in enumerate(top10['total_revenue'] / 1e3):
        axes8[1].text(val + 0.5, i, f'R${val:,.0f}K', va='center', fontsize=8)
    style_ax(axes8[1])
    axes8[1].grid(axis='x', color='#e5e7eb', linewidth=0.7)
    axes8[1].grid(axis='y', visible=False)

    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

st.markdown(
    '<div class="insight-box">💡 <b>Insight:</b> Pesanan sangat terkonsentrasi di <b>SP (São Paulo)</b> '
    'dengan 40.501 pesanan dan pendapatan R$ 6,07 juta. Wilayah timur laut Brasil seperti CE, PE, dan BA '
    'masih memiliki potensi ekspansi yang besar.</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ── KESIMPULAN ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📝 Kesimpulan & Rekomendasi</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pertanyaan 1", "Pertanyaan 2", "Pertanyaan 3", "Pertanyaan 4", "Rekomendasi"])

with tab1:
    st.markdown("""
    Tren pendapatan bulanan Olist tumbuh konsisten dari September 2016 hingga Agustus 2018.
    Puncak pendapatan terjadi pada **November 2017** sebesar R$ 1.205.033 dengan 7.289 order,
    yang diduga kuat dipicu oleh event Black Friday. Sejak awal 2018, pendapatan stabil
    di kisaran R$ 1 juta per bulan, menunjukkan platform telah mencapai kematangan pasar.
    """)

with tab2:
    st.markdown("""
    Kategori **books_imported** dan **books_general_interest** memiliki skor ulasan tertinggi
    (4.53 dan 4.50), sedangkan **diapers_and_hygiene** dan **office_furniture** memiliki skor
    terendah (3.38 dan 3.55). Korelasi antara harga dan skor ulasan sangat lemah (0.138),
    membuktikan bahwa harga produk bukan faktor penentu utama kepuasan pelanggan.
    """)

with tab3:
    st.markdown("""
    Segmentasi RFM terhadap 93.358 pelanggan unik menunjukkan mayoritas hanya melakukan
    1 kali pembelian. Segmen terbesar adalah **New Customers** (35.611) dan **At Risk** (22.230),
    mengindikasikan rendahnya tingkat retensi. **Champions** (6.463 pelanggan) menjadi
    kontributor pendapatan terbesar dengan rata-rata belanja tertinggi.
    """)

with tab4:
    st.markdown("""
    Distribusi pesanan sangat terkonsentrasi di wilayah tenggara Brasil, khususnya
    **São Paulo (SP)** dengan 40.501 pesanan dan pendapatan R$ 6,07 juta.
    Wilayah barat dan utara Brasil hampir tidak memiliki aktivitas transaksi,
    menunjukkan potensi ekspansi pasar yang sangat besar.
    """)

with tab5:
    st.markdown("""
    - Manfaatkan momentum **Black Friday** dan akhir tahun dengan mempersiapkan stok dan promosi lebih awal.
    - Lakukan **quality control** pada kategori dengan ulasan rendah seperti *diapers_and_hygiene* dan *office_furniture*.
    - Rancang **program retensi** per segmen RFM: reward eksklusif untuk Champions, re-engagement untuk At Risk & Lost, welcome offer untuk New Customers.
    - Fokuskan **ekspansi logistik dan marketing** ke wilayah utara dan barat Brasil yang belum tergarap.
    """)

st.markdown("---")
st.markdown(
    "<center><small>Dashboard E-Commerce Olist · Regina Arija Zuhra · "
    "Dibuat dengan Streamlit & Matplotlib</small></center>",
    unsafe_allow_html=True
)
