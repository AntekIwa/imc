import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# --- KONFIGURACJA STRONY I ESTETYKI ---
st.set_page_config(page_title="IMC Prosperity 4 - Quant Terminal", layout="wide", page_icon="📈")

# Ustawienia profesjonalnego wyglądu wykresów (stylizacja Bloomberg-like)
plt.style.use('dark_background')
sns.set_context("notebook", font_scale=1.1)
COLOR_UP = '#00ff00'
COLOR_DOWN = '#ff0000'
COLOR_NEUTRAL = '#00ccff'

@st.cache_data
def load_data(file, sep):
    if file is not None:
        try:
            return pd.read_csv(file, sep=sep)
        except Exception as e:
            st.error(f"Błąd wczytywania pliku: {e}")
    return None

# --- SIDEBAR: WGRYWANIE DANYCH ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Sars-cov-2_SARS-CoV-2_spike_protein.png/120px-Sars-cov-2_SARS-CoV-2_spike_protein.png", width=50) # Placeholder logo
    st.header("⚙️ Terminal Danych")
    separator = st.selectbox("Separator CSV", [";", ","])
    prices_file = st.file_uploader("1. Wgraj Prices (prices.csv)", type=['csv'])
    trades_file = st.file_uploader("2. Wgraj Trades (trades.csv)", type=['csv'])
    st.markdown("---")
    st.markdown("### 🛠️ Opcje Wykresów")
    chart_ma_fast = st.slider("Szybka EMA", 5, 50, 10)
    chart_ma_slow = st.slider("Wolna EMA", 20, 200, 50)

prices_df = load_data(prices_file, separator)
trades_df = load_data(trades_file, separator)

st.title("📈 IMC Prosperity 4 - Advanced Quant Terminal")
st.markdown("Kompleksowe narzędzie do analizy mikrostruktury rynku, arbitrażu statystycznego i zachowań algorytmicznych (Order Flow).")

if prices_df is not None:
    products = prices_df['product'].unique().tolist()
    
    # Przebudowany system zakładek na 12 potężnych modułów
    tabs = st.tabs([
        "🕯️ Price Action & Zmienność", 
        "📊 Volume Profile & CVD", 
        "🔗 Arbitraż & Korelacje", 
        "⏱️ Lead-Lag Analysis",
        "⚖️ Order Book Imbalance", 
        "🕵️ Tape Reading (ID)", 
        "🤖 Bot Fingerprint (Blind)", 
        "🌊 Sygnały & Obserwacje", 
        "📈 Opcje (Black-Scholes)",
        "📚 Deep L2/L3 VWAP",
        "🔄 Autokorelacja (ACF)",
        "💰 Backtester Sandbox"
    ])

    # ==========================================
    # ZAKŁADKA 1: PRICE ACTION & BOLLINGER BANDS
    # ==========================================
    with tabs[0]:
        st.header("Analiza Ceny, Trendu i Zmienności")
        prod_1 = st.selectbox("Wybierz aktywo:", products, key='t1_prod')
        df_p1 = prices_df[prices_df['product'] == prod_1].copy()
        
        # Obliczenia techniczne
        df_p1['EMA_Fast'] = df_p1['mid_price'].ewm(span=chart_ma_fast).mean()
        df_p1['EMA_Slow'] = df_p1['mid_price'].ewm(span=chart_ma_slow).mean()
        bb_window = 20
        df_p1['BB_Mid'] = df_p1['mid_price'].rolling(window=bb_window).mean()
        df_p1['BB_Std'] = df_p1['mid_price'].rolling(window=bb_window).std()
        df_p1['BB_Upper'] = df_p1['BB_Mid'] + (df_p1['BB_Std'] * 2)
        df_p1['BB_Lower'] = df_p1['BB_Mid'] - (df_p1['BB_Std'] * 2)

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Główny wykres ceny
        ax1.plot(df_p1['timestamp'], df_p1['mid_price'], color='white', label='Mid Price', linewidth=1.5)
        ax1.plot(df_p1['timestamp'], df_p1['EMA_Fast'], color='#00ffcc', label=f'EMA {chart_ma_fast}', alpha=0.8)
        ax1.plot(df_p1['timestamp'], df_p1['EMA_Slow'], color='#ff00ff', label=f'EMA {chart_ma_slow}', alpha=0.8)
        ax1.fill_between(df_p1['timestamp'], df_p1['BB_Lower'], df_p1['BB_Upper'], color='#336699', alpha=0.2, label='Bollinger Bands (20, 2)')
        
        if 'bid_price_1' in df_p1.columns:
            ax1.fill_between(df_p1['timestamp'], df_p1['bid_price_1'], df_p1['ask_price_1'], color='#444444', alpha=0.5, label='Spread')
            
        ax1.set_title(f"Price Action & Wstęgi Bollingera: {prod_1}", fontsize=16)
        ax1.legend(loc='upper left')
        ax1.grid(color='#333333', linestyle='--')

        # Wykres zmienności (Spread & ATR proxy)
        if 'bid_price_1' in df_p1.columns:
            df_p1['Spread'] = df_p1['ask_price_1'] - df_p1['bid_price_1']
            ax2.plot(df_p1['timestamp'], df_p1['Spread'], color='#ffff00', label='Bid-Ask Spread')
            ax2.set_title("Dynamika Spreadu", fontsize=12)
            ax2.legend(loc='upper left')
            ax2.grid(color='#333333', linestyle='--')

        st.pyplot(fig1)

    # ==========================================
    # ZAKŁADKA 2: VOLUME PROFILE & CVD
    # ==========================================
    with tabs[1]:
        st.header("Mikrostruktura Wolumenu (Volume Profile & CVD)")
        if trades_df is not None:
            prod_2 = st.selectbox("Wybierz aktywo:", products, key='t2_prod')
            df_t2 = trades_df[trades_df['product'] == prod_2].copy()
            
            # Wstępne przypisanie kierunku transakcji (Tick Rule) dla CVD
            df_t2['price_change'] = df_t2['price'].diff()
            df_t2['trade_dir'] = np.where(df_t2['price_change'] > 0, 1, np.where(df_t2['price_change'] < 0, -1, 0))
            # Wypełniamy zera poprzednim kierunkiem
            df_t2['trade_dir'] = df_t2['trade_dir'].replace(0, method='ffill').fillna(1) 
            df_t2['signed_vol'] = df_t2['quantity'] * df_t2['trade_dir']
            df_t2['CVD'] = df_t2['signed_vol'].cumsum()

            col_vp1, col_vp2 = st.columns([1, 2])
            
            with col_vp1:
                # Volume Profile
                st.subheader("Volume Profile (VP)")
                st.write("Wykrywanie stref wsparcia/oporu.")
                vp_data = df_t2.groupby('price')['quantity'].sum().reset_index()
                fig2a, ax2a = plt.subplots(figsize=(6, 8))
                ax2a.barh(vp_data['price'], vp_data['quantity'], color=COLOR_NEUTRAL, alpha=0.7)
                
                # Znajdź POC (Point of Control)
                poc_price = vp_data.loc[vp_data['quantity'].idxmax(), 'price']
                ax2a.axhline(poc_price, color='red', linestyle='-', linewidth=2, label=f'POC: {poc_price}')
                ax2a.set_xlabel("Całkowity Wolumen")
                ax2a.set_ylabel("Cena Transakcji")
                ax2a.legend()
                ax2a.grid(color='#333333', linestyle='--')
                st.pyplot(fig2a)

            with col_vp2:
                # Cumulative Volume Delta (CVD)
                st.subheader("Cumulative Volume Delta (CVD)")
                st.write("Mierzy siłę agresywnych kupujących vs sprzedających w czasie.")
                fig2b, ax2b1 = plt.subplots(figsize=(10, 8))
                ax2b2 = ax2b1.twinx()
                
                df_p2 = prices_df[prices_df['product'] == prod_2]
                ax2b1.plot(df_p2['timestamp'], df_p2['mid_price'], color='white', label='Mid Price', alpha=0.7)
                ax2b2.plot(df_t2['timestamp'], df_t2['CVD'], color='#ff9900', label='CVD', linewidth=2)
                
                ax2b1.set_ylabel("Cena")
                ax2b2.set_ylabel("Skumulowana Delta Wolumenu")
                
                # Poprawka dla legendy
                lines_1, labels_1 = ax2b1.get_legend_handles_labels()
                lines_2, labels_2 = ax2b2.get_legend_handles_labels()
                ax2b1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
                ax2b1.grid(color='#333333', linestyle='--')
                st.pyplot(fig2b)
        else:
            st.warning("Ta sekcja wymaga wgrania pliku trades.csv")

    # ==========================================
    # ZAKŁADKA 3: STAT-ARB & KORELACJE
    # ==========================================
    with tabs[2]:
        st.header("Arbitraż Statystyczny (Pairs Trading / Baskets)")
        pivot = prices_df.pivot(index='timestamp', columns='product', values='mid_price').ffill()
        returns = pivot.pct_change().dropna()
        
        st.subheader("Matryca Korelacji Stóp Zwrotu")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', ax=ax3, linewidths=.5)
        st.pyplot(fig3)
        
        st.markdown("---")
        st.subheader("Analiza Spreadu (Z-Score)")
        c_sa1, c_sa2, c_sa3 = st.columns(3)
        prod_sa_a = c_sa1.selectbox("Aktywo A (LONG)", products, index=0)
        prod_sa_b = c_sa2.selectbox("Aktywo B (SHORT)", products, index=min(1, len(products)-1))
        z_win = c_sa3.slider("Okno średniej kroczącej Z-Score", 10, 1000, 100)
        
        if prod_sa_a != prod_sa_b:
            spread = pivot[prod_sa_a] - pivot[prod_sa_b]
            mean_spread = spread.rolling(window=z_win).mean()
            std_spread = spread.rolling(window=z_win).std()
            z_score = (spread - mean_spread) / std_spread
            
            fig_sa, (ax_sa1, ax_sa2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
            ax_sa1.plot(spread, color=COLOR_NEUTRAL, label=f'Spread Rzeczywisty ({prod_sa_a} - {prod_sa_b})')
            ax_sa1.plot(mean_spread, color='yellow', linestyle='--', label=f'Średnia Spreadu ({z_win})')
            ax_sa1.legend(); ax_sa1.set_title("Historia Spreadu Ceny")
            ax_sa1.grid(color='#333333', linestyle='--')
            
            ax_sa2.plot(z_score, color='#ff00ff', label='Z-Score Spreadu')
            ax_sa2.axhline(2, color=COLOR_DOWN, linestyle='--', label='+2 (Sygnał Sell A / Buy B)')
            ax_sa2.axhline(-2, color=COLOR_UP, linestyle='--', label='-2 (Sygnał Buy A / Sell B)')
            ax_sa2.axhline(0, color='gray', linestyle='-')
            ax_sa2.legend(); ax_sa2.set_title("Z-Score (Sygnały Mean-Reversion)")
            ax_sa2.grid(color='#333333', linestyle='--')
            st.pyplot(fig_sa)

    # ==========================================
    # ZAKŁADKA 4: LEAD-LAG ANALYSIS
    # ==========================================
    with tabs[3]:
        st.header("Analiza Wyprzedzenia (Lead-Lag Relationship)")
        st.write("Sprawdź, czy zmiana ceny jednego produktu systematycznie wyprzedza (o $N$ timestampów) ruchy innego.")
        
        c_ll1, c_ll2, c_ll3 = st.columns(3)
        prod_lead = c_ll1.selectbox("Potencjalny Lider (Lead)", products, index=0, key='ll1')
        prod_lag = c_ll2.selectbox("Potencjalny Naśladowca (Lag)", products, index=min(1, len(products)-1), key='ll2')
        max_lag = c_ll3.slider("Maksymalne przesunięcie (Kroki)", 1, 50, 20)

        if prod_lead != prod_lag:
            s_lead = returns[prod_lead]
            s_lag = returns[prod_lag]
            
            corrs = []
            lags_range = range(-max_lag, max_lag + 1)
            for l in lags_range:
                # Jeśli l jest dodatnie, przesuwamy 'Lead' do przodu (sprawdzamy, czy Lead_t tłumaczy Lag_{t+l})
                corrs.append(s_lag.corr(s_lead.shift(l)))
                
            fig_ll, ax_ll = plt.subplots(figsize=(12, 6))
            ax_ll.bar(lags_range, corrs, color=np.where(np.array(lags_range)<0, '#aaaaaa', COLOR_NEUTRAL))
            ax_ll.axvline(0, color='red', linestyle='--')
            ax_ll.set_title(f"Cross-Korelacja: {prod_lead} vs {prod_lag}")
            ax_ll.set_xlabel("Przesunięcie w czasie (Lags)")
            ax_ll.set_ylabel("Współczynnik Korelacji")
            st.write("💡 *Najwyższy słupek po PRAWEJ stronie od zera oznacza, że Aktywo 1 skutecznie wyprzedza Aktywo 2.*")
            st.pyplot(fig_ll)

    # ==========================================
    # ZAKŁADKA 5: ORDER BOOK IMBALANCE (OBI)
    # ==========================================
    with tabs[4]:
        st.header("Order Book Imbalance (OBI) i Predictive Power")
        prod_obi = st.selectbox("Wybierz aktywo:", products, key='t5_prod')
        df_obi = prices_df[prices_df['product'] == prod_obi].copy()
        
        if 'bid_volume_1' in df_obi.columns and 'ask_volume_1' in df_obi.columns:
            df_obi['OBI'] = (df_obi['bid_volume_1'] - df_obi['ask_volume_1']) / (df_obi['bid_volume_1'] + df_obi['ask_volume_1'])
            obi_ma_win = st.slider("Wygładzanie OBI", 1, 50, 10)
            df_obi['OBI_MA'] = df_obi['OBI'].rolling(window=obi_ma_win).mean()
            
            fig_obi1, ax_o1 = plt.subplots(figsize=(16, 5))
            ax_o2 = ax_o1.twinx()
            ax_o1.plot(df_obi['timestamp'], df_obi['mid_price'], color='white', label='Mid Price')
            ax_o2.plot(df_obi['timestamp'], df_obi['OBI_MA'], color='#ff00ff', label=f'OBI MA({obi_ma_win})', alpha=0.7)
            ax_o2.axhline(0, color='gray', linestyle='--')
            ax_o1.legend(loc='upper left'); ax_o2.legend(loc='upper right')
            st.pyplot(fig_obi1)
            
            # Scatter Plot: OBI vs Future Returns
            st.subheader("Czy wysokie OBI przewiduje wzrost ceny?")
            fut_steps = st.slider("Horyzont predykcji (kroki w przód)", 1, 50, 10)
            df_obi['Future_Return'] = df_obi['mid_price'].shift(-fut_steps) - df_obi['mid_price']
            
            fig_obi2, ax_o3 = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='OBI', y='Future_Return', data=df_obi.dropna(), alpha=0.2, color=COLOR_NEUTRAL, ax=ax_o3)
            ax_o3.axhline(0, color='red', linestyle='--'); ax_o3.axvline(0, color='red', linestyle='--')
            ax_o3.set_title(f"OBI vs Zmiana ceny po {fut_steps} krokach")
            # Linia regresji
            idx = df_obi[['OBI', 'Future_Return']].dropna()
            if not idx.empty:
                m, b = np.polyfit(idx['OBI'], idx['Future_Return'], 1)
                ax_o3.plot(idx['OBI'], m*idx['OBI'] + b, color='yellow', linewidth=2, label=f'Trend (Nachylenie: {m:.4f})')
                ax_o3.legend()
            st.pyplot(fig_obi2)
        else:
            st.warning("Brak danych o wolumenie L1 w pliku.")

    # ==========================================
    # ZAKŁADKA 6: TAPE READING (INSIDERS)
    # ==========================================
    with tabs[5]:
        st.header("Identyfikacja Poinformowanych Botów (Tape Reading)")
        if trades_df is not None and 'buyer' in trades_df.columns:
            unique_ids = trades_df['buyer'].dropna().unique()
            if len(unique_ids) > 1 and unique_ids[0] != '?':
                prod_tr = st.selectbox("Produkt do śledzenia:", products, key='t6_prod')
                horiz = st.slider("Horyzont weryfikacji (Kroki)", 1, 50, 10)
                
                price_map = prices_df[prices_df['product'] == prod_tr].set_index('timestamp')['mid_price']
                df_tr = trades_df[trades_df['product'] == prod_tr].copy()
                df_tr['future_price'] = df_tr['timestamp'].map(lambda x: price_map.get(x + horiz * 100))
                df_tr['profit'] = df_tr['future_price'] - df_tr['price'] # W muszelkach
                
                col_tr1, col_tr2 = st.columns(2)
                with col_tr1:
                    st.subheader("Najlepsi Kupujący (LONG)")
                    st.write(f"Ile zarobili po {horiz} krokach?")
                    b_perf = df_tr.groupby('buyer').agg({'profit':'mean', 'price':'count'}).rename(columns={'profit':'Avg Profit (Shells)', 'price':'Trades'})
                    st.dataframe(b_perf.query('Trades > 5').sort_values(by='Avg Profit (Shells)', ascending=False).style.background_gradient(cmap='Greens'))
                with col_tr2:
                    st.subheader("Najlepsi Sprzedający (SHORT)")
                    st.write(f"Ile oszczędzili (spadek ceny) po {horiz} krokach?")
                    s_perf = df_tr.groupby('seller').agg({'profit':'mean', 'price':'count'}).rename(columns={'profit':'Avg Drop (Shells)', 'price':'Trades'})
                    st.dataframe(s_perf.query('Trades > 5').sort_values(by='Avg Drop (Shells)', ascending=True).style.background_gradient(cmap='Reds_r'))
            else:
                st.info("Zarządzanie uczestnikami anonimowe w tej rundzie.")
        else: st.warning("Wymagany plik Trades CSV z kolumnami buyer/seller.")

    # ==========================================
    # ZAKŁADKA 7: BOT FINGERPRINT (BLIND)
    # ==========================================
    with tabs[6]:
        st.header("Algorithmic Fingerprinting (Blind Detection)")
        st.write("Wykrywanie mechanicznych zachowań w anonimowym Order Flow.")
        if trades_df is not None:
            prod_fp = st.selectbox("Produkt:", products, key='t7_prod')
            df_fp = trades_df[trades_df['product'] == prod_fp]
            
            c_fp1, c_fp2 = st.columns(2)
            with c_fp1:
                st.subheader("Rozkład Rozmiarów Zleceń")
                vols = df_fp['quantity'].value_counts().head(10).reset_index()
                vols.columns = ['Rozmiar Zlecenia', 'Liczba Wystąpień']
                st.dataframe(vols, hide_index=True)
                
            with c_fp2:
                top_vol = st.selectbox("Wybierz stały wolumen do analizy częstości:", vols['Rozmiar Zlecenia'])
                df_spec = df_fp[df_fp['quantity'] == top_vol].copy()
                df_spec['time_diff'] = df_spec['timestamp'].diff()
                fig_fp, ax_fp = plt.subplots(figsize=(6, 4))
                sns.histplot(df_spec['time_diff'].dropna(), bins=30, color='orange', ax=ax_fp)
                ax_fp.set_title(f"Rozkład odstępów czasu dla zleceń o rozmiarze {top_vol}")
                st.pyplot(fig_fp)
        else: st.warning("Wymagany plik Trades CSV.")

    # ==========================================
    # ZAKŁADKA 8: SYGNAŁY ZEWNĘTRZNE
    # ==========================================
    with tabs[7]:
        st.header("Analiza Danych Egzogenicznych (Obserwacje)")
        std_cols = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
        obs_cols = [c for c in prices_df.columns if c not in std_cols and pd.api.types.is_numeric_dtype(prices_df[c])]
        
        if obs_cols:
            prod_obs = st.selectbox("Produkt:", products, key='t8_prod')
            sig_col = st.selectbox("Obserwacja (Sygnał):", obs_cols)
            df_obs = prices_df[prices_df['product'] == prod_obs].copy()
            
            fig_obs, ax_obs1 = plt.subplots(figsize=(16, 5))
            ax_obs2 = ax_obs1.twinx()
            ax_obs1.plot(df_obs['timestamp'], df_obs['mid_price'], color='white', label='Cena')
            ax_obs2.plot(df_obs['timestamp'], df_obs[sig_col], color='cyan', alpha=0.7, label=sig_col)
            ax_obs1.set_ylabel("Cena"); ax_obs2.set_ylabel("Wartość Obserwacji")
            ax_obs1.legend(loc='upper left'); ax_obs2.legend(loc='upper right')
            st.pyplot(fig_obs)
            
            diff_corr = df_obs['mid_price'].diff().corr(df_obs[sig_col].diff())
            st.metric("Korelacja Pochodnych (Ruch vs Ruch)", f"{diff_corr:.4f}")
        else:
            st.info("Brak dodatkowych sygnałów w pliku (np. DOLPHIN_SIGHTINGS, SUNLIGHT).")

    # ==========================================
    # ZAKŁADKA 9: OPCJE (BLACK-SCHOLES)
    # ==========================================
    with tabs[8]:
        st.header("Wycena Opcji / Voucherów (Black-Scholes)")
        st.write("Wykorzystaj matematykę do znalezienia arbitrażu na derywatach.")
        c_bs1, c_bs2, c_bs3 = st.columns(3)
        c_bs4, c_bs5, c_bs6 = st.columns(3)
        
        S = c_bs1.number_input("Cena Bazowa (Underlying - S)", value=10000.0)
        K = c_bs2.number_input("Cena Wykonania (Strike - K)", value=10000.0)
        T = c_bs3.number_input("Czas (T - ułamek roku/okresu)", value=1.0)
        r = c_bs4.number_input("Stopa wolna od ryzyka (r)", value=0.0)
        sigma = c_bs5.number_input("Zmienność Implikowana (IV - σ)", value=0.16)
        
        if T > 0 and sigma > 0:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            call_val = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
            put_val = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
            
            c_bs6.metric("Teoretyczne CALL", f"{call_val:.2f}")
            c_bs6.metric("Teoretyczne PUT", f"{put_val:.2f}")
            
            # Generowanie uśmiechu zmienności (Volatility Smile)
            st.subheader("Symulacja cen dla różnych wartości IV")
            ivs = np.linspace(0.05, 0.50, 50)
            calls = [S*stats.norm.cdf((np.log(S/K) + (r + 0.5*v**2)*T) / (v*np.sqrt(T))) - K*np.exp(-r*T)*stats.norm.cdf(((np.log(S/K) + (r + 0.5*v**2)*T) / (v*np.sqrt(T))) - v*np.sqrt(T)) for v in ivs]
            
            fig_bs, ax_bs = plt.subplots(figsize=(8, 4))
            ax_bs.plot(ivs, calls, color='#00ffcc')
            ax_bs.axvline(sigma, color='red', linestyle='--', label='Twoje IV')
            ax_bs.set_xlabel("Implied Volatility (IV)"); ax_bs.set_ylabel("Cena Opcji Call")
            ax_bs.grid(color='#333333', linestyle='--'); ax_bs.legend()
            st.pyplot(fig_bs)

    # ==========================================
    # ZAKŁADKA 10: DEEP L2/L3 VWAP
    # ==========================================
    with tabs[9]:
        st.header("Analiza Głębokości L2/L3 & True VWAP")
        prod_l2 = st.selectbox("Produkt:", products, key='t10_prod')
        df_l2 = prices_df[prices_df['product'] == prod_l2].copy()
        
        if 'bid_price_2' in df_l2.columns:
            b_vol_tot = df_l2['bid_volume_1'].fillna(0) + df_l2['bid_volume_2'].fillna(0) + df_l2.get('bid_volume_3', pd.Series(0, index=df_l2.index)).fillna(0)
            a_vol_tot = df_l2['ask_volume_1'].fillna(0) + df_l2['ask_volume_2'].fillna(0) + df_l2.get('ask_volume_3', pd.Series(0, index=df_l2.index)).fillna(0)
            
            b_vwap = (df_l2['bid_price_1']*df_l2['bid_volume_1'].fillna(0) + df_l2['bid_price_2']*df_l2['bid_volume_2'].fillna(0) + df_l2.get('bid_price_3', 0)*df_l2.get('bid_volume_3', 0).fillna(0)) / np.where(b_vol_tot==0, 1, b_vol_tot)
            a_vwap = (df_l2['ask_price_1']*df_l2['ask_volume_1'].fillna(0) + df_l2['ask_price_2']*df_l2['ask_volume_2'].fillna(0) + df_l2.get('ask_price_3', 0)*df_l2.get('ask_volume_3', 0).fillna(0)) / np.where(a_vol_tot==0, 1, a_vol_tot)
            
            df_l2['True_VWAP'] = (b_vwap + a_vwap) / 2
            
            fig_l2, ax_l2 = plt.subplots(figsize=(16, 5))
            ax_l2.plot(df_l2['timestamp'], df_l2['mid_price'], color='gray', label='Standard Mid Price', alpha=0.5)
            ax_l2.plot(df_l2['timestamp'], df_l2['True_VWAP'], color='#ff00ff', label='L3 VWAP', linestyle='--')
            ax_l2.set_title("Mid Price vs Rzeczywisty VWAP z uwzględnieniem głębokości")
            ax_l2.legend(); ax_l2.grid(color='#333333', linestyle='--')
            st.pyplot(fig_l2)
        else: st.warning("Brak poziomów L2 w arkuszu.")

    # ==========================================
    # ZAKŁADKA 11: AUTOKORELACJA (ACF)
    # ==========================================
    with tabs[10]:
        st.header("Badanie Cykliczności i Stacjonarności (ACF/PACF)")
        prod_acf = st.selectbox("Aktywo:", products, key='t11_prod')
        ts_data = prices_df[prices_df['product'] == prod_acf]['mid_price'].dropna()
        lags = st.slider("Liczba opóźnień (Lags)", 10, 200, 40)
        
        adf_stat, p_val, _, _, _, _ = adfuller(ts_data)
        st.metric("Test Dickeya-Fullera (Stacjonarność, p-value)", f"{p_val:.4f}")
        if p_val < 0.05:
            st.success("Szereg jest stacjonarny! Idealny kandydat do strategii powrotu do średniej (Mean-Reversion).")
        else:
            st.error("Szereg niestacjonarny (Trend Random Walk). Lepsze będą strategie podążania za trendem (Momentum).")
            
        fig_acf, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(ts_data, lags=lags, ax=ax_a1, color=COLOR_NEUTRAL)
        plot_pacf(ts_data, lags=lags, ax=ax_a2, color=COLOR_NEUTRAL)
        st.pyplot(fig_acf)

    # ==========================================
    # ZAKŁADKA 12: BACKTESTER SANDBOX (NOWOŚĆ!)
    # ==========================================
    with tabs[11]:
        st.header("💰 Strategiczny Sandbox (Symulacja PnL)")
        st.write("Przetestuj prostą strategię w locie na wczytanych danych.")
        
        col_bt1, col_bt2, col_bt3 = st.columns(3)
        bt_prod = col_bt1.selectbox("Wybierz Produkt do testów:", products, key='bt_prod')
        bt_fast = col_bt2.number_input("Szybka EMA", value=10)
        bt_slow = col_bt3.number_input("Wolna EMA", value=50)
        
        df_bt = prices_df[prices_df['product'] == bt_prod].copy()
        df_bt['EMA_F'] = df_bt['mid_price'].ewm(span=bt_fast).mean()
        df_bt['EMA_S'] = df_bt['mid_price'].ewm(span=bt_slow).mean()
        
        # Logika sygnału: 1 za LONG (Fast > Slow), -1 za SHORT (Fast < Slow)
        df_bt['Signal'] = np.where(df_bt['EMA_F'] > df_bt['EMA_S'], 1, -1)
        df_bt['Position'] = df_bt['Signal'].shift(1).fillna(0) # Przesunięcie o 1 krok (nie handlujemy przyszłością)
        
        # Liczenie PnL (zakładamy handel 1 sztuką po cenie mid)
        df_bt['Market_Return'] = df_bt['mid_price'].diff()
        df_bt['Strategy_Return'] = df_bt['Position'] * df_bt['Market_Return']
        df_bt['Cumulative_PnL'] = df_bt['Strategy_Return'].cumsum()
        
        fig_bt, ax_bt1 = plt.subplots(figsize=(16, 6))
        ax_bt2 = ax_bt1.twinx()
        
        ax_bt1.plot(df_bt['timestamp'], df_bt['mid_price'], color='white', alpha=0.4, label='Cena Aktywa')
        ax_bt2.plot(df_bt['timestamp'], df_bt['Cumulative_PnL'], color=COLOR_UP, linewidth=2, label='Skumulowany Zysk (PnL)')
        ax_bt2.axhline(0, color='red', linestyle='--')
        
        ax_bt1.set_ylabel("Cena")
        ax_bt2.set_ylabel("Zysk / Strata (Muszelki)")
        ax_bt1.legend(loc='upper left'); ax_bt2.legend(loc='upper right')
        
        st.pyplot(fig_bt)
        st.metric("Całkowity Wynik Strategii (PnL na 1 jednostce obrotu)", f"{df_bt['Cumulative_PnL'].iloc[-1]:.2f} muszelek")

else:
    st.info("👋 System gotowy. Wgraj pliki CSV (Prices i opcjonalnie Trades) w lewym panelu, aby uruchomić silnik analityczny.")
