import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="IMC Prosperity 4 - Alpha Analyzer", layout="wide", page_icon="🦍")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data(file, sep):
    if file is not None:
        return pd.read_csv(file, sep=sep)
    return None

# --- SIDEBAR: WGRYWANIE DANYCH ---
st.sidebar.header("📁 Wczytaj dane z rundy")
separator = st.sidebar.selectbox("Separator CSV", [";", ","])
prices_file = st.sidebar.file_uploader("Wgraj plik z cenami (Prices CSV)", type=['csv'])
trades_file = st.sidebar.file_uploader("Wgraj plik z transakcjami (Trades CSV)", type=['csv'])

prices_df = load_data(prices_file, separator)
trades_df = load_data(trades_file, separator)

st.title("🦍 IMC Prosperity 4 - Universal Market Analyzer (Ultimate Edition)")

if prices_df is not None:
    products = prices_df['product'].unique().tolist()
    
    # 8 Zakładek reprezentujących wszystkie wypracowane strategie
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Mikrostruktura", 
        "🔗 Stat-Arb", 
        "🤖 Bot Fingerprint", 
        "🕵️ Insider Detection",
        "🌊 Sygnały Zewn.",
        "⚖️ Order Book Imbalance",
        "📈 Opcje (Black-Scholes)",
        "🗺️ Landscape Stability"
    ])

    # ==========================================
    # ZAKŁADKA 1: MIKROSTRUKTURA & CENY
    # ==========================================
    with tab1:
        st.header("Analiza Price Action i Wartości Godziwej (Fair Value)")
        selected_product = st.selectbox("Wybierz produkt:", products, key='t1_prod')
        df_prod = prices_df[prices_df['product'] == selected_product].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_prod['timestamp'], df_prod['mid_price'], label='Mid Price', color='black')
            if 'bid_price_1' in df_prod.columns and 'ask_price_1' in df_prod.columns:
                ax1.fill_between(df_prod['timestamp'], df_prod['bid_price_1'], df_prod['ask_price_1'], 
                                color='gray', alpha=0.3, label='Bid-Ask Spread')
            ax1.set_title(f'Price Action: {selected_product}')
            ax1.legend()
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.histplot(df_prod['mid_price'], kde=True, ax=ax2, color='blue', bins=50)
            ax2.set_title('Rozkład Ceny (Szukaj kotwicy rynkowej)')
            st.pyplot(fig2)

    # ==========================================
    # ZAKŁADKA 2: STAT-ARB & KOSZYKI
    # ==========================================
    with tab2:
        st.header("Arbitraż Statystyczny i Z-Score Spreadu")
        
        st.subheader("1. Heatmapa Korelacji Stóp Zwrotu")
        pivot = prices_df.pivot(index='timestamp', columns='product', values='mid_price')
        returns = pivot.pct_change().dropna()
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.heatmap(returns.corr(), annot=True, cmap='RdYlGn', vmin=-1, vmax=1, center=0, fmt='.2f', ax=ax3)
        st.pyplot(fig3)
        
        st.subheader("2. Analiza Spreadu i Z-Score (Pary / Koszyki)")
        col3, col4, col5 = st.columns(3)
        prod_a = col3.selectbox("Aktywo A (np. Koszyk)", products, index=0)
        prod_b = col4.selectbox("Aktywo B (np. Komponent)", products, index=min(1, len(products)-1))
        z_window = col5.slider("Okno Z-Score", 10, 500, 100)
        
        if prod_a != prod_b:
            spread = pivot[prod_a] - pivot[prod_b]
            z_score = (spread - spread.rolling(z_window).mean()) / spread.rolling(z_window).std()
            
            fig4, ax4 = plt.subplots(figsize=(15, 5))
            ax4.plot(z_score, label=f'Z-Score Spreadu ({prod_a} - {prod_b})', color='purple')
            ax4.axhline(2, color='red', linestyle='--', label='+2 (Sprzedaj A, Kup B)')
            ax4.axhline(-2, color='green', linestyle='--', label='-2 (Kup A, Sprzedaj B)')
            ax4.set_title("Analiza Mean-Reversion")
            ax4.legend()
            st.pyplot(fig4)

    # ==========================================
    # ZAKŁADKA 3: BOT FINGERPRINTING
    # ==========================================
    with tab3:
        st.header("Analiza Behawioralna Algorytmów (Blind Detection)")
        if trades_df is not None:
            prod_fp = st.selectbox("Wybierz produkt do śledztwa:", products, key='t3_prod')
            df_trades = trades_df[trades_df['product'] == prod_fp].copy()
            df_prices = prices_df[prices_df['product'] == prod_fp].copy()
            
            st.subheader("1. Mapa 'Odcisków Palców' (Klastrowanie Wolumenów)")
            behavior = df_trades.groupby(['price', 'quantity']).size().reset_index(name='frequency')
            behavior = behavior[behavior['frequency'] > behavior['frequency'].median()]
            
            fig5, ax5 = plt.subplots(figsize=(15, 5))
            scatter = ax5.scatter(behavior['price'], behavior['quantity'], 
                                c=behavior['frequency'], s=behavior['frequency']*10, 
                                cmap='Reds', alpha=0.6, edgecolors='black')
            plt.colorbar(scatter, label='Częstotliwość występowania')
            ax5.set_title("Gdzie boty stawiają 'ściany' z wolumenem?")
            ax5.set_xlabel('Cena')
            ax5.set_ylabel('Wolumen (Quantity)')
            st.pyplot(fig5)
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("2. Boty-Zegarynki (Interwały Czasowe)")
                if not df_trades.empty:
                    top_vols = df_trades['quantity'].value_counts().head(3).index.tolist()
                    if top_vols:
                        target_vol = st.selectbox("Wybierz powtarzalny wolumen:", top_vols)
                        df_vol = df_trades[df_trades['quantity'] == target_vol].copy()
                        df_vol['time_diff'] = df_vol['timestamp'].diff()
                        
                        fig6, ax6 = plt.subplots(figsize=(8, 4))
                        sns.histplot(df_vol['time_diff'].dropna(), bins=30, color='orange', ax=ax6)
                        ax6.set_title(f'Częstotliwość dla wolumenu = {target_vol}')
                        st.pyplot(fig6)
                
            with col_t2:
                st.subheader("3. Detekcja 'Price Shadowing'")
                df_prices['prev_mid_diff'] = df_prices['mid_price'].diff().shift(1)
                if 'bid_price_1' in df_prices.columns:
                    df_prices['current_bid_diff'] = df_prices['bid_price_1'].diff()
                    corr_shadow = df_prices['prev_mid_diff'].corr(df_prices['current_bid_diff'])
                    st.metric("Korelacja Opóźniona", f"{corr_shadow:.4f}")
                    if corr_shadow > 0.6:
                        st.error("🚨 Wykryto zachowanie typu Shadowing! Bot koryguje Bid do Mid-Price.")
                    else:
                        st.success("Brak wyraźnego Shadowingu.")
        else:
            st.warning("Wgraj plik z transakcjami (Trades CSV) w panelu bocznym.")

    # ==========================================
    # ZAKŁADKA 4: INSIDER DETECTION
    # ==========================================
    with tab4:
        st.header("Wykrywanie Poinformowanych Botów (np. 'Olivia')")
        if trades_df is not None:
            if 'buyer' in trades_df.columns and 'seller' in trades_df.columns:
                unique_buyers = trades_df['buyer'].dropna().unique()
                if len(unique_buyers) > 1 and not (len(unique_buyers) == 1 and unique_buyers[0] == '?'):
                    st.success("Wykryto tożsamość botów! Rozpoczynam mapowanie...")
                    
                    prod_insider = st.selectbox("Produkt:", products, key='t4_prod')
                    future_window = st.slider("Przesunięcie czasowe (kroki w przód)", 1, 20, 5)
                    
                    price_map = prices_df[prices_df['product'] == prod_insider].set_index('timestamp')['mid_price']
                    df_t = trades_df[trades_df['product'] == prod_insider].copy()
                    
                    df_t['future_price'] = df_t['timestamp'].map(lambda x: price_map.get(x + future_window * 100))
                    df_t['profit_potential'] = (df_t['future_price'] - df_t['price']) / df_t['price']
                    
                    col_b, col_s = st.columns(2)
                    with col_b:
                        st.subheader("Najlepsi Kupujący (LONG)")
                        buyer_perf = df_t.groupby('buyer').agg({'profit_potential': 'mean', 'price': 'count'})
                        buyer_perf = buyer_perf[buyer_perf['price'] > 5].sort_values(by='profit_potential', ascending=False)
                        st.dataframe(buyer_perf)
                        
                    with col_s:
                        st.subheader("Najlepsi Sprzedający (SHORT)")
                        seller_perf = df_t.groupby('seller').agg({'profit_potential': 'mean', 'price': 'count'})
                        seller_perf = seller_perf[seller_perf['price'] > 5].sort_values(by='profit_potential', ascending=True)
                        st.dataframe(seller_perf)
                else:
                    st.info("Dane są anonimowe. Analiza Tape Reading jest niedostępna (Rundy 1-4).")
            else:
                st.info("Brak kolumn 'buyer' i 'seller'.")
        else:
            st.warning("Wgraj plik z transakcjami (Trades CSV).")

    # ==========================================
    # ZAKŁADKA 5: SYGNAŁY ZEWNĘTRZNE
    # ==========================================
    with tab5:
        st.header("Analiza Obserwacji Zewnętrznych i Sezonowości")
        standard_cols = ['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 
                         'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 
                         'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss']
        potential_signals = [col for col in prices_df.columns if col not in standard_cols and pd.api.types.is_numeric_dtype(prices_df[col])]
        
        if potential_signals:
            col_s1, col_s2 = st.columns(2)
            prod_sig = col_s1.selectbox("Wybierz produkt:", products, key='t5_prod')
            signal_col = col_s2.selectbox("Wybierz sygnał do zbadania:", potential_signals)
            
            df_sig = prices_df[prices_df['product'] == prod_sig].copy()
            
            fig7, ax7 = plt.subplots(figsize=(15, 6))
            ax8 = ax7.twinx()
            
            ax7.plot(df_sig['timestamp'], df_sig['mid_price'], color='blue', label='Cena')
            ax8.plot(df_sig['timestamp'], df_sig[signal_col], color='orange', label=f'Sygnał: {signal_col}', alpha=0.6)
            
            ax7.set_ylabel('Mid Price', color='blue')
            ax8.set_ylabel('Wartość Sygnału', color='orange')
            ax7.set_title(f"Korelacja Ceny vs {signal_col}")
            
            lines_1, labels_1 = ax7.get_legend_handles_labels()
            lines_2, labels_2 = ax8.get_legend_handles_labels()
            ax7.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            st.pyplot(fig7)
            
            corr_diff = df_sig['mid_price'].diff().corr(df_sig[signal_col].diff())
            st.metric("Korelacja Pochodnej (Zmiana Ceny vs Zmiana Sygnału)", f"{corr_diff:.4f}")
        else:
            st.info("Brak dodatkowych kolumn numerycznych (sygnałów) w tym pliku cenowym.")

    # ==========================================
    # ZAKŁADKA 6: ORDER BOOK IMBALANCE
    # ==========================================
    with tab6:
        st.header("Order Book Imbalance (Presja Arkusza Zleceń)")
        prod_obi = st.selectbox("Wybierz produkt:", products, key='t6_prod')
        df_obi = prices_df[prices_df['product'] == prod_obi].copy()
        
        if 'bid_volume_1' in df_obi.columns and 'ask_volume_1' in df_obi.columns:
            df_obi['OBI'] = (df_obi['bid_volume_1'] - df_obi['ask_volume_1']) / (df_obi['bid_volume_1'] + df_obi['ask_volume_1'])
            window_obi = st.slider("Okno wygładzania OBI", 1, 50, 10)
            df_obi['OBI_MA'] = df_obi['OBI'].rolling(window=window_obi).mean()
            
            fig9, ax9 = plt.subplots(figsize=(15, 6))
            ax10 = ax9.twinx()
            
            ax9.plot(df_obi['timestamp'], df_obi['mid_price'], color='black', label='Mid Price', alpha=0.7)
            ax10.plot(df_obi['timestamp'], df_obi['OBI_MA'], color='purple', label=f'OBI ({window_obi} MA)')
            ax10.axhline(0, color='red', linestyle='--', alpha=0.5)
            
            ax9.set_ylabel('Cena')
            ax10.set_ylabel('Presja OBI (-1 do 1)')
            ax9.set_title(f"Zależność Ceny od Presji Arkusza: {prod_obi}")
            st.pyplot(fig9)
        else:
            st.warning("Brak danych o wolumenie w arkuszu.")

    # ==========================================
    # ZAKŁADKA 7: OPCJE I BLACK-SCHOLES
    # ==========================================
    with tab7:
        st.header("Wycena Opcji (Model Blacka-Scholesa)")
        
        col_bs1, col_bs2, col_bs3 = st.columns(3)
        S = col_bs1.number_input("Cena Instrumentu Bazowego (S)", value=10000.0)
        K = col_bs2.number_input("Cena Wykonania Opcji (Strike - K)", value=10000.0)
        T = col_bs3.number_input("Czas do wygaśnięcia (T)", value=1.0)
        
        col_bs4, col_bs5 = st.columns(2)
        r = col_bs4.number_input("Stopa wolna od ryzyka (r)", value=0.0)
        sigma = col_bs5.number_input("Zmienność Implikowana (IV - sigma)", value=0.16)
        
        if T > 0 and sigma > 0:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
            st.success(f"Wartość Godziwa Opcji CALL: **{call_price:.2f} muszelek**")
            st.info(f"Wartość Godziwa Opcji PUT: **{put_price:.2f} muszelek**")

    # ==========================================
    # ZAKŁADKA 8: LANDSCAPE STABILITY
    # ==========================================
    with tab8:
        st.header("Landscape Stability (Optymalizacja Hiperparametrów)")
        st.write("Wizualizacja testów parametrów (np. Z-Score). Szukaj szerokich regionów, unikaj szpilek!")
        
        if st.button("Wygeneruj przykładowy krajobraz (Demo)"):
            entry_thresholds = np.linspace(1.0, 3.0, 10)
            exit_thresholds = np.linspace(0.0, 1.0, 10)
            
            pnl_matrix = np.zeros((10, 10))
            for i, ent in enumerate(entry_thresholds):
                for j, ext in enumerate(exit_thresholds):
                    pnl_matrix[i, j] = 100000 * np.exp(-((ent-2.0)**2 + (ext-0.5)**2)*5) + np.random.normal(0, 5000)
                    
            pnl_df = pd.DataFrame(pnl_matrix, index=np.round(entry_thresholds, 2), columns=np.round(exit_thresholds, 2))
            
            fig10, ax11 = plt.subplots(figsize=(10, 8))
            sns.heatmap(pnl_df, cmap='viridis', ax=ax11, annot=False)
            ax11.set_ylabel("Próg Wejścia (Entry Z-Score)")
            ax11.set_xlabel("Próg Wyjścia (Exit Z-Score)")
            ax11.set_title("Krajobraz PnL")
            st.pyplot(fig10)

else:
    st.info("👋 Witaj w środowisku analitycznym IMC Prosperity. Wgraj pliki CSV po lewej stronie, aby rozpocząć.")
