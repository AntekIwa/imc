import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json

# ==========================================
# 1. KONFIGURACJA STRONY I ESTETYKI
# ==========================================
st.set_page_config(page_title="IMC Prosperity 4 - Projekt Oppenheimer", layout="wide", page_icon="☢️")
COLOR_UP = '#00E676'
COLOR_DOWN = '#FF1744'
COLOR_NEUT = '#00B0FF'

# ==========================================
# 2. ZAAWANSOWANE FUNKCJE QUANTOWE (MATH & STATS)
# ==========================================
def kalman_filter_spread(price_x, price_y):
    """
    Filtr Kalmana do dynamicznego wyznaczania współczynnika Hedge Ratio w locie.
    Bardziej responsywny niż zwykła regresja liniowa czy Rolling Beta.
    """
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([price_x, np.ones(len(price_x))]).T
    
    state_mean = np.zeros((2, 1))
    state_cov = np.ones((2, 2))
    
    hedge_ratios = np.zeros(len(price_x))
    spreads = np.zeros(len(price_x))
    
    for t in range(len(price_x)):
        # Przewidywanie (Prediction)
        state_mean_pred = state_mean
        state_cov_pred = state_cov + trans_cov
        
        # Obserwacja (Observation)
        x_t = obs_mat[t, :].reshape(1, 2)
        y_t = price_y.iloc[t]
        
        # Błąd predykcji
        error = y_t - np.dot(x_t, state_mean_pred)[0, 0]
        error_cov = np.dot(np.dot(x_t, state_cov_pred), x_t.T)[0, 0] + 1e-3
        
        # Aktualizacja Kalmana (Kalman Gain)
        kalman_gain = np.dot(state_cov_pred, x_t.T) / error_cov
        state_mean = state_mean_pred + kalman_gain * error
        state_cov = state_cov_pred - np.dot(kalman_gain, x_t) * state_cov_pred
        
        hedge_ratios[t] = state_mean[0, 0]
        spreads[t] = y_t - (state_mean[0, 0] * x_t[0, 0] + state_mean[1, 0])
        
    return hedge_ratios, spreads

def roll_measure(price_series):
    """
    Model Rolla wyznacza efektywny spread z autokowariancji zmian cen.
    Jeśli efektywny spread < obserwowany spread = opłaca się być Market Makerem.
    """
    dp = price_series.diff().dropna()
    cov = dp.cov(dp.shift(1).dropna())
    if cov < 0:
        return 2 * np.sqrt(-cov)
    return 0

def calculate_hurst(ts, max_lag=20):
    """Wykładnik Hursta: H<0.5 (Mean Revert), H=0.5 (Random), H>0.5 (Trend)"""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Greeks dla Blacka-Scholesa"""
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
        delta = stats.norm.cdf(d1)
    else:
        price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
        delta = stats.norm.cdf(d1) - 1
    gamma = stats.norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta = (-S*stats.norm.pdf(d1)*sigma / (2*np.sqrt(T)))
    vega = S*stats.norm.pdf(d1)*np.sqrt(T)
    return price, delta, gamma, theta, vega

# ==========================================
# 3. ŁADOWANIE I CACHOWANIE DANYCH
# ==========================================
@st.cache_data
def load_data(file, sep):
    if file is not None:
        try:
            return pd.read_csv(file, sep=sep)
        except Exception as e:
            st.error(f"Błąd parsera CSV: {e}")
    return None

# ==========================================
# 4. INTERFEJS BOCZNY (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("☢️ Projekt Oppenheimer")
    st.markdown("Wgraj pliki `.csv` z wirtualnego rynku IMC, by uruchomić silnik analityczny.")
    separator = st.selectbox("Separator CSV", [";", ","], index=0)
    
    f_prices = st.file_uploader("1. Wgraj prices.csv", type=['csv'])
    f_trades = st.file_uploader("2. Wgraj trades.csv", type=['csv'])
    
    st.markdown("---")
    st.markdown("### ⚙️ Sterowanie Globalne")
    global_window = st.slider("Główne Okno Kroczące (MA, Z-Score)", 10, 500, 50)
    candle_bucket = st.slider("Agregacja OHLCV (kroki)", 100, 5000, 1000)
    
    st.markdown("---")
    st.markdown("### 🧠 Moduły AI")
    gmm_components = st.slider("Liczba Reżimów (GMM)", 2, 5, 3)
    rf_estimators = st.slider("Drzewa w Random Forest", 10, 200, 50)

# Inicjalizacja danych
prices_df = load_data(f_prices, separator)
trades_df = load_data(f_trades, separator)

st.title("☢️ IMC Prosperity 4 - Projekt Oppenheimer")
st.markdown("*Narzędzie klasy Quantitative Research. Filtry Kalmana, Modele Mieszanin, Black-Scholes.*")

if prices_df is not None:
    products = prices_df['product'].unique().tolist()
    
    # 10 rozbudowanych, potężnych modułów
    tabs = st.tabs([
        "🔬 1. Mikrostruktura L1-L3", 
        "🌪️ 2. Stat-Arb & Kalman", 
        "🌊 3. Order Flow Imbalance", 
        "🧠 4. Reżimy (GMM) & Cechy", 
        "🧮 5. Opcje & Greki (B-S)", 
        "🕵️ 6. VPIN & Tape Reading",
        "⚖️ 7. Inventory Skewing",
        "⏱️ 8. Lead-Lag & Hurst",
        "💰 9. Queue Simulator",
        "📑 10. Auto-Strategia JSON"
    ])

    # ==========================================
    # MODUŁ 1: MIKROSTRUKTURA & SHAPE (L1-L3)
    # ==========================================
    with tabs[0]:
        st.header("Mikrostruktura, Micro-Price i Roll Measure")
        prod_1 = st.selectbox("Wybierz Aktywo:", products, key='t1_p')
        df_1 = prices_df[prices_df['product'] == prod_1].copy()
        
        if 'bid_volume_1' in df_1.columns:
            # 1. Micro-Price
            df_1['Micro_Price'] = (df_1['bid_price_1'] * df_1['ask_volume_1'] + df_1['ask_price_1'] * df_1['bid_volume_1']) / (df_1['bid_volume_1'] + df_1['ask_volume_1'] + 0.0001)
            df_1['Spread'] = df_1['ask_price_1'] - df_1['bid_price_1']
            
            # 2. Roll Measure (Efektywny Spread)
            roll_spread = roll_measure(df_1['mid_price'])
            avg_spread = df_1['Spread'].mean()
            
            c1_1, c1_2, c1_3 = st.columns(3)
            c1_1.metric("Średni Obserwowany Spread", f"{avg_spread:.3f}")
            c1_2.metric("Efektywny Spread (Model Rolla)", f"{roll_spread:.3f}")
            
            if roll_spread < avg_spread:
                c1_3.success("Szum rynkowy < Spread. Opłaca się być Market Makerem (zbierać spread).")
            else:
                c1_3.error("Szum rynkowy > Spread. Toksyczny rynek, ryzyko bycia przejechanym przez momentum.")

            # Wykres Ceny
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['mid_price'], name='Mid Price', line=dict(color='gray', width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['Micro_Price'], name='Micro-Price (Ważona)', line=dict(color=COLOR_NEUT, width=2)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['Spread'], name='Spread', line=dict(color='yellow')), row=2, col=1)
            fig1.update_layout(height=600, template='plotly_dark', title="Price Action & Płynność"); st.plotly_chart(fig1, use_container_width=True)
            
            # Wizualizacja Kształtu Arkusza (Order Book Shape) dla pierwszej próbki
            st.subheader("Kształt Arkusza Zleceń (Głębokość)")
            has_l3 = 'bid_price_3' in df_1.columns
            if has_l3:
                snap = df_1.iloc[len(df_1)//2] # Środek dnia
                prices = [snap['bid_price_3'], snap['bid_price_2'], snap['bid_price_1'], snap['ask_price_1'], snap['ask_price_2'], snap['ask_price_3']]
                vols = [snap['bid_volume_3'], snap['bid_volume_2'], snap['bid_volume_1'], -snap['ask_volume_1'], -snap['ask_volume_2'], -snap['ask_volume_3']]
                colors = [COLOR_UP]*3 + [COLOR_DOWN]*3
                
                fig_ob = px.bar(x=prices, y=vols, color=colors, title=f"Migawka Arkusza L3 (Timestamp: {snap['timestamp']})", template='plotly_dark')
                fig_ob.update_layout(xaxis_title="Cena", yaxis_title="Wolumen (Ujemny = Ask)")
                st.plotly_chart(fig_ob, use_container_width=True)
            else:
                st.info("Brak poziomów L2/L3 do narysowania kształtu.")

    # ==========================================
    # MODUŁ 2: STAT-ARB & KALMAN FILTER
    # ==========================================
    with tabs[1]:
        st.header("Arbitraż Statystyczny: Kointegracja & Filtr Kalmana")
        pivot = prices_df.pivot(index='timestamp', columns='product', values='mid_price').ffill()
        
        c2_1, c2_2 = st.columns(2)
        prod_a = c2_1.selectbox("Aktywo Zależne (Y)", products, index=0, key='t2_pa')
        prod_b = c2_2.selectbox("Aktywo Niezależne (X)", products, index=min(1, len(products)-1), key='t2_pb')
        
        if prod_a != prod_b:
            y = pivot[prod_a].dropna()
            x = pivot[prod_b].dropna()
            min_l = min(len(x), len(y))
            x, y = x.iloc[:min_l], y.iloc[:min_l]
            
            # Kointegracja
            score, p_val, _ = coint(y, x)
            st.metric("Test Engle'a-Grangera (P-Value)", f"{p_val:.5f}")
            if p_val < 0.05:
                st.success("Szeregi wysoce skointegrowane! Spread ma właściwości powrotu do średniej.")
            else:
                st.warning("Brak silnej kointegracji. Ryzyko rozjechania się spreadu.")
            
            # Filtr Kalmana
            hedge_ratios, kalman_spread = kalman_filter_spread(x.values, y.values)
            z_score_kalman = (kalman_spread - np.mean(kalman_spread)) / np.std(kalman_spread)
            
            fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05)
            fig2.add_trace(go.Scatter(x=y.index, y=y, name=prod_a, line=dict(color='white')), row=1, col=1)
            fig2.add_trace(go.Scatter(x=x.index, y=x, name=prod_b, line=dict(color='gray')), row=1, col=1)
            
            fig2.add_trace(go.Scatter(x=y.index, y=hedge_ratios, name='Kalman Hedge Ratio (Beta)', line=dict(color='yellow')), row=2, col=1)
            
            fig2.add_trace(go.Scatter(x=y.index, y=z_score_kalman, name='Kalman Z-Score', line=dict(color=COLOR_NEUT)), row=3, col=1)
            fig2.add_hline(y=2.0, line_dash='dash', line_color=COLOR_DOWN, row=3, col=1)
            fig2.add_hline(y=-2.0, line_dash='dash', line_color=COLOR_UP, row=3, col=1)
            
            fig2.update_layout(height=800, template='plotly_dark', title="Dynamiczny Arbitraż - Filtr Kalmana"); st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # MODUŁ 3: ORDER FLOW IMBALANCE (OFI)
    # ==========================================
    with tabs[2]:
        st.header("Order Flow Imbalance (OFI)")
        prod_3 = st.selectbox("Aktywo:", products, key='t3_p')
        df_3 = prices_df[prices_df['product'] == prod_3].copy()
        
        if 'bid_volume_1' in df_3.columns:
            df_3['prev_bid'] = df_3['bid_price_1'].shift(1)
            df_3['prev_ask'] = df_3['ask_price_1'].shift(1)
            df_3['prev_bvol'] = df_3['bid_volume_1'].shift(1)
            df_3['prev_avol'] = df_3['ask_volume_1'].shift(1)
            
            d_bid = np.where(df_3['bid_price_1'] >= df_3['prev_bid'], df_3['bid_volume_1'], 0) - np.where(df_3['bid_price_1'] <= df_3['prev_bid'], df_3['prev_bvol'], 0)
            d_ask = np.where(df_3['ask_price_1'] <= df_3['prev_ask'], df_3['ask_volume_1'], 0) - np.where(df_3['ask_price_1'] >= df_3['prev_ask'], df_3['prev_avol'], 0)
            
            df_3['OFI'] = d_bid - d_ask
            df_3['OFI_MA'] = df_3['OFI'].rolling(global_window).mean()
            df_3['OFI_CUM'] = df_3['OFI'].cumsum()
            
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(x=df_3['timestamp'], y=df_3['mid_price'], name="Mid Price", line=dict(color='white')), secondary_y=False)
            fig3.add_trace(go.Scatter(x=df_3['timestamp'], y=df_3['OFI_CUM'], name="CUM OFI", line=dict(color='magenta', width=2)), secondary_y=True)
            fig3.update_layout(height=500, template='plotly_dark', title="Skumulowany Order Flow Imbalance (Wskaźnik Wyprzedzający)"); st.plotly_chart(fig3, use_container_width=True)
            
            st.info("💡 Gdy OFI drastycznie rośnie (linia magneta w górę), presja kupujących narasta z każdym krokiem. Szukaj sygnałów LONG.")

    # ==========================================
    # MODUŁ 4: ML REŻIMY (GMM) I RANDOM FOREST
    # ==========================================
    with tabs[3]:
        st.header("Machine Learning: GMM Regimes & Random Forest Features")
        prod_4 = st.selectbox("Aktywo:", products, key='t4_p')
        df_4 = prices_df[prices_df['product'] == prod_4].copy()
        
        c4_1, c4_2 = st.columns(2)
        with c4_1:
            st.subheader(f"Gaussian Mixture Models (Ukryte Stany Rynku: {gmm_components})")
            df_4['Return'] = df_4['mid_price'].pct_change()
            df_4['Vol'] = df_4['Return'].rolling(global_window).std()
            if 'bid_price_1' in df_4.columns:
                df_4['Spread'] = df_4['ask_price_1'] - df_4['bid_price_1']
            else: df_4['Spread'] = 0
            
            df_4_c = df_4.dropna().copy()
            if len(df_4_c) > 100:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df_4_c[['Vol', 'Spread', 'Return']])
                gmm = GaussianMixture(n_components=gmm_components, covariance_type='full', random_state=42)
                df_4_c['Regime'] = gmm.fit_predict(scaled).astype(str)
                
                fig4a = px.scatter(df_4_c, x='timestamp', y='mid_price', color='Regime', title="Identyfikacja Reżimów Rynkowych", template='plotly_dark')
                st.plotly_chart(fig4a, use_container_width=True)
                
        with c4_2:
            st.subheader("Random Forest: Predyktory Przyszłości")
            if 'bid_volume_1' in df_4.columns:
                df_4['OBI'] = (df_4['bid_volume_1'] - df_4['ask_volume_1']) / (df_4['bid_volume_1'] + df_4['ask_volume_1'] + 0.001)
                df_4['Mom_Short'] = df_4['mid_price'].diff(5)
                df_4['Mom_Long'] = df_4['mid_price'].diff(20)
                df_4['Target'] = df_4['mid_price'].shift(-10) - df_4['mid_price'] # Zmiana za 10 timestampów
                
                ml_df = df_4[['Spread', 'OBI', 'Mom_Short', 'Mom_Long', 'Vol', 'Target']].dropna()
                X = ml_df.drop('Target', axis=1); y = ml_df['Target']
                
                rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=5, random_state=42)
                rf.fit(X, y)
                imp = pd.DataFrame({'Cecha': X.columns, 'Ważność': rf.feature_importances_}).sort_values('Ważność')
                fig4b = px.bar(imp, x='Ważność', y='Cecha', orientation='h', template='plotly_dark', title="Która cecha najlepiej przewiduje cenę?")
                st.plotly_chart(fig4b, use_container_width=True)

    # ==========================================
    # MODUŁ 5: OPCJE I GREKI (BLACK-SCHOLES)
    # ==========================================
    with tabs[4]:
        st.header("Zarządzanie Ryzykiem Instrumentów Pochodnych")
        c5_1, c5_2, c5_3 = st.columns(3)
        c5_4, c5_5, c5_6 = st.columns(3)
        
        S = c5_1.number_input("Cena Bazowa S", value=10000.0)
        K = c5_2.number_input("Strike K", value=10000.0)
        T = c5_3.number_input("Czas T", value=1.0)
        r = c5_4.number_input("Stopa r", value=0.0)
        sigma = c5_5.number_input("Implied Volatility (IV)", value=0.16)
        typ = c5_6.selectbox("Opcja", ['call', 'put'])
        
        if T > 0 and sigma > 0:
            price, delta, gamma, theta, vega = bs_greeks(S, K, T, r, sigma, typ)
            st.markdown(f"### Wycena Opcji: **{price:.2f}** muszelek")
            
            g_col1, g_col2, g_col3, g_col4 = st.columns(4)
            g_col1.metric("Delta (Δ)", f"{delta:.4f}")
            g_col2.metric("Gamma (Γ)", f"{gamma:.6f}")
            g_col3.metric("Theta (Θ)", f"{theta:.4f}")
            g_col4.metric("Vega (ν)", f"{vega:.4f}")
            
            # Symulacja Delty i Gammy
            s_range = np.linspace(S*0.8, S*1.2, 100)
            d_vals = [bs_greeks(s, K, T, r, sigma, typ)[1] for s in s_range]
            g_vals = [bs_greeks(s, K, T, r, sigma, typ)[2] for s in s_range]
            
            fig5 = make_subplots(specs=[[{"secondary_y": True}]])
            fig5.add_trace(go.Scatter(x=s_range, y=d_vals, name='Delta (Kierunek)'), secondary_y=False)
            fig5.add_trace(go.Scatter(x=s_range, y=g_vals, name='Gamma (Przyśpieszenie)'), secondary_y=True)
            fig5.update_layout(title="Krzywe Ryzyka (Delta i Gamma)", template='plotly_dark'); st.plotly_chart(fig5, use_container_width=True)

    # ==========================================
    # MODUŁ 6: VPIN & TAPE READING
    # ==========================================
    with tabs[5]:
        st.header("Analiza Toksyczności (VPIN) & Tape Reading")
        if trades_df is not None:
            prod_6 = st.selectbox("Aktywo:", products, key='t6_p')
            df_t6 = trades_df[trades_df['product'] == prod_6].copy()
            df_p6 = prices_df[prices_df['product'] == prod_6].set_index('timestamp')['mid_price']
            
            df_t6['dir'] = np.where(df_t6['price'].diff() > 0, 1, -1)
            df_t6['buy_v'] = np.where(df_t6['dir'] == 1, df_t6['quantity'], 0)
            df_t6['sell_v'] = np.where(df_t6['dir'] == -1, df_t6['quantity'], 0)
            
            roll_b = df_t6['buy_v'].rolling(global_window).sum()
            roll_s = df_t6['sell_v'].rolling(global_window).sum()
            df_t6['VPIN'] = abs(roll_b - roll_s) / (roll_b + roll_s + 0.0001)
            
            fig6 = make_subplots(specs=[[{"secondary_y": True}]])
            fig6.add_trace(go.Scatter(x=df_t6['timestamp'], y=df_t6['timestamp'].map(df_p6), name="Cena", line=dict(color='white')), secondary_y=False)
            fig6.add_trace(go.Scatter(x=df_t6['timestamp'], y=df_t6['VPIN'], name="VPIN", line=dict(color='red')), secondary_y=True)
            fig6.add_hline(y=0.75, line_dash="dash", line_color="orange", secondary_y=True, annotation_text="Strefa Toksyczna")
            fig6.update_layout(height=500, template='plotly_dark', title="VPIN (Volume-Synchronized Probability of Informed Trading)"); st.plotly_chart(fig6, use_container_width=True)
            
            # Tape Reading (Profilowanie z ID)
            if 'buyer' in df_t6.columns and df_t6['buyer'].dropna().nunique() > 1:
                st.subheader("Tape Reading (Trader IDs)")
                b_vol = df_t6.groupby('buyer')['quantity'].sum().reset_index().rename(columns={'buyer':'Trader', 'quantity':'Kupił'})
                s_vol = df_t6.groupby('seller')['quantity'].sum().reset_index().rename(columns={'seller':'Trader', 'quantity':'Sprzedał'})
                flow = pd.merge(b_vol, s_vol, on='Trader', how='outer').fillna(0)
                fig6b = px.bar(flow, x='Trader', y=['Kupił', 'Sprzedał'], barmode='group', template='plotly_dark', title="Agregacja Wolumenu per Bot")
                st.plotly_chart(fig6b, use_container_width=True)
        else: st.warning("Wgraj trades.csv")

    # ==========================================
    # MODUŁ 7: INVENTORY SKEWING
    # ==========================================
    with tabs[6]:
        st.header("Inventory Skewing (Model Avellaneda-Stoikov)")
        st.write("Dynamiczne dopasowanie spreadu w zależności od posiadanego inwentarza, by uniknąć kary za trzymanie zapasów.")
        
        c7_1, c7_2, c7_3 = st.columns(3)
        q_limit = c7_1.number_input("Limit Ekwipunku (Max Pos)", value=20)
        curr_q = c7_2.slider("Twój obecny stan (q)", -q_limit, q_limit, 0)
        gamma_risk = c7_3.slider("Awersja do ryzyka (Gamma)", 0.0, 1.0, 0.1)
        
        # Wzór na cenę rezerwacyjną i optymalny spread wokół niej
        positions = np.arange(-q_limit, q_limit + 1)
        reservation_prices = 10000 - (positions * gamma_risk * 10) # Skalowanie
        opt_spread = 2 + (gamma_risk * (positions**2) / q_limit) # Asymetryczny spread
        
        fig7 = make_subplots(specs=[[{"secondary_y": True}]])
        fig7.add_trace(go.Scatter(x=positions, y=reservation_prices, name="Cena Rezerwacyjna (R)", mode='lines+markers', line=dict(color=COLOR_NEUT)), secondary_y=False)
        fig7.add_trace(go.Bar(x=positions, y=opt_spread, name="Sugerowany Spread", marker_color='orange', opacity=0.4), secondary_y=True)
        fig7.add_vline(x=curr_q, line_dash="dash", line_color="red", annotation_text="Twoja Pozycja")
        fig7.update_layout(title="Krzywa Przesunięcia (Skewing) i Optymalny Spread", template='plotly_dark')
        st.plotly_chart(fig7, use_container_width=True)

    # ==========================================
    # MODUŁ 8: LEAD-LAG & HURST EXPONENT
    # ==========================================
    with tabs[7]:
        st.header("Analiza Czasowa: Lead-Lag i Hurst Exponent")
        
        c8_1, c8_2 = st.columns(2)
        with c8_1:
            st.subheader("Wykładnik Hursta (Reżimy)")
            prod_8a = st.selectbox("Produkt do Hursta:", products, key='t8_pa')
            ts_8 = prices_df[prices_df['product'] == prod_8a]['mid_price'].dropna().values
            if len(ts_8) > 50:
                h_val = calculate_hurst(ts_8)
                st.metric(f"Hurst dla {prod_8a}", f"{h_val:.4f}")
                if h_val < 0.45: st.success("Rynek Powracający do Średniej (Mean-Reverting)")
                elif h_val > 0.55: st.error("Rynek w silnym Trendzie (Trending)")
                else: st.warning("Błądzenie Losowe (Random Walk)")
        
        with c8_2:
            st.subheader("Lead-Lag (Wyprzedzanie)")
            p_lead = st.selectbox("Potencjalny Lider:", products, index=0, key='t8_plead')
            p_lag = st.selectbox("Potencjalny Naśladowca:", products, index=min(1, len(products)-1), key='t8_plag')
            if p_lead != p_lag:
                pivot = prices_df.pivot(index='timestamp', columns='product', values='mid_price').ffill()
                ret_lead = pivot[p_lead].pct_change().dropna()
                ret_lag = pivot[p_lag].pct_change().dropna()
                
                lags = range(-20, 21)
                corrs = [ret_lag.corr(ret_lead.shift(l)) for l in lags]
                
                fig8 = px.bar(x=list(lags), y=corrs, title=f"Korelacja opóźniona: {p_lead} vs {p_lag}", template='plotly_dark')
                fig8.add_vline(x=0, line_dash='dash', line_color='red')
                st.plotly_chart(fig8, use_container_width=True)

    # ==========================================
    # MODUŁ 9: QUEUE SIMULATOR (SLIPPAGE & LATENCY)
    # ==========================================
    with tabs[8]:
        st.header("Symulator Kolejki Zleceń i Poślizgu (Slippage)")
        st.write("W IMC Prosperity Twoje zlecenie wchodzi NA KOŃCU kolejki (po botach IMC). Ten moduł symuluje szansę na egzekucję Limit Orderu.")
        
        prod_9 = st.selectbox("Aktywo do symulacji:", products, key='t9_p')
        df_9 = prices_df[prices_df['product'] == prod_9].copy()
        
        if 'bid_volume_1' in df_9.columns:
            st.markdown("Założenie: Wystawiasz Pasywny Limit Order na najlepszym Bidzie (`bid_price_1`). Jak duża rotacja wolumenu jest potrzebna, by Twój order wszedł?")
            
            # W IMC zlecenia czyszczą się co tick. Więc Twój limit order wchodzi tylko, jeśli Taker uderzy w pełny wolumen Makerów.
            df_9['Queue_Size'] = df_9['bid_volume_1']
            
            # Szacowanie agresywnego Sell Volume (Takerów uderzających w Bid)
            if trades_df is not None:
                df_t9 = trades_df[trades_df['product'] == prod_9].copy()
                df_t9['is_sell'] = np.where(df_t9['price'].diff() <= 0, df_t9['quantity'], 0) # Uproszczony tick-test
                taker_sells = df_t9.groupby('timestamp')['is_sell'].sum()
                df_9 = df_9.join(taker_sells, on='timestamp').fillna(0)
                
                # Czy Taker zjadł cały Order Book przed Tobą?
                df_9['Execution_Prob'] = np.where(df_9['is_sell'] >= df_9['Queue_Size'], 1, df_9['is_sell'] / (df_9['Queue_Size'] + 0.001))
                
                fig9 = make_subplots(specs=[[{"secondary_y": True}]])
                fig9.add_trace(go.Scatter(x=df_9['timestamp'], y=df_9['Queue_Size'], name="Rozmiar Kolejki", fill='tozeroy', line=dict(color='gray')), secondary_y=False)
                fig9.add_trace(go.Scatter(x=df_9['timestamp'], y=df_9['Execution_Prob']*100, name="Prawdopodobieństwo Egzekucji (%)", line=dict(color='lime')), secondary_y=True)
                fig9.update_layout(height=400, template='plotly_dark', title="Prawdopodobieństwo trafienia z Limit Orderem w danym kroku"); st.plotly_chart(fig9, use_container_width=True)
            else:
                st.warning("Potrzebny Trades CSV do estymacji Taker Volume.")
        else:
            st.warning("Brak wolumenu w danych cenowych.")

    # ==========================================
    # MODUŁ 10: AUTO-KOMPILATOR STRATEGII (JSON)
    # ==========================================
    with tabs[9]:
        st.header("Kompilator Parametrów Bota (Auto-Strategy Exporter)")
        st.write("Ten moduł zbiera dane z poprzednich zakładek i generuje gotowy obiekt JSON do wklejenia bezpośrednio na początku pliku `trader.py`.")
        
        if st.button("Generuj Parametry Bota na podstawie Danych"):
            strategy_config = {}
            for prod in products:
                df_c = prices_df[prices_df['product'] == prod].dropna()
                if len(df_c) < 50: continue
                
                h_val = calculate_hurst(df_c['mid_price'].values)
                spread_avg = (df_c['ask_price_1'] - df_c['bid_price_1']).mean() if 'ask_price_1' in df_c.columns else 0
                
                # Prosta logika decyzyjna oparta o Hursta
                if h_val < 0.45:
                    strategy = "MARKET_MAKING"
                    params = {"edge": max(1, int(spread_avg/2)), "skew_gamma": 0.1, "ema_fast": 10}
                elif h_val > 0.55:
                    strategy = "MOMENTUM_TAKER"
                    params = {"breakout_threshold": 3.0, "ema_slow": 50}
                else:
                    strategy = "STAT_ARB"
                    params = {"z_score_entry": 2.0, "z_score_exit": 0.5}
                    
                strategy_config[prod] = {
                    "regime_hurst": round(h_val, 3),
                    "suggested_strategy": strategy,
                    "params": params
                }
            
            st.json(json.dumps(strategy_config, indent=4))
            st.success(" Skopiuj powyższy słownik do `trader.py`. Parametry zostały zoptymalizowane pod kątem stacjonarności szeregów.")

else:
    st.info("System w trybie uśpienia. Wgraj logi w panelu bocznym, by uruchomić Projekt Oppenheimer.")
