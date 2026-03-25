import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURACJA STRONY ---
st.set_page_config(page_title="IMC Prosperity 4 - Lewiatan", layout="wide", page_icon="🐋")
COLOR_UP = '#00E676'
COLOR_DOWN = '#FF1744'
COLOR_NEUT = '#00B0FF'

# --- 2. FUNKCJE MATEMATYCZNE I QUANTOWE ---
def calculate_hurst(ts, max_lag=20):
    """Oblicza wykładnik Hursta. H<0.5 (Mean Reverting), H=0.5 (Random Walk), H>0.5 (Trend)."""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def markov_chain_matrix(price_series):
    """Tworzy macierz prawdopodobieństw przejść dla ruchów ceny (Up, Down, Flat)."""
    diffs = np.sign(price_series.diff().dropna())
    states = diffs.map({1.0: 'Up', -1.0: 'Down', 0.0: 'Flat'})
    transitions = pd.crosstab(states.shift(), states, normalize='index')
    return transitions

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Oblicza pełny zestaw Greków dla modelu Blacka-Scholesa."""
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
        delta = stats.norm.cdf(d1)
        theta = (-S*stats.norm.pdf(d1)*sigma / (2*np.sqrt(T))) - r*K*np.exp(-r*T)*stats.norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
        delta = stats.norm.cdf(d1) - 1
        theta = (-S*stats.norm.pdf(d1)*sigma / (2*np.sqrt(T))) + r*K*np.exp(-r*T)*stats.norm.cdf(-d2)
        
    gamma = stats.norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*stats.norm.pdf(d1)*np.sqrt(T)
    return price, delta, gamma, theta, vega

@st.cache_data
def load_data(file, sep):
    if file is not None:
        try:
            return pd.read_csv(file, sep=sep)
        except Exception as e:
            st.error(f"Błąd: {e}")
    return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🐋 Terminal Lewiatan")
    sep = st.selectbox("Separator", [";", ","])
    f_prices = st.file_uploader("1. Wgraj prices.csv", type=['csv'])
    f_trades = st.file_uploader("2. Wgraj trades.csv", type=['csv'])
    
    st.markdown("---")
    st.markdown("### ⚙️ Hiperparametry")
    chart_bucket = st.slider("Agregacja Świec (Kroki)", 100, 5000, 500)
    roll_win = st.slider("Okno Kroczące (MA/Z-Score)", 10, 500, 50)

prices_df = load_data(f_prices, sep)
trades_df = load_data(f_trades, sep)

st.title("🐋 Projekt Lewiatan - Advanced Quant Environment")

if prices_df is not None:
    products = prices_df['product'].unique().tolist()
    
    tabs = st.tabs([
        "🔬 Mikrostruktura & Micro-Price", 
        "🎲 Łańcuchy Markowa & Hurst", 
        "🔗 Kointegracja (Stat-Arb)", 
        "🌊 Order Flow Imbalance (OFI)", 
        "🧠 ML: Reżimy & Cechy", 
        "🧮 Greki Opcji (B-S)", 
        "🕵️ Tape Reading (VPIN)",
        "💰 Backtester & Kelly"
    ])

    # ==========================================
    # ZAKŁADKA 1: MIKROSTRUKTURA & MICRO-PRICE
    # ==========================================
    with tabs[0]:
        st.header("Mikrostruktura i Micro-Price")
        st.write("Ważona cena mikrostrukturalna szybciej reaguje na ukrytą presję w arkuszu niż standardowy Mid-Price.")
        
        prod_1 = st.selectbox("Aktywo:", products, key='t1_p')
        df_1 = prices_df[prices_df['product'] == prod_1].copy()
        
        if 'bid_volume_1' in df_1.columns and 'ask_volume_1' in df_1.columns:
            # Kalkulacja Micro-Price (Ważona płynnością przeciwną)
            # Wzór: P_micro = (Bid * AskVol + Ask * BidVol) / (BidVol + AskVol)
            df_1['Micro_Price'] = (df_1['bid_price_1'] * df_1['ask_volume_1'] + df_1['ask_price_1'] * df_1['bid_volume_1']) / (df_1['bid_volume_1'] + df_1['ask_volume_1'])
            df_1['Spread'] = df_1['ask_price_1'] - df_1['bid_price_1']
            
            fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['mid_price'], name='Mid Price', line=dict(color='gray', width=1)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['Micro_Price'], name='Micro-Price', line=dict(color=COLOR_NEUT, width=2)), row=1, col=1)
            fig1.add_trace(go.Scatter(x=df_1['timestamp'], y=df_1['Spread'], name='Spread', line=dict(color='yellow')), row=2, col=1)
            
            fig1.update_layout(height=600, template='plotly_dark', title="Micro-Price vs Mid-Price z Dynamiką Spreadu")
            st.plotly_chart(fig1, use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Średni Spread", f"{df_1['Spread'].mean():.2f}")
            c2.metric("Maksymalny Spread (Flash Crash)", f"{df_1['Spread'].max():.2f}")
        else:
            st.warning("Brak danych poziomu 1 arkusza z wolumenem.")

    # ==========================================
    # ZAKŁADKA 2: ŁAŃCUCHY MARKOWA I HURST EXPONENT
    # ==========================================
    with tabs[1]:
        st.header("Prawdopodobieństwo i Pamięć Szeregu Czasowego")
        prod_2 = st.selectbox("Aktywo:", products, key='t2_p')
        ts_2 = prices_df[prices_df['product'] == prod_2]['mid_price'].dropna().values
        
        c2_1, c2_2 = st.columns(2)
        
        with c2_1:
            st.subheader("Wykładnik Hursta (Hurst Exponent)")
            st.latex(r"E\left[\frac{R(n)}{S(n)}\right] = C n^H")
            
            if len(ts_2) > 100:
                h_val = calculate_hurst(ts_2)
                st.metric("Hurst Exponent (H)", f"{h_val:.4f}")
                
                if h_val < 0.45:
                    st.success("H < 0.5: Szereg silnie powracający do średniej (Mean-Reverting). Graj Market Making i Stat-Arb!")
                elif h_val > 0.55:
                    st.error("H > 0.5: Szereg podążający za trendem (Trending). Graj Momentum, nie łap spadających noży!")
                else:
                    st.warning("H ≈ 0.5: Random Walk (Błądzenie Losowe). Rynek nieprzewidywalny.")
            else:
                st.write("Za mało danych dla Hursta.")

        with c2_2:
            st.subheader("Łańcuchy Markowa (Transition Matrix)")
            st.write("Jakie jest prawdopodobieństwo ruchu w górę, jeśli poprzedni krok też był w górę?")
            
            ts_series = pd.Series(ts_2)
            trans_matrix = markov_chain_matrix(ts_series)
            
            fig2 = go.Figure(data=go.Heatmap(
                   z=trans_matrix.values,
                   x=trans_matrix.columns,
                   y=trans_matrix.index,
                   colorscale='Viridis', text=np.round(trans_matrix.values, 2), texttemplate="%{text}"))
            fig2.update_layout(template='plotly_dark', height=400, xaxis_title="Obecny Stan", yaxis_title="Poprzedni Stan")
            st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # ZAKŁADKA 3: KOINTEGRACJA I ARBITRAŻ
    # ==========================================
    with tabs[2]:
        st.header("Test Kointegracji Engle'a-Grangera")
        st.write("Prosta korelacja to za mało. Kointegracja udowadnia, że spread między aktywami zawsze wraca do zera.")
        
        pivot = prices_df.pivot(index='timestamp', columns='product', values='mid_price').ffill()
        
        c3_1, c3_2 = st.columns(2)
        prod_a = c3_1.selectbox("Aktywo A", products, index=0, key='t3_pa')
        prod_b = c3_2.selectbox("Aktywo B", products, index=min(1, len(products)-1), key='t3_pb')
        
        if prod_a != prod_b:
            ts_a = pivot[prod_a].dropna()
            ts_b = pivot[prod_b].dropna()
            
            # Wyrównanie długości
            min_len = min(len(ts_a), len(ts_b))
            ts_a = ts_a.iloc[:min_len]; ts_b = ts_b.iloc[:min_len]
            
            score, p_value, _ = coint(ts_a, ts_b)
            
            st.metric("P-Value Kointegracji", f"{p_value:.5f}")
            if p_value < 0.05:
                st.success("✅ Szeregi są skointegrowane! Spread jest stacjonarny. Idealne warunki do Pairs Tradingu.")
            else:
                st.error("❌ Szeregi NIE SĄ skointegrowane. Spread może odjechać w nieskończoność. Arbitraż bardzo ryzykowny.")
                
            # Hedge Ratio (Zwykła regresja OLS)
            slope, intercept, r_val, p_val, std_err = stats.linregress(ts_b, ts_a)
            spread_coint = ts_a - (slope * ts_b)
            z_coint = (spread_coint - spread_coint.mean()) / spread_coint.std()
            
            fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig3.add_trace(go.Scatter(x=ts_a.index, y=ts_a, name=f'{prod_a}'), row=1, col=1)
            fig3.add_trace(go.Scatter(x=ts_b.index, y=ts_b * slope, name=f'{prod_b} * {slope:.2f}'), row=1, col=1)
            fig3.add_trace(go.Scatter(x=ts_a.index, y=z_coint, name='Z-Score Spreadu', line=dict(color='magenta')), row=2, col=1)
            fig3.add_hline(y=2.0, line_dash="dash", line_color=COLOR_DOWN, row=2, col=1)
            fig3.add_hline(y=-2.0, line_dash="dash", line_color=COLOR_UP, row=2, col=1)
            
            fig3.update_layout(height=600, template='plotly_dark', title=f"Kointegracja: Hedge Ratio = {slope:.3f}")
            st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # ZAKŁADKA 4: ORDER FLOW IMBALANCE (OFI)
    # ==========================================
    with tabs[3]:
        st.header("Order Flow Imbalance (OFI)")
        st.write("Dynamiczna zmiana arkusza zleceń w czasie. Silny predyktor krótkoterminowych ruchów cenowych.")
        
        prod_4 = st.selectbox("Aktywo:", products, key='t4_p')
        df_4 = prices_df[prices_df['product'] == prod_4].copy()
        
        if 'bid_volume_1' in df_4.columns:
            # Kalkulacja OFI (Uproszczona wersja modelu Conta)
            df_4['prev_bid'] = df_4['bid_price_1'].shift(1)
            df_4['prev_ask'] = df_4['ask_price_1'].shift(1)
            df_4['prev_bid_vol'] = df_4['bid_volume_1'].shift(1)
            df_4['prev_ask_vol'] = df_4['ask_volume_1'].shift(1)
            
            # Zmiany po stronie Bid (Popyt)
            cond_b1 = df_4['bid_price_1'] >= df_4['prev_bid']
            cond_b2 = df_4['bid_price_1'] <= df_4['prev_bid']
            delta_bid_vol = np.where(cond_b1, df_4['bid_volume_1'], 0) - np.where(cond_b2, df_4['prev_bid_vol'], 0)
            
            # Zmiany po stronie Ask (Podaż)
            cond_a1 = df_4['ask_price_1'] <= df_4['prev_ask']
            cond_a2 = df_4['ask_price_1'] >= df_4['prev_ask']
            delta_ask_vol = np.where(cond_a1, df_4['ask_volume_1'], 0) - np.where(cond_a2, df_4['prev_ask_vol'], 0)
            
            df_4['OFI'] = delta_bid_vol - delta_ask_vol
            df_4['OFI_CUM'] = df_4['OFI'].cumsum()
            
            fig4 = make_subplots(specs=[[{"secondary_y": True}]])
            fig4.add_trace(go.Scatter(x=df_4['timestamp'], y=df_4['mid_price'], name="Mid Price", line=dict(color='white')), secondary_y=False)
            fig4.add_trace(go.Scatter(x=df_4['timestamp'], y=df_4['OFI_CUM'], name="Skumulowane OFI", line=dict(color=COLOR_NEUT)), secondary_y=True)
            
            fig4.update_layout(height=500, template='plotly_dark', title="Skumulowany Order Flow Imbalance vs Cena")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Brak danych głębokości arkusza.")

    # ==========================================
    # ZAKŁADKA 5: ML REŻIMY RYNKOWE I CECHY
    # ==========================================
    with tabs[4]:
        st.header("ML: Reżimy (K-Means) & Znaczenie Cech (Random Forest)")
        prod_5 = st.selectbox("Aktywo:", products, key='t5_p')
        df_5 = prices_df[prices_df['product'] == prod_5].copy()
        
        c5_1, c5_2 = st.columns(2)
        
        with c5_1:
            st.subheader("K-Means: Stany Rynku")
            df_5['Ret'] = df_5['mid_price'].pct_change()
            df_5['Vol'] = df_5['Ret'].rolling(roll_win).std()
            df_5['Spread'] = df_5['ask_price_1'] - df_5['bid_price_1'] if 'bid_price_1' in df_5.columns else 0
            df_5_clean = df_5.dropna().copy()
            
            if len(df_5_clean) > 50:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(df_5_clean[['Vol', 'Spread']])
                kmeans = KMeans(n_clusters=ml_clusters, random_state=42, n_init=10)
                df_5_clean['Regime'] = kmeans.fit_predict(scaled).astype(str)
                
                fig5a = px.scatter(df_5_clean, x='timestamp', y='mid_price', color='Regime', template='plotly_dark')
                st.plotly_chart(fig5a, use_container_width=True)
        
        with c5_2:
            st.subheader("Random Forest: Predyktory")
            if 'bid_volume_1' in df_5.columns:
                df_5['OBI'] = (df_5['bid_volume_1'] - df_5['ask_volume_1']) / (df_5['bid_volume_1'] + df_5['ask_volume_1'] + 0.001)
                df_5['Target'] = df_5['mid_price'].shift(-10) - df_5['mid_price']
                ml_df = df_5[['Spread', 'OBI', 'Vol', 'Target']].dropna()
                
                rf = RandomForestRegressor(n_estimators=rf_trees, max_depth=5, random_state=42)
                rf.fit(ml_df.drop('Target', axis=1), ml_df['Target'])
                
                imp = pd.DataFrame({'Cecha': ['Spread', 'OBI', 'Zmienność'], 'Ważność': rf.feature_importances_})
                fig5b = px.bar(imp, x='Ważność', y='Cecha', orientation='h', template='plotly_dark', color='Ważność')
                st.plotly_chart(fig5b, use_container_width=True)
            else:
                st.write("Brak wolumenu do wyliczenia cech.")

    # ==========================================
    # ZAKŁADKA 6: GREKI OPCJI
    # ==========================================
    with tabs[5]:
        st.header("Model Blacka-Scholesa i Zarządzanie Ryzykiem (Greki)")
        
        c6_1, c6_2, c6_3 = st.columns(3)
        c6_4, c6_5, c6_6 = st.columns(3)
        S_opt = c6_1.number_input("Cena Instrumentu Bazowego (S)", value=10000.0)
        K_opt = c6_2.number_input("Cena Wykonania (Strike - K)", value=10000.0)
        T_opt = c6_3.number_input("Czas (T)", value=1.0)
        r_opt = c6_4.number_input("Stopa wolna od ryzyka (r)", value=0.0)
        iv_opt = c6_5.number_input("Zmienność IV (σ)", value=0.16)
        o_type = c6_6.selectbox("Typ Opcji", ['call', 'put'])
        
        if T_opt > 0 and iv_opt > 0:
            price, delta, gamma, theta, vega = bs_greeks(S_opt, K_opt, T_opt, r_opt, iv_opt, o_type)
            
            st.markdown(f"### Wycena Teoretyczna: **{price:.2f}**")
            
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Δ Delta (Ekspozycja kierunkowa)", f"{delta:.4f}")
            g2.metric("Γ Gamma (Zmiana Delty)", f"{gamma:.6f}")
            g3.metric("Θ Theta (Utrata wartości w czasie)", f"{theta:.4f}")
            g4.metric("ν Vega (Ekspozycja na zmienność)", f"{vega:.4f}")
            
            # Symulacja Delty
            prices = np.linspace(S_opt*0.8, S_opt*1.2, 100)
            deltas = [bs_greeks(p, K_opt, T_opt, r_opt, iv_opt, o_type)[1] for p in prices]
            fig6 = px.line(x=prices, y=deltas, title="Krzywa Delty w zależności od ceny S", template='plotly_dark')
            fig6.update_layout(xaxis_title="Cena S", yaxis_title="Delta")
            st.plotly_chart(fig6, use_container_width=True)

    # ==========================================
    # ZAKŁADKA 7: VPIN I TAPE READING
    # ==========================================
    with tabs[6]:
        st.header("Tape Reading i VPIN (Toksyczność Przepływu)")
        if trades_df is not None:
            prod_7 = st.selectbox("Aktywo:", products, key='t7_p')
            df_t7 = trades_df[trades_df['product'] == prod_7].copy()
            df_p7 = prices_df[prices_df['product'] == prod_7].set_index('timestamp')['mid_price']
            
            # VPIN
            df_t7['dir'] = np.where(df_t7['price'].diff() > 0, 1, -1)
            df_t7['b_vol'] = np.where(df_t7['dir'] == 1, df_t7['quantity'], 0)
            df_t7['s_vol'] = np.where(df_t7['dir'] == -1, df_t7['quantity'], 0)
            
            rb = df_t7['b_vol'].rolling(roll_win).sum()
            rs = df_t7['s_vol'].rolling(roll_win).sum()
            df_t7['VPIN'] = abs(rb - rs) / (rb + rs + 0.001)
            
            fig7 = make_subplots(specs=[[{"secondary_y": True}]])
            fig7.add_trace(go.Scatter(x=df_t7['timestamp'], y=df_t7['timestamp'].map(df_p7), name="Cena", line=dict(color='white')), secondary_y=False)
            fig7.add_trace(go.Scatter(x=df_t7['timestamp'], y=df_t7['VPIN'], name="VPIN", line=dict(color='red')), secondary_y=True)
            fig7.add_hline(y=0.7, line_dash="dash", line_color="orange", secondary_y=True)
            fig7.update_layout(height=500, template='plotly_dark', title="Wskaźnik Toksyczności VPIN"); st.plotly_chart(fig7, use_container_width=True)
            
            # Tape Reading
            if 'buyer' in df_t7.columns and df_t7['buyer'].dropna().nunique() > 1:
                st.subheader("Aktywność Traderów")
                b_agg = df_t7.groupby('buyer')['quantity'].sum().reset_index()
                fig7b = px.bar(b_agg.head(15), x='buyer', y='quantity', title="Najwięksi Kupujący", template='plotly_dark', color='quantity')
                st.plotly_chart(fig7b, use_container_width=True)
        else:
            st.warning("Wgraj trades.csv")

    # ==========================================
    # ZAKŁADKA 8: BACKTESTER I KELLY CRITERION
    # ==========================================
    with tabs[7]:
        st.header("Backtester Sygnałów MA & Kelly Criterion")
        st.write("Testujemy, i na podstawie wyników obliczamy optymalne pozycjonowanie kapitału.")
        
        c8_1, c8_2, c8_3 = st.columns(3)
        prod_8 = c8_1.selectbox("Aktywo:", products, key='t8_p')
        fast_ma = c8_2.number_input("Szybka MA", value=10)
        slow_ma = c8_3.number_input("Wolna MA", value=50)
        
        df_8 = prices_df[prices_df['product'] == prod_8].copy()
        df_8['MA_F'] = df_8['mid_price'].rolling(fast_ma).mean()
        df_8['MA_S'] = df_8['mid_price'].rolling(slow_ma).mean()
        
        df_8['Signal'] = np.where(df_8['MA_F'] > df_8['MA_S'], 1, -1)
        df_8['Position'] = df_8['Signal'].shift(1).fillna(0)
        df_8['Trade_Return'] = df_8['Position'] * df_8['mid_price'].diff()
        df_8['Cum_PnL'] = df_8['Trade_Return'].cumsum()
        
        fig8 = make_subplots(specs=[[{"secondary_y": True}]])
        fig8.add_trace(go.Scatter(x=df_8['timestamp'], y=df_8['mid_price'], name="Cena", line=dict(color='gray')), secondary_y=False)
        fig8.add_trace(go.Scatter(x=df_8['timestamp'], y=df_8['Cum_PnL'], name="PnL", line=dict(color=COLOR_UP, width=2)), secondary_y=True)
        fig8.update_layout(height=500, template='plotly_dark'); st.plotly_chart(fig8, use_container_width=True)
        
        # Kelly Criterion
        st.subheader("Zarządzanie Rozmiarem Pozycji (Kryterium Kelly'ego)")
        winning_trades = df_8[df_8['Trade_Return'] > 0]['Trade_Return']
        losing_trades = df_8[df_8['Trade_Return'] < 0]['Trade_Return']
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            W = len(winning_trades) / (len(winning_trades) + len(losing_trades))
            R = winning_trades.mean() / abs(losing_trades.mean())
            
            kelly_pct = W - ((1 - W) / R)
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Win Rate (W)", f"{W*100:.1f}%")
            k2.metric("Risk/Reward Ratio (R)", f"{R:.2f}")
            
            if kelly_pct > 0:
                k3.metric("Optymalny rozmiar pozycji (Kelly)", f"{kelly_pct*100:.1f}% kapitału")
                st.success("Strategia posiada pozytywną wartość oczekiwaną matematycznie.")
            else:
                k3.metric("Optymalny rozmiar pozycji (Kelly)", "0.0% (Nie graj!)")
                st.error("Strategia generuje długoterminową stratę. Kryterium Kelly'ego zaleca brak ekspozycji.")
        else:
            st.write("Za mało transakcji do wyliczenia Kryterium Kelly'ego.")

else:
    st.info("Oczekuję na wgranie plików (prices.csv / trades.csv), Dowódco.")
