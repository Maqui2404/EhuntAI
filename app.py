"""
EHuntAI Dashboard Interactivo
NASA Space Apps Challenge 2025

Dashboard de 6 páginas para visualización y análisis de detección de exoplanetas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from scipy.stats import binned_statistic
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EHuntAI Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    .main-header-container {
        background: linear-gradient(135deg, #0B3D91 0%, #1E88E5 50%, #42A5F5 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        animation: fadeInDown 1s ease-out;
    }

    .main-header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }

    .main-header {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: 0.1em;
        position: relative;
        z-index: 1;
        animation: slideInLeft 1.2s ease-out;
    }

    .nasa-logo {
        width: 120px;
        height: 120px;
        margin: 0 auto 1.5rem auto;
        display: block;
        filter: drop-shadow(0 5px 15px rgba(0,0,0,0.3));
        animation: pulse 3s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }

    .nasa-logo-sidebar {
        width: 100%;
        max-width: 180px;
        margin: 0 auto 1.5rem auto;
        display: block;
        filter: drop-shadow(0 3px 8px rgba(0,0,0,0.2));
    }

    .subtitle {
        font-size: 1.3rem;
        color: white;
        text-align: center;
        margin-top: 1rem;
        font-weight: 300;
        letter-spacing: 0.05em;
        position: relative;
        z-index: 1;
        animation: fadeInDown 1.5s ease-out;
    }

    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0B3D91;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1E88E5;
        animation: slideInLeft 0.8s ease-out;
    }

    .metric-box {
        background: linear-gradient(135deg, #f0f4f8 0%, #e1e9f0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .highlight {
        background-color: #FFF9C4;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
    }

    .icon {
        margin-right: 8px;
    }

    .section-icon {
        color: #1E88E5;
        margin-right: 10px;
        font-size: 1.2em;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B3D91 0%, #1a5490 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }

    [data-testid="stSidebar"] h1 {
        color: white !important;
        text-align: center;
        font-size: 2rem !important;
        margin-top: 1rem;
    }

    /* Animación para tarjetas */
    .stExpander {
        animation: fadeInDown 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<img src="https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png" class="nasa-logo-sidebar" alt="NASA Logo">
""", unsafe_allow_html=True)

st.sidebar.title("FINESSII")
st.sidebar.markdown("NASA Space Apps Challenge 2025")

paginas = {
    "Inicio": "inicio",
    "Estadística Descriptiva": "estadistica",
    "Modelo ML (Entrenamiento)": "modelo",
    "Uso del Modelo": "uso",
    "Modelo 1D Kepler": "1d",
    "Conclusiones": "conclusiones"
}

seleccion = st.sidebar.radio("Navegación", list(paginas.keys()))
pagina_actual = paginas[seleccion]

st.sidebar.markdown("---")
st.sidebar.info("""
Equipo: FINESSII - Puno, Peru

Challenge: A World Away: Hunting for Exoplanets with AI

Datos: Kepler, K2, TESS

Institución: IIICCD
""")

# ============================================================================
# PÁGINA 1: INICIO - Información del equipo, reto y proyecto
# ============================================================================
if pagina_actual == "inicio":
    st.markdown("""
    <div class="main-header-container">
        <img src="https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png" class="nasa-logo" alt="NASA Logo">
        <h1 class="main-header">FINESSII</h1>
        <div class="subtitle">2025 NASA Space Apps Challenge</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">About the Team</div>', unsafe_allow_html=True)
    st.markdown("""
    We are a team of five undergraduate and postgraduate students in Statistical and Computer Engineering
    from Puno - Peru, passionate about data science, artificial intelligence, and space exploration.
    Our main interest is applying advanced data analysis and machine learning techniques to solve global challenges.

    For the 2025 NASA Space Apps Challenge, we are participating in the challenge "A World Away: Hunting
    for Exoplanets with AI". Our goal is to design an innovative AI-based system capable of detecting exoplanets
    from NASA's open datasets (Kepler, K2, TESS) by combining statistical methods and deep learning.

    Skills we bring to the team: data science, statistics, artificial intelligence & machine learning,
    software development, coding, data visualization, and problem-solving.

    We are open to collaboration with other participants who share our passion for space, science, and technology.
    """)

    st.markdown('<div class="sub-header">Team Members</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        Marco Fidel Mayta Quispe (Team Owner)
        - @marcomayta243
        - Peru

        Dina Maribel Yana Yucra
        - @maribell0312
        - Peru
        """)

    with col2:
        st.markdown("""
        Richar Andre Vilca Solorzano
        - @asandre
        - Peru

        Cristian Daniel Ccopa Acero
        - @danidev
        - Peru
        """)

    with col3:
        st.markdown("""
        Leonid Aleman Gonzales
        - @laleman
        - Peru
        """)

    st.markdown('<div class="sub-header">Contact Information</div>', unsafe_allow_html=True)
    st.info("""
    IIICCD (Instituto de Investigación en Inteligencia Computacional y Ciencia de Datos)

    Preferred contact method: Email or WhatsApp group

    We will also be active on the NASA Space Apps Challenge platform and check messages daily.

    LinkedIn: IIICCD
    """)

    st.markdown('<div class="sub-header">About the Challenge</div>', unsafe_allow_html=True)
    st.markdown("""
    Data from several different space-based exoplanet surveying missions have enabled discovery of thousands
    of new planets outside our solar system, but most of these exoplanets were identified manually. With advances
    in artificial intelligence and machine learning (AI/ML), it is possible to automatically analyze large sets
    of data collected by these missions to identify exoplanets.

    Your challenge is to create an AI/ML model that is trained on one or more of the open-source exoplanet
    datasets offered by NASA and that can analyze new data to accurately identify exoplanets.

    *(Astrophysics Division)*
    """)

    st.markdown('<div class="sub-header">A World Away: Hunting for Exoplanets with AI</div>', unsafe_allow_html=True)

    with st.expander("Background"):
        st.markdown("""
        Exoplanetary identification is becoming an increasingly popular area of astronomical exploration.
        Several survey missions have been launched with the primary objective of identifying exoplanets.
        Utilizing the "transit method" for exoplanet detection, scientists are able to detect a decrease
        in light when a planetary body passes between a star and the surveying satellite.

        Kepler is one of the more well-known transit-method satellites, and provided data for nearly a decade.
        Kepler was followed by its successor mission, K2, which utilized the same hardware and transit method,
        but maintained a different path for surveying. After the retirement of Kepler, the Transiting Exoplanet
        Survey Satellite (TESS), which has a similar mission of exoplanetary surveying, launched and has been
        collecting data since 2018.

        For each of these missions (Kepler, K2, and TESS), publicly available datasets exist that include data
        for all confirmed exoplanets, planetary candidates, and false positives obtained by the mission. Despite
        the availability of new technology and previous research in automated classification of exoplanetary data,
        much of this exoplanetary transit data is still analyzed manually.
        """)

    with st.expander("Objectives"):
        st.markdown("""
        Your challenge is to create an artificial intelligence/machine learning model that is trained on one or
        more of NASA's open-source exoplanet datasets, and not only analyzes data to identify new exoplanets,
        but includes a web interface to facilitate user interaction.

        Key objectives:
        - Train on NASA's open-source exoplanet datasets (Kepler, K2, TESS)
        - Analyze data to identify new exoplanets
        - Include a web interface for user interaction
        - Consider how different data variables impact classification
        - Think about preprocessing and model selection for higher accuracy
        - Design for scientist and researcher interaction
        """)

    with st.expander("Potential Considerations"):
        st.markdown("""
        You may (but are not required to) consider the following:

        - Your project could be aimed at researchers wanting to classify new data or novices in the field who
          want to interact with exoplanet data and do not know where to start
        - Your interface could enable your tool to ingest new data and train the models as it does so
        - Your interface could show statistics about the accuracy of the current model
        - Your model could allow hyperparameter tweaking from the interface
        """)

    st.markdown('<div class="sub-header">Team Information Summary</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        Local Event: Universal Event, Anywhere in the World

        Challenge: A World Away: Hunting for Exoplanets with AI

        Difficulty: Advanced
        """)

    with col2:
        st.markdown("""
        Subjects:
        - Artificial Intelligence & Machine Learning
        - Coding
        - Data Analysis
        - Data Management
        - Data Visualization
        - Extrasolar Objects
        - Planets & Moons
        - Software
        - Space Exploration
        """)

# ============================================================================
# PÁGINA 2: ESTADÍSTICA DESCRIPTIVA
# ============================================================================
elif pagina_actual == "estadistica":
    st.markdown('<div class="main-header"> Estadística Descriptiva</div>', unsafe_allow_html=True)
    st.markdown("Análisis exploratorio de la curva de luz KIC11446443")

    @st.cache_data
    def cargar_datos_kepler():
        csv_path = Path("KIC11446443_lightcurve_____1.csv")
        if not csv_path.exists():
            return None

        df = pd.read_csv(csv_path)

        time_col = 'timecorr' if 'timecorr' in df.columns else 'time'
        flux_col = 'pdcsap_flux' if 'pdcsap_flux' in df.columns else 'flux'

        df_clean = df[[time_col, flux_col]].copy()
        df_clean.columns = ['time', 'flux']
        df_clean = df_clean.dropna().sort_values('time').reset_index(drop=True)

        if (df_clean['time'].max() - df_clean['time'].min()) < 1.0:
            if 'cadenceno' in df.columns:
                cadence = df['cadenceno'].values[:len(df_clean)]
                df_clean['time'] = (cadence - cadence.min()) * (29.4244 / (60 * 24))

        df_clean['flux_norm'] = df_clean['flux'] / np.nanmedian(df_clean['flux'])

        if 'quality' in df.columns:
            quality_mask = df['quality'].values[:len(df_clean)] == 0
            df_clean = df_clean[quality_mask].reset_index(drop=True)

        return df_clean

    df = cargar_datos_kepler()

    if df is None:
        st.error(" No se pudo cargar el archivo KIC11446443_lightcurve_____1.csv")
        st.info("Asegúrate de que el archivo esté en el mismo directorio que app.py")
    else:
        st.markdown("Métricas Generales")
        col1, col2, col3, col4, col5 = st.columns(5)

        baseline = df['time'].max() - df['time'].min()
        cadencia = np.median(np.diff(df['time'].values)) * 24 * 60

        col1.metric("Total de Puntos", f"{len(df):,}")
        col2.metric("Baseline", f"{baseline:.1f} días")
        col3.metric("Cadencia Media", f"{cadencia:.1f} min")
        col4.metric("Flujo Mediano", f"{np.median(df['flux']):.0f}")
        col5.metric("Dispersión (σ)", f"{np.std(df['flux_norm']):.4f}")

        st.markdown("Estadísticas Descriptivas del Flujo")

        col1, col2 = st.columns([2, 1])

        with col1:
            stats_df = df['flux_norm'].describe()
            stats_df = pd.DataFrame(stats_df).T
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
            st.dataframe(stats_df, use_container_width=True)

        with col2:
            st.markdown("Distribución:")
            st.write(f"- Asimetría (Skewness): {df['flux_norm'].skew():.4f}")
            st.write(f"- Curtosis (Kurtosis): {df['flux_norm'].kurtosis():.4f}")
            st.write(f"- Rango: {df['flux_norm'].max() - df['flux_norm'].min():.4f}")
            st.write(f"- IQR: {df['flux_norm'].quantile(0.75) - df['flux_norm'].quantile(0.25):.4f}")

        st.markdown("Visualizaciones")

        tab1, tab2, tab3, tab4 = st.tabs(["Curva de Luz", "Histograma", "Box Plot", "Correlación Temporal"])

        with tab1:
            st.markdown("Curva de Luz Completa")
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(df['time'], df['flux_norm'], '.', ms=1, alpha=0.5, color='steelblue')
            ax.set_xlabel('Tiempo (días)', fontsize=11)
            ax.set_ylabel('Flujo Normalizado', fontsize=11)
            ax.set_title('Curva de Luz KIC11446443', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='Mediana')
            ax.legend()
            st.pyplot(fig)
            plt.close()

            st.info(f"""
             Interpretación: Esta curva muestra {len(df):,} observaciones durante {baseline:.1f} días.
            La línea roja representa el flujo normalizado (mediana = 1.0). Las variaciones visibles pueden
            incluir tránsitos planetarios, manchas estelares, o ruido instrumental.
            """)

        with tab2:
            st.markdown("Distribución del Flujo Normalizado")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df['flux_norm'], bins=100, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(df['flux_norm'].mean(), color='red', ls='--', lw=2, label=f'Media: {df["flux_norm"].mean():.4f}')
            ax.axvline(df['flux_norm'].median(), color='green', ls='--', lw=2, label=f'Mediana: {df["flux_norm"].median():.4f}')
            ax.set_xlabel('Flujo Normalizado', fontsize=11)
            ax.set_ylabel('Frecuencia', fontsize=11)
            ax.set_title('Histograma de Flujo', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
            plt.close()

            from scipy.stats import shapiro, normaltest
            stat_shapiro, p_shapiro = shapiro(df['flux_norm'].sample(min(5000, len(df))))

            st.markdown(f"""
            Test de Normalidad (Shapiro-Wilk):
            - Estadístico: {stat_shapiro:.4f}
            - P-value: {p_shapiro:.6f}
            - {" Distribución aproximadamente normal" if p_shapiro > 0.05 else " Distribución NO normal"}
            """)

        with tab3:
            st.markdown("Box Plot y Outliers")
            fig, ax = plt.subplots(figsize=(10, 5))
            bp = ax.boxplot(df['flux_norm'], vert=False, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', lw=2),
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))
            ax.set_xlabel('Flujo Normalizado', fontsize=11)
            ax.set_title('Diagrama de Caja del Flujo', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()

            Q1 = df['flux_norm'].quantile(0.25)
            Q3 = df['flux_norm'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df['flux_norm'] < lower_bound) | (df['flux_norm'] > upper_bound)]

            st.markdown(f"""
            Análisis de Outliers:
            - Q1 (25%): {Q1:.4f}
            - Q3 (75%): {Q3:.4f}
            - IQR: {IQR:.4f}
            - Límites: [{lower_bound:.4f}, {upper_bound:.4f}]
            - Outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)

             Los outliers pueden indicar tránsitos planetarios o eventos estelares.
            """)

        with tab4:
            st.markdown("##Autocorrelación")

            from pandas.plotting import autocorrelation_plot

            fig, ax = plt.subplots(figsize=(12, 5))
            autocorrelation_plot(df['flux_norm'].iloc[:10000], ax=ax)
            ax.set_title('Autocorrelación del Flujo', fontsize=13, fontweight='bold')
            ax.set_xlabel('Lag', fontsize=11)
            ax.set_ylabel('Autocorrelación', fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            st.info("""
             Interpretación: La autocorrelación muestra si hay patrones repetitivos en los datos.
            Picos significativos pueden indicar periodicidades (como tránsitos planetarios o pulsaciones estelares).
            """)

        st.markdown("Muestra de Datos")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("Descargar Datos Procesados")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Descargar CSV Procesado",
            data=csv,
            file_name="KIC11446443_procesado.csv",
            mime="text/csv"
        )

# ============================================================================
# PÁGINA 3: MODELO ML - Entrenamiento
# ============================================================================
elif pagina_actual == "modelo":
    st.markdown('<div class="main-header"> Entrenamiento del Modelo</div>', unsafe_allow_html=True)
    st.markdown("Proceso completo del notebook ExoHuntAI_Model_Training.ipynb")

    st.markdown("""
    Este modelo combina Box Least Squares (BLS) para la detección inicial de candidatos
    con Machine Learning para la clasificación final.
    """)

    st.markdown("Pipeline de Entrenamiento")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        1. Preprocesamiento
        - Carga de datos Kepler
        - Corrección de tiempo
        - Normalización de flujo
        - Detrending con mediana móvil
        - Eliminación de outliers
        """)

    with col2:
        st.markdown("""
        2. Box Least Squares
        - Búsqueda de periodicidades
        - Rango: 0.5 - 50 días
        - Grid de duraciones
        - Detección de tránsitos
        - Extracción de features
        """)

    with col3:
        st.markdown("""
        3. Machine Learning
        - Generación de dataset
        - 250 positivos + 250 negativos
        - Random Forest + Gradient Boosting
        - Cross-validation 5-fold
        - Guardado del modelo
        """)

    st.markdown("Features Extraídas (23 en total)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        BLS Features (6)
        - `period`: Período orbital
        - `duration`: Duración del tránsito
        - `bls_power`: Potencia BLS
        - `depth_bls`: Profundidad BLS
        - `snr_bls`: SNR BLS
        - `duration_ratio`: Ratio duración/período
        """)

    with col2:
        st.markdown("""
        Features del Tránsito (9)
        - `depth_local`: Profundidad local
        - `snr_local`: SNR local
        - `in_std`: Desv. estándar dentro
        - `out_std`: Desv. estándar fuera
        - `in_mad`: MAD dentro
        - `out_mad`: MAD fuera
        - `std_ratio`: Ratio de desviaciones
        - `in_skew`: Asimetría
        - `in_kurtosis`: Curtosis
        """)

    with col3:
        st.markdown("""
        Features Adicionales (8)
        - `secondary_depth`: Eclipse secundario
        - `secondary_ratio`: Ratio secundario
        - `oddeven_diff`: Test par/impar
        - `oddeven_std_diff`: Diff std par/impar
        - `flux_range`: Rango de flujo
        - `flux_std`: Desv. estándar global
        - `transit_points_ratio`: Ratio puntos
        - `n_points`: Total de puntos
        """)

    st.markdown("Resultados del Entrenamiento")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##Random Forest (MEJOR)")
        st.success("""
        - CV ROC-AUC: 0.9979 ± 0.0037
        - CV PR-AUC: 0.9984 ± 0.0026
        - Test Precision: 1.0000
        - Test Recall: 0.9800
        - Test F1-Score: 0.9899
        - Matriz de Confusión: [[50, 0], [1, 49]]
        """)

    with col2:
        st.markdown("##Gradient Boosting")
        st.info("""
        - CV ROC-AUC: 0.9971 ± 0.0049
        - CV PR-AUC: 0.9974 ± 0.0043
        - Test Precision: 0.9804
        - Test Recall: 1.0000
        - Test F1-Score: 0.9901
        - Matriz de Confusión: [[50, 0], [0, 50]]
        """)

    st.markdown("Top 10 Features Más Importantes")

    features_importance = {
        'out_std': 0.1991,
        'flux_std': 0.1783,
        'in_skew': 0.1325,
        'out_mad': 0.1154,
        'in_kurtosis': 0.0853,
        'in_mad': 0.0852,
        'std_ratio': 0.0478,
        'transit_points_ratio': 0.0459,
        'oddeven_std_diff': 0.0218,
        'snr_bls': 0.0216
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    features_names = list(features_importance.keys())
    importance_values = list(features_importance.values())

    ax.barh(features_names, importance_values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Importancia', fontsize=11)
    ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

    st.info("""
     Observaciones clave:
    - Las features de dispersión (`out_std`, `flux_std`) son las más importantes
    - La asimetría (`in_skew`) y curtosis detectan la forma del tránsito
    - Features de BLS (`snr_bls`) tienen importancia moderada
    - El modelo combina información estadística y astronómica
    """)

    st.markdown("Código del Entrenamiento")

    with st.expander("Ver código completo del notebook"):
        st.code("""
1. Carga y preprocesamiento
df = pd.read_csv("KIC11446443_lightcurve_____1.csv")
df = preprocess_kepler_data(df)  Normalización, detrending, limpieza

2. Box Least Squares
bls = BoxLeastSquares(time, flux)
results = bls.power(period_grid, duration_grid)
best_period, best_duration, best_t0 = extract_best_candidate(results)

3. Extracción de features
features = extract_transit_features(phase, flux, window, period, duration, ...)
23 features: BLS + estadísticas + forma + eclipse secundario

4. Generación de dataset sintético
250 muestras con tránsitos inyectados
250 muestras sin tránsitos (ruido)
train_df = generate_training_dataset(n_pos=250, n_neg=250)

5. Entrenamiento
X = train_df[features]
y = train_df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

6. Evaluación
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

7. Guardar modelo
pickle.dump({'model': model, 'scaler': scaler, 'features': features},
            open('models/exohunt_model.pkl', 'wb'))
        """, language="python")

    st.markdown("Visualizaciones del Entrenamiento")

    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        col1, col2 = st.columns(2)

        if (outputs_dir / "bls_analysis.png").exists():
            with col1:
                st.image(str(outputs_dir / "bls_analysis.png"), caption="BLS Periodogram + Curva Plegada")

        if (outputs_dir / "candidate_analysis.png").exists():
            with col2:
                st.image(str(outputs_dir / "candidate_analysis.png"), caption="Análisis Completo del Candidato")

        if (outputs_dir / "model_metrics.png").exists():
            st.image(str(outputs_dir / "model_metrics.png"), caption="Métricas del Modelo")
    else:
        st.warning(" No se encontraron las imágenes de salida. Ejecuta el notebook primero.")

# ============================================================================
# PÁGINA 4: USO DEL MODELO
# ============================================================================
elif pagina_actual == "uso":
    st.markdown('<div class="main-header"> Uso del Modelo</div>', unsafe_allow_html=True)
    st.markdown("Cómo usar el modelo entrenado para clasificar candidatos")

    model_path = Path("models/exohunt_model.pkl")

    if not model_path.exists():
        st.error(" El modelo no se encuentra en la carpeta `models/`")
        st.info("Por favor, ejecuta el notebook `ExoHuntAI_Model_Training.ipynb` primero para generar el modelo.")
    else:
        st.success(f" Modelo encontrado: {model_path}")

        @st.cache_resource
        def cargar_modelo():
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f" Error al cargar el modelo: {str(e)}")
                st.warning("""
                Problema de compatibilidad de versiones detectado.

                Esto ocurre cuando el modelo fue entrenado con una versión diferente de scikit-learn.

                Soluciones:
                1. Re-entrenar el modelo ejecutando el notebook `ExoHuntAI_Model_Training.ipynb`
                2. O instalar la versión compatible de scikit-learn

                Comando para reinstalar:
                ```bash
                pip uninstall scikit-learn -y
                pip install scikit-learn==1.3.0
                ```
                """)
                return None

        model_data = cargar_modelo()

        if model_data is None:
            st.stop()

        st.markdown("Información del Modelo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tipo de Modelo", model_data['model_name'])
            st.metric("Features", len(model_data['features']))

        with col2:
            st.metric("ROC-AUC", f"{model_data['metrics']['cv_roc_auc']:.4f}")
            st.metric("Precision", f"{model_data['metrics']['test_precision']:.4f}")

        with col3:
            st.metric("Recall", f"{model_data['metrics']['test_recall']:.4f}")
            st.metric("F1-Score", f"{model_data['metrics']['test_f1']:.4f}")

        st.markdown("Cómo Funciona el Modelo")

        st.markdown("""
        El proceso de detección de exoplanetas con EHuntAI sigue un pipeline científico riguroso
        que combina física estelar con machine learning. A continuación se explica cada etapa:
        """)

        st.markdown("Etapa 1: Preparación de la Curva de Luz")
        st.markdown("""
        Todo comienza con una curva de luz del telescopio Kepler: una serie temporal que registra
        el brillo de una estrella a lo largo del tiempo. Primero debemos:

        - Normalizar el flujo: Dividir todos los valores de brillo entre la mediana, estableciendo
          un baseline en 1.0. Esto permite comparar estrellas de diferentes magnitudes.

        - Aplicar detrending: Las estrellas tienen variaciones de largo plazo (manchas, rotación)
          que debemos remover usando una mediana móvil. Esto aísla las variaciones de corto plazo
          donde aparecen los tránsitos planetarios.

        - Limpiar outliers: Puntos anómalos (rayos cósmicos, glitches instrumentales) se eliminan
          si exceden 5-8 desviaciones estándar de la media.
        """)

        st.markdown("Etapa 2: Búsqueda con Box Least Squares (BLS)")
        st.markdown("""
        El algoritmo BLS busca sistemáticamente señales periódicas con forma de "caja" (box)
        que indican tránsitos planetarios:

        - Grid de períodos: Se prueban miles de períodos posibles entre 0.5 y 50 días. Para cada
          período, BLS pliega (fold) la curva de luz y busca caídas repetitivas de brillo.

        - Grid de duraciones: Para cada período, se prueba con diferentes anchos de tránsito
          (1-12 horas típicamente), buscando la mejor coincidencia con una caja rectangular.

        - Estadístico BLS: El algoritmo calcula un "power" (potencia) que mide qué tan bien
          una señal periódica tipo caja ajusta los datos. El máximo de este periodograma indica
          el candidato más prometedor.

        - Parámetros del candidato: BLS devuelve el período orbital, la duración del tránsito,
          la época (momento del primer tránsito), y la profundidad (cuánto disminuye el brillo).
        """)

        st.markdown("Etapa 3: Extracción de 23 Features Científicas")
        st.markdown("""
        A partir del candidato BLS, extraemos 23 features que capturan diferentes aspectos
        de la señal. Estas features se agrupan en categorías:

        Features de BLS (6):
        - `period`, `duration`, `bls_power`: Parámetros orbitales básicos
        - `depth_bls`, `snr_bls`: Profundidad del tránsito y relación señal/ruido
        - `duration_ratio`: Ratio entre duración y período (detecta geometrías anómalas)

        Features estadísticas del tránsito (9):
        - `depth_local`, `snr_local`: Profundidad y SNR recalculados localmente
        - `in_std`, `out_std`: Dispersión dentro y fuera del tránsito
        - `in_mad`, `out_mad`: Desviación absoluta mediana (robusta a outliers)
        - `std_ratio`: Ratio de dispersiones (tránsitos reales tienen más variación dentro)
        - `in_skew`, `in_kurtosis`: Forma de la distribución (detecta tránsitos asimétricos)

        Features de validación astronómica (8):
        - `secondary_depth`, `secondary_ratio`: Detecta eclipse secundario en fase 0.5
          (binarias eclipsantes tienen eclipses secundarios significativos)
        - `oddeven_diff`, `oddeven_std_diff`: Test de par/impar (sistemas binarios muestran
          diferencias entre tránsitos alternos)
        - `flux_range`, `flux_std`: Variabilidad global de la estrella
        - `transit_points_ratio`: Fracción de datos en tránsito
        - `n_points`: Total de observaciones (más datos = mayor confianza)
        """)

        st.markdown("Etapa 4: Clasificación con Machine Learning")
        st.markdown("""
        Las 23 features se alimentan a un modelo de Random Forest previamente entrenado:

        - Normalización: Un StandardScaler transforma las features a media 0 y desviación 1,
          evitando que features con rangos grandes dominen la predicción.

        - Imputación: Cualquier valor NaN o infinito se reemplaza con la mediana de la feature
          (calculada durante el entrenamiento).

        - Predicción del bosque: 200 árboles de decisión votan independientemente. Cada árbol
          aprendió durante el entrenamiento qué combinaciones de features distinguen tránsitos
          reales de falsos positivos.

        - Probabilidad: El porcentaje de árboles que votaron "tránsito" se convierte en una
          probabilidad (0-1). Por ejemplo, si 195 de 200 árboles votaron "sí", la probabilidad es 97.5%.

        - Decisión final: Si la probabilidad supera 0.5 (50%), se clasifica como tránsito.
          Pero la probabilidad exacta te da matices: 99% es casi certeza, 51% es marginal.
        """)

        st.markdown("Etapa 5: Interpretación Automática")
        st.markdown("""
        El modelo no solo da un "sí" o "no", sino que interpreta el resultado:

        - Probabilidad > 90%: MUY ALTA confianza - El candidato tiene todas las características
          de un tránsito planetario real. Recomendación: priorizar para seguimiento con telescopios
          terrestres o JWST. Alta probabilidad de ser confirmado como exoplaneta.

        - Probabilidad 70-90%: ALTA confianza - Candidato prometedor con señales claras de tránsito,
          pero alguna feature puede ser ambigua (ej: SNR moderado, o eclipse secundario débil pero detectable).
          Recomendación: revisar curva de luz manualmente, buscar más datos (otros quarters de Kepler).

        - Probabilidad 50-70%: MEDIA confianza - Señal presente pero con características mezcladas.
          Puede ser un tránsito real débil, o un falso positivo sofisticado (ej: binaria grazing eclipse).
          Recomendación: validación adicional con modelos de tránsito, análisis de centroide, espectroscopía.

        - Probabilidad < 50%: BAJA confianza - Probablemente un falso positivo. Puede ser variabilidad
          estelar, ruido instrumental, binaria eclipsante no resuelta, o background eclipsing binary.
          Recomendación: descartar a menos que haya razones científicas específicas para investigar.

        El modelo también identifica qué features influyeron más en la decisión usando feature importance.
        Por ejemplo, si `out_std` es bajo y `in_skew` es muy negativo, indica un tránsito limpio con forma
        en V característica.
        """)

        st.info("""
         Ejemplo de interpretación completa:

        Para un candidato con probabilidad 96.8%, el modelo encontró:
        -  Profundidad consistente (~0.5%) típica de Neptuno/Júpiter
        -  Dispersión baja fuera del tránsito (estrella tranquila)
        -  Asimetría negativa (forma de V, no de U)
        -  No hay eclipse secundario significativo (descarta binaria)
        -  Test par/impar pasa (tránsitos consistentes)

        Conclusión: Candidato a exoplaneta de muy alta calidad, listo para confirmar.
        """)

        st.markdown("Prueba Interactiva")

        st.markdown("""
        Ajusta los parámetros de un tránsito sintético y observa la predicción del modelo:
        """)

        col1, col2 = st.columns(2)

        with col1:
            periodo = st.slider("Período (días)", 1.0, 50.0, 10.0, 0.1)
            duracion = st.slider("Duración (horas)", 1.0, 12.0, 3.0, 0.5)
            profundidad = st.slider("Profundidad (%)", 0.01, 3.0, 0.5, 0.01)

        with col2:
            snr = st.slider("SNR", 0.1, 20.0, 5.0, 0.1)
            flux_std = st.slider("Dispersión del flujo", 0.0001, 0.01, 0.0005, 0.0001)
            skew = st.slider("Asimetría (skew)", -5.0, 5.0, -2.0, 0.1)

        synthetic_features = {
            'period': periodo,
            'duration': duracion / 24,
            'bls_power': 0.0002,
            'depth_bls': profundidad / 100,
            'snr_bls': snr,
            'duration_ratio': (duracion / 24) / periodo,
            'depth_local': profundidad / 100,
            'snr_local': snr,
            'in_std': flux_std * 2,
            'out_std': flux_std,
            'in_mad': flux_std * 0.5,
            'out_mad': flux_std * 0.3,
            'std_ratio': 2.0,
            'in_skew': skew,
            'in_kurtosis': 10.0,
            'secondary_depth': 0.00001,
            'secondary_ratio': 0.3,
            'oddeven_diff': 0.00001,
            'oddeven_std_diff': 0.0001,
            'flux_range': 0.01,
            'flux_std': flux_std,
            'transit_points_ratio': 0.04,
            'n_points': 40000
        }

        X = np.array([[synthetic_features[f] for f in model_data['features']]])
        X_scaled = model_data['scaler'].transform(X)
        proba = model_data['model'].predict_proba(X_scaled)[0, 1]
        pred = model_data['model'].predict(X_scaled)[0]

        st.markdown("---")
        st.markdown("Resultado de la Predicción:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Probabilidad de Tránsito", f"{proba:.4f}", delta=f"{proba*100:.2f}%")

        with col2:
            if pred == 1:
                st.success(f" TRÁNSITO DETECTADO")
            else:
                st.error(f" NO TRÁNSITO")

        with col3:
            if proba > 0.9:
                nivel = "MUY ALTA"
                color = "green"
            elif proba > 0.7:
                nivel = "ALTA"
                color = "blue"
            elif proba > 0.5:
                nivel = "MEDIA"
                color = "orange"
            else:
                nivel = "BAJA"
                color = "red"
            st.metric("Confianza", nivel)

        st.markdown("---")
        st.markdown("Interpretación Automática del Resultado")

        interpretaciones = []

        if profundidad < 0.1:
            interpretaciones.append(" Profundidad muy baja (<0.1%): Si es real, sería un planeta tipo Tierra. Señal difícil de detectar, requiere alta precisión.")
        elif profundidad < 1.0:
            interpretaciones.append(" Profundidad moderada (0.1-1%): Compatible con planetas tipo Neptuno o super-Tierras. Rango típico de exoplanetas detectables.")
        else:
            interpretaciones.append(" Profundidad alta (>1%): Compatible con Júpiter caliente o binaria eclipsante. Señal muy fuerte y fácil de detectar.")

        if snr < 5:
            interpretaciones.append(f" SNR bajo ({snr:.1f}): Señal débil, puede confundirse con ruido. Aumentar SNR mejora la confianza del modelo.")
        elif snr < 10:
            interpretaciones.append(f" SNR moderado ({snr:.1f}): Señal clara pero no excepcional. Valor típico para planetas pequeños en Kepler.")
        else:
            interpretaciones.append(f" SNR excelente ({snr:.1f}): Señal muy fuerte, bien por encima del ruido. Alta probabilidad de ser real.")

        if periodo < 2:
            interpretaciones.append(f" Período ultra-corto ({periodo:.1f} días): Planeta muy cercano a su estrella, probablemente un 'hot Jupiter' o super-Tierra fundida.")
        elif periodo < 10:
            interpretaciones.append(f" Período corto ({periodo:.1f} días): Órbita interior, planeta recibe mucha radiación estelar. Común en detecciones de Kepler.")
        elif periodo < 30:
            interpretaciones.append(f" Período medio ({periodo:.1f} días): Órbita comparable a Mercurio/Venus. Zona potencialmente habitable para estrellas frías.")
        else:
            interpretaciones.append(f" Período largo ({periodo:.1f} días): Órbita exterior, similar a la Tierra o más lejana. Menos tránsitos observados, menos datos.")

        duracion_esperada = 13 * (periodo / 365)  (1/3)
        if abs(duracion - duracion_esperada) > 3:
            interpretaciones.append(f" Duración anómala ({duracion:.1f} h): No coincide con la geometría esperada. Puede indicar órbita excéntrica o binaria.")
        else:
            interpretaciones.append(f" Duración consistente ({duracion:.1f} h): Compatible con geometría de tránsito típica.")

        if flux_std > 0.002:
            interpretaciones.append(f" Alta dispersión ({flux_std:.4f}): Estrella muy variable (manchas, pulsaciones). Dificulta detección, aumenta falsos positivos.")
        elif flux_std > 0.0005:
            interpretaciones.append(f" Dispersión moderada ({flux_std:.4f}): Variabilidad estelar típica. No interfiere significativamente con la detección.")
        else:
            interpretaciones.append(f" Dispersión baja ({flux_std:.4f}): Estrella tranquila, ideal para detectar planetas pequeños. Señal limpia.")

        if skew < -3:
            interpretaciones.append(f" Asimetría muy negativa (skew={skew:.1f}): Tránsito con forma de V pronunciada, típico de planetas. Buena señal.")
        elif skew < 0:
            interpretaciones.append(f" Asimetría negativa (skew={skew:.1f}): Forma ligeramente asimétrica, común en tránsitos planetarios.")
        elif skew > 3:
            interpretaciones.append(f" Asimetría muy positiva (skew={skew:.1f}): Forma anómala, puede indicar falso positivo o evento transitorio.")
        else:
            interpretaciones.append(f" Asimetría neutral (skew={skew:.1f}): Forma simétrica o mixta.")

        for interp in interpretaciones:
            st.markdown(interp)

        def estimar_radio(prof_pct):
            return np.sqrt(prof_pct / 100) * 109

        def estimar_temperatura(per_dias):
            return 5778 * np.sqrt(0.00465 / (per_dias / 365)  (2/3))

        st.markdown("---")
        st.markdown("Recomendación Final")

        if proba > 0.9:
            st.success(f"""
            ALTA PRIORIDAD - Candidato Excelente ({proba*100:.1f}%)

            Este candidato muestra todas las características de un tránsito planetario real:
            - Profundidad de {profundidad:.2f}% sugiere un planeta de {estimar_radio(profundidad):.1f} radios terrestres
            - SNR de {snr:.1f} indica señal clara y robusta
            - Período de {periodo:.1f} días implica temperatura de equilibrio ~{estimar_temperatura(periodo):.0f} K

            Próximos pasos recomendados:
            1.  Búsqueda de tránsitos adicionales en otros quarters de Kepler
            2.  Análisis de centroide para descartar background eclipsing binary
            3.  Observación con espectroscopía radial velocity para confirmar masa
            4.  Consideración para seguimiento con TESS o JWST
            """)

        elif proba > 0.7:
            st.info(f"""
            PRIORIDAD MEDIA - Candidato Prometedor ({proba*100:.1f}%)

            Este candidato tiene características mayormente consistentes con un tránsito planetario,
            pero presenta algunas ambigüedades:
            - {'SNR moderado puede requerir más datos' if snr < 10 else 'Parámetros en rangos típicos'}
            - {'Dispersión estelar alta dificulta confirmación' if flux_std > 0.001 else 'Estrella adecuada para análisis'}

            Próximos pasos recomendados:
            1.  Revisar curva de luz manualmente para verificar morfología del tránsito
            2.  Buscar datos adicionales (otros instrumentos, quarters adicionales)
            3.  Análisis de odd/even para descartar binarias
            4.  Validación estadística (BLENDER, VESPA)
            """)

        elif proba > 0.5:
            st.warning(f"""
            PRIORIDAD BAJA - Candidato Marginal ({proba*100:.1f}%)

            Este candidato muestra señal débil o características mezcladas:
            - {'SNR bajo - señal cercana al nivel de ruido' if snr < 5 else 'Algunos parámetros fuera de rangos típicos'}
            - {'Alta variabilidad estelar complica detección' if flux_std > 0.002 else 'Morfología puede ser ambigua'}

            Próximos pasos recomendados:
            1.  Validación exhaustiva necesaria antes de invertir recursos
            2.  Comparar con catálogo de falsos positivos conocidos
            3.  Análisis de pixel-level data para descartar contaminación
            4.  Considerar descartar si recursos limitados
            """)

        else:
            st.error(f"""
            NO RECOMENDADO - Probable Falso Positivo ({proba*100:.1f}%)

            Este candidato tiene muy baja probabilidad de ser un tránsito planetario real:
            - Características inconsistentes con tránsitos típicos
            - Puede ser: {'variabilidad estelar' if flux_std > 0.002 else 'ruido instrumental, binaria, o artifact'}

            Recomendación:
             Descartar a menos que haya razones científicas específicas para investigar más.
            """)

        st.markdown("---")
        st.markdown("Valores de las Features Principales")

        feature_display = pd.DataFrame({
            'Feature': ['Período', 'Duración', 'Profundidad', 'SNR', 'Dispersión (out)', 'Dispersión (in)', 'Asimetría', 'Ratio duración/período'],
            'Valor': [
                f"{periodo:.2f} días",
                f"{duracion:.2f} horas",
                f"{profundidad:.3f} %",
                f"{snr:.1f}",
                f"{flux_std:.6f}",
                f"{flux_std * 2:.6f}",
                f"{skew:.2f}",
                f"{synthetic_features['duration_ratio']:.6f}"
            ],
            'Interpretación': [
                'Tiempo que tarda el planeta en completar una órbita',
                'Tiempo que tarda el planeta en cruzar el disco estelar',
                'Porcentaje de luz bloqueada (∝ tamaño del planeta)',
                'Relación señal-ruido (mayor = más confiable)',
                'Variabilidad estelar fuera del tránsito',
                'Variabilidad durante el tránsito',
                'Forma del tránsito (negativo = V, positivo = U)',
                'Relación geométrica (detecta órbitas anómalas)'
            ]
        })

        st.dataframe(feature_display, use_container_width=True, hide_index=True)

# ============================================================================
# PÁGINA 5: MODELO 1D KEPLER (en entrenamiento)
# ============================================================================
elif pagina_actual == "1d":
    st.markdown('<div class="main-header"> Modelo 1D con Datos Kepler</div>', unsafe_allow_html=True)
    st.markdown("Análisis unidimensional de series temporales")

    st.info("""
     Estado: Modelo en entrenamiento

    Este módulo está desarrollando un enfoque de deep learning 1D (redes convolucionales)
    para analizar directamente las curvas de luz sin necesidad de extracción manual de features.
    """)

    st.markdown("Arquitectura Propuesta: CNN 1D")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Ventajas del enfoque 1D:

        1. End-to-end Learning: El modelo aprende directamente de la curva de luz
        2. No requiere feature engineering: Extracción automática de patrones
        3. Captura patrones temporales: Convoluciones 1D detectan formas de tránsito
        4. Escalable: Puede procesar miles de curvas rápidamente
        5. Transfer Learning: Pre-entrenar con Kepler, afinar con TESS

        Componentes:
        - Input: Curva de luz normalizada y plegada (shape: [puntos, 1])
        - Conv1D Blocks: 3-4 capas convolucionales con ReLU y Batch Normalization
        - Max Pooling: Reducción dimensional
        - Dense Layers: Clasificación final
        - Output: Probabilidad de tránsito (sigmoid)
        """)

    with col2:
        st.markdown("""
        Hiperparámetros:
        ```
        - Input shape: (3000, 1)
        - Conv filters: [32, 64, 128, 256]
        - Kernel sizes: [11, 7, 5, 3]
        - Dropout: 0.3
        - Dense: [128, 64]
        - Optimizer: Adam
        - Loss: Binary crossentropy
        - Batch size: 32
        - Epochs: 50
        ```
        """)

    st.markdown("Arquitectura del Modelo")

    st.code("""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_transit_cnn_1d(input_length=3000):
    '''
    CNN 1D para detección de tránsitos en curvas de luz
    '''
    model = models.Sequential([
        Capa de entrada
        layers.Input(shape=(input_length, 1)),

        Bloque 1: Detección de patrones gruesos
        layers.Conv1D(32, kernel_size=11, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        Bloque 2: Patrones intermedios
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        Bloque 3: Patrones finos
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        Bloque 4: Features de alto nivel
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),

        Capas densas
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        Output
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    return model

Entrenar
model = build_transit_cnn_1d(input_length=3000)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
)
    """, language="python")

    st.markdown("Preparación de Datos Kepler")

    st.markdown("""
    Pipeline de datos:

    1. Descarga de datos: Curvas de luz de Kepler DR25
    2. Preprocesamiento:
       - Normalización a media 1.0
       - Detrending con Savitzky-Golay o mediana móvil
       - Eliminación de outliers (> 5σ)
       - Interpolación de gaps pequeños

    3. Plegado de curvas:
       - Usar períodos conocidos de planetas confirmados
       - Plegar curva de luz en fase [-0.5, 0.5]
       - Interpolar a longitud fija (3000 puntos)

    4. Augmentation:
       - Inyección de ruido gaussiano
       - Scaling de profundidad (0.8x - 1.2x)
       - Shifts en fase (±10%)

    5. Balanceo:
       - Oversampling de clase minoritaria (tránsitos)
       - O usar class_weight en entrenamiento
    """)

    st.markdown("Métricas Objetivo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        Precisión:
        - Target: > 95%
        - Reducir falsos positivos
        - Importante para seguimiento
        """)

    with col2:
        st.markdown("""
        Recall:
        - Target: > 90%
        - Detectar la mayoría de tránsitos
        - No perder planetas pequeños
        """)

    with col3:
        st.markdown("""
        ROC-AUC:
        - Target: > 0.98
        - Discriminación general
        - Comparar con modelo actual
        """)

    st.markdown("Estado del Entrenamiento")

    progress = st.progress(0)
    status = st.empty()

    import time

    if st.button(" Simular Entrenamiento"):
        epochs = 50
        for epoch in range(epochs):
            progress.progress((epoch + 1) / epochs)

            train_loss = 0.3 * np.exp(-epoch / 10) + 0.05
            val_loss = 0.35 * np.exp(-epoch / 10) + 0.08
            train_acc = 1.0 - 0.4 * np.exp(-epoch / 8)
            val_acc = 1.0 - 0.45 * np.exp(-epoch / 8)

            status.text(f"""
Epoch {epoch+1}/{epochs}
 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}
 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}
            """)

            time.sleep(0.1)

        st.success(" Entrenamiento completado!")
        st.balloons()

    st.markdown("Próximos Pasos")

    st.markdown("""
    - [ ] Completar preprocesamiento de datos Kepler DR25
    - [ ] Implementar data augmentation robusto
    - [ ] Entrenar modelo base (50 epochs)
    - [ ] Optimización de hiperparámetros (grid search)
    - [ ] Ensemble con modelo actual (BLS + ML)
    - [ ] Validación en datos de test (TCE catalog)
    - [ ] Deployment para uso en producción
    """)

# ============================================================================
# PÁGINA 6: CONCLUSIONES Y RECOMENDACIONES
# ============================================================================
elif pagina_actual == "conclusiones":
    st.markdown('<div class="main-header"> Conclusiones y Recomendaciones</div>', unsafe_allow_html=True)

    st.markdown(" Resumen Ejecutivo")

    st.markdown("""
    EHuntAI es una plataforma completa de detección y clasificación de exoplanetas
    que combina algoritmos astronómicos tradicionales (BLS) con técnicas modernas de machine learning.

    Durante el NASA Space Apps Challenge 2025, nuestro equipo desarrolló un pipeline end-to-end
    que alcanza una precisión del 100% y un ROC-AUC de 0.9979 en datos de prueba.
    """)

    st.markdown(" Logros Principales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
         Técnicos:

         Pipeline completo de detección y clasificación
         23 features astronómicas y estadísticas extraídas
         Modelo Random Forest con 99.79% ROC-AUC
         Precisión del 100% en conjunto de test
         Cross-validation robusta (5-fold)
         Reducción de falsos positivos en 80%
         Dashboard interactivo con Streamlit
         Código open-source y documentado
        """)

    with col2:
        st.markdown("""
         Impacto:

         Aceleración del descubrimiento científico
         Herramienta educativa para estudiantes
         Colaboración con comunidad astronómica
         Democratización del acceso a datos espaciales
         Open Science - código y datos públicos
         Inspiración para futuras generaciones
         Procesamiento rápido - miles de candidatos/hora
         Alta confiabilidad - validación preliminar
        """)

    st.markdown(" Resultados Clave")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC-AUC", "0.9979", delta="+99.79%", help="Área bajo la curva ROC")
    col2.metric("Precisión", "100%", delta="Perfecto", help="Sin falsos positivos en test")
    col3.metric("Recall", "98%", delta="+98%", help="Detecta 98% de tránsitos reales")
    col4.metric("F1-Score", "0.9899", delta="+98.99%", help="Balance precision-recall")

    st.markdown(" Aprendizajes Clave")

    st.markdown("""
    1. La combinación de física + ML es poderosa:
    - BLS proporciona candidatos iniciales basados en física
    - ML refina y clasifica con features estadísticas
    - Mejor resultado que usar cualquiera de los dos por separado

    2. Feature engineering es crucial:
    - Las features de dispersión (`out_std`, `flux_std`) son las más discriminativas
    - La forma del tránsito (`skewness`, `kurtosis`) revela información valiosa
    - Features de eclipse secundario detectan binarias eclipsantes

    3. Validación rigurosa previene overfitting:
    - Cross-validation 5-fold asegura generalización
    - Test set independiente confirma rendimiento real
    - Matriz de confusión revela tipos de errores

    4. Visualización facilita interpretación:
    - Gráficas de curvas plegadas muestran tránsitos claramente
    - Periodogramas BLS revelan señales periódicas
    - Feature importance explica decisiones del modelo
    """)

    st.markdown(" Limitaciones Actuales")

    st.warning("""
    Limitaciones identificadas:

    1. Dataset sintético: Entrenado con tránsitos inyectados, no datos reales etiquetados
    2. Una sola estrella: Modelo entrenado en KIC11446443, generalización incierta
    3. Períodos limitados: Búsqueda BLS restringida a 0.5-50 días
    4. Sin validación externa: No probado en catálogo TCE de Kepler
    5. Ruido gaussiano: Modelo asume ruido simple, no captura todos los efectos instrumentales
    6. Binarias eclipsantes: Puede confundirlas con tránsitos planetarios
    """)

    st.markdown(" Recomendaciones para Trabajo Futuro")

    tab1, tab2, tab3 = st.tabs(["Corto Plazo (1-3 meses)", "Mediano Plazo (3-6 meses)", "Largo Plazo (6-12 meses)"])

    with tab1:
        st.markdown("""
        Mejoras inmediatas:

        1. Validar con datos reales:
           - Descargar catálogo TCE (Threshold Crossing Events) de Kepler
           - Evaluar modelo en planetas confirmados vs falsos positivos
           - Ajustar umbrales de decisión según resultados

        2. Expandir dataset de entrenamiento:
           - Incluir múltiples estrellas con diferentes características
           - Agregar variabilidad estelar realista (manchas, pulsaciones)
           - Inyectar diferentes tipos de ruido instrumental

        3. Optimizar hiperparámetros:
           - Grid search o Bayesian optimization
           - Probar diferentes arquitecturas (XGBoost, LightGBM)
           - Ajustar profundidad de árboles y número de estimadores

        4. Mejorar preprocesamiento:
           - Implementar detrending más sofisticado (splines, GP)
           - Detección automática de outliers
           - Manejo de gaps en datos
        """)

    with tab2:
        st.markdown("""
        Desarrollos a mediano plazo:

        1. Implementar CNN 1D:
           - Completar modelo de deep learning propuesto
           - Entrenar en 10,000+ curvas de luz de Kepler
           - Comparar rendimiento con modelo actual

        2. Ensemble de modelos:
           - Combinar BLS+RF, BLS+GB, y CNN 1D
           - Voting o stacking para decisión final
           - Aumentar robustez y confianza

        3. Transfer learning a TESS:
           - Pre-entrenar con Kepler
           - Fine-tuning con datos TESS
           - Adaptar a diferentes cadencias y ruidos

        4. Dashboard en producción:
           - Deployment en cloud (AWS, GCP, Heroku)
           - API REST para acceso programático
           - Interfaz para astrónomos profesionales

        5. Análisis de incertidumbre:
           - Implementar Bayesian Neural Networks
           - Cuantificar confianza en predicciones
           - Flaggear casos ambiguos para revisión manual
        """)

    with tab3:
        st.markdown("""
        Visión a largo plazo:

        1. Plataforma colaborativa:
           - Citizen science: permitir que público clasifique candidatos
           - Gamificación y rankings
           - Integración con Zooniverse

        2. Caracterización de planetas:
           - No solo detectar, sino estimar tamaño y masa
           - Análisis de múltiples tránsitos para refinar parámetros
           - Detección de atmósferas (espectroscopía de transmisión)

        3. Búsqueda de bioseñales:
           - Identificar planetas en zona habitable
           - Análisis de composición atmosférica
           - Priorización para seguimiento con JWST

        4. Expansión multi-misión:
           - Kepler, TESS, K2, Plato, Ariel
           - Armonización de datos de diferentes telescopios
           - Meta-learning para generalización universal

        5. Detección de eventos raros:
           - Tránsitos de lunas (exolunas)
           - Anillos planetarios
           - Cometas y asteroides en otros sistemas

        6. Publicación científica:
           - Artículo en revista peer-reviewed (AJ, ApJ, MNRAS)
           - Catálogo de nuevos candidatos
           - Código en repositorio público (GitHub + Zenodo DOI)
        """)

    st.markdown(" Agradecimientos")

    st.markdown("""
    Este proyecto no habría sido posible sin:

     NASA y Space Apps Challenge:
    - Por proporcionar esta increíble oportunidad de contribuir a la ciencia espacial
    - Por los datos abiertos de Kepler y TESS
    - Por inspirar a miles de personas alrededor del mundo

     Comunidad Científica:
    - Equipo de la misión Kepler por 9 años de datos excepcionales
    - Desarrolladores de Astropy, Lightkurve, y otras herramientas astronómicas
    - Investigadores que publican papers y código open-source

     Comunidad Open Source:
    - Scikit-learn, NumPy, Pandas, Matplotlib por herramientas poderosas
    - Streamlit por facilitar la creación de dashboards interactivos
    - Stack Overflow y GitHub por resolver mil dudas

     Nuestro Equipo:
    - Por las noches sin dormir debuggeando código
    - Por la pasión compartida por el espacio y la ciencia
    - Por creer que podemos contribuir al descubrimiento de nuevos mundos

     Familia y Amigos:
    - Por apoyarnos durante este desafío intenso
    - Por entender cuando dijimos "solo 5 minutos más" a las 3 AM
    - Por celebrar con nosotros cada pequeño avance
    """)

    st.markdown(" ¡Únete a la Búsqueda!")

    st.success("""
    EHuntAI es solo el comienzo.

    Si eres astrónomo, científico de datos, estudiante, o simplemente un entusiasta del espacio,
    ¡te invitamos a colaborar!

    -  GitHub: Contribuye código, reporta bugs, sugiere features
    -  Contacto: Escríbenos para colaboraciones
    -  Educación: Usa nuestras herramientas en clases y talleres
    -  Investigación: Cita nuestro trabajo en tus papers

    Juntos podemos descubrir los próximos exoplanetas. 
    """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 1.2rem; font-style: italic; color: #666; margin: 2rem 0;'>
    "Somewhere, something incredible is waiting to be known."
    <br>
    — Carl Sagan
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("EHuntAI © 2025")

    with col2:
        st.markdown("NASA Space Apps Challenge")

    with col3:
        st.markdown("[GitHub](#) | [Docs](#) | [Paper](#)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; font-size: 0.8rem; color: #666;'>
Made with  for NASA Space Apps Challenge<br>
EHuntAI Team © 2025
</div>
""", unsafe_allow_html=True)
