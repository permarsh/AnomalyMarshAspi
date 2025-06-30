import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import streamlit as st

# Definizione delle caratteristiche numeriche
numeric_features = [
    'Cash (Prompt Payment) Discount Days',
    'Total value at time of release',
    'Days from creation',
    'Difference from Target'
]

# Funzione per caricare e preparare i dati
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        delimiter='|',
        encoding='latin1',
        on_bad_lines='skip',
        low_memory=False,
        thousands='.',
        decimal=',',
    )
    df.columns = df.columns.str.strip()
    
    # Filtra le righe dove 'Total value at time of release' è diverso da 0
    df = df[df['Total value at time of release'] != 0]
    
    return df

# Funzione per addestrare il modello
def train_model(df):
    fields_to_analyze = [
        'Purchasing Document Number',
        'Cash (Prompt Payment) Discount Days',
        'Total value at time of release',
        'Days from creation',
        'Difference from Target'
    ]
    
    df_selected = df[fields_to_analyze]
    df_selected['Purchasing Document Number'] = df_selected['Purchasing Document Number'].astype(str)
    df_selected = df_selected.dropna()
    
    df_selected[numeric_features] = df_selected[numeric_features].apply(pd.to_numeric, errors='coerce')
    df_selected = df_selected.dropna(subset=numeric_features)

    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('isolation_forest', IsolationForest(n_estimators=300, contamination=0.008, random_state=42))
    ])

    model.fit(df_selected)

    outliers = model.predict(df_selected)
    anomaly_scores = model.named_steps['isolation_forest'].decision_function(model.named_steps['preprocessor'].transform(df_selected))

    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    normalized_scores = 1 - ((anomaly_scores - min_score) / (max_score - min_score))

    df_selected['Outlier'] = outliers
    df_selected['Anomaly_Score'] = normalized_scores
    df_selected['Anomalia'] = df_selected['Outlier'].apply(lambda x: 1 if x == -1 else 0)

    return df_selected

# Interfaccia Streamlit
st.set_page_config(page_title="Anomaly Detection", layout="wide")  # Imposta il layout della pagina

# Crea una riga per il logo e il titolo
col1, col2 = st.columns([1, 4])  # Colonna per il logo e colonna per il titolo

# Inserimento dellogo nella prima colonna
with col1:
   logo_url = "https://logowik.com/content/uploads/images/marsh9653.jpg"  # Sostituisci con l'URL reale
   st.image(logo_url, width=180)


# Inserimento del titolo nella seconda colonna
with col2:
    st.title("Marsh Advisory - Anomaly Detection")
    st.markdown("<h5 style='margin-top: -10px;'>Analisi delle anomalie nei dati</h5>", unsafe_allow_html=True)  # Sottotitolo

uploaded_file = st.file_uploader("Carica il tuo file CSV", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Dati caricati con successo!")
    st.dataframe(df.head())

    if st.button("Addestra il modello"):
        results = train_model(df)
        st.write("Modello addestrato con successo!")
        st.dataframe(results)

        # Visualizzazione del pairplot
        df_numeric = results[numeric_features].copy()
        df_numeric['Outlier'] = results['Outlier']

        # Rinomina le colonne
        df_numeric.rename(columns={
            'Total value at time of release': 'importo ordine',
            'Days from creation': 'gg di ritardo consegna'
        }, inplace=True)
        
        plt.figure(figsize=(7, 4))
        pairplot = sns.pairplot(df_numeric, hue='Outlier', palette={1: 'blue', -1: 'red'}, plot_kws={'alpha': 0.7})

        # Ruota le etichette degli assi
        for ax in pairplot.axes.flatten():
            ax.set_xlabel(ax.get_xlabel(), rotation=45)
        plt.suptitle('Pairplot con Outliers (in rosso)', y=1.02, fontsize=10)
        # Aggiungi spazio extra tra le righe
        plt.subplots_adjust(hspace=0.5)  # Aumenta lo spazio verticale tra le righe
        st.pyplot(plt)

        # Salvataggio del file CSV
        results.to_csv(r'C:\Users\u1208854\OneDrive - MMC\General\ML\DEMO\OUTPUT\anomalie_con_score.csv', index=False)
        st.success("File CSV con anomalie e punteggi salvato come 'anomalie_con_score.csv'.")
