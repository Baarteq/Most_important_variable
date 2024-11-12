import streamlit as st
import pandas as pd
from pycaret.classification import setup as setup_klas, compare_models as compare_models_klas, finalize_model as finalize_model_klas, plot_model as plot_model_klas, save_model as save_model_klas, load_model as load_model_klas, predict_model as predict_model_klas
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, finalize_model as finalize_model_reg, plot_model as plot_model_reg, save_model as save_model_reg, load_model as load_model_reg, predict_model as predict_model_reg
from dotenv import dotenv_values
from openai import OpenAI
import base64

env = dotenv_values(".env")

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env['OPENAI_API_KEY']
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

openai_client = get_openai_client()

def prepare_image_for_open_ai(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:image/png;base64,{image_data}"

def classify_data(df, predicted_column_name="kategoria"):
    model_klasyfikacja= load_model_klas("model_klasyfikujacy_pipeline")
    df= predict_model_klas(model_klasyfikacja, data=df)
    df= df.rename(columns={'prediction_label': predicted_column_name})
    return df

def regres_data(df, predicted_column_name="kategoria"):
    model_regresja= load_model_reg("model_regresji_pipeline")
    df= predict_model_klas(model_regresja, data=df)
    df= df.rename(columns={'prediction_label': predicted_column_name})
    return df

def has_less_than_10_unique_numbers(tabela, kolumna):
    # Sprawdzamy unikalne wartości w kolumnie docelowej
    unique_values = tabela[kolumna].unique()
    # Sprawdzamy, czy liczba unikalnych wartości w kolumnie docelowej jest większa niż 10
    return len(unique_values) < 10

def describe_image(image_path):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '''Jesteś Data Scientist, który musi w prosty i przystępny sposób podsumować i opisać wykres z obrazka dla osoby, która nie umie analizować wykresów.
                          '''
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_open_ai(image_path),
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


#MAIN

st.set_page_config(page_title="Najbardziej wartościowa zmienna", layout="wide")
st.title("Najbardziej wartościowa zmienna")

#v1 - możliwość wczytania pliku CSV i wyświetlenie przykładowych danych

with st.sidebar:
    Plik = st.file_uploader("Wybierz plik CSV z danymi do analizy", type=['csv'])
    if Plik is not None:
        separator= st.selectbox("Podaj typ separatora użytego w pliku CSV", [";", ",", 'tab', 'spacja'])
        Data_Frame= pd.read_csv(Plik, sep=separator)


if Plik is not None:
    tabela = Data_Frame
    st.write("Losowe wiersze z pliku")
    x=min(10, len(tabela))
    st.dataframe(tabela.sample(x),hide_index=True)

#v2 - Wybór kolumny docelowej

    kolumna = st.selectbox("Wybierz kolumnę docelową", tabela.columns)
    tabela= tabela.dropna(how='all')
    result_less_10_values = has_less_than_10_unique_numbers(tabela, kolumna)
    if st.button("Analizuj dane"):
        if result_less_10_values:
            with st.spinner("Analizuję dane. Czekaj...."):
                    setup_klas(data=tabela, target=kolumna, session_id=123)
                    best_model_klasyfikacja = compare_models_klas()
                    final_model_klas = finalize_model_klas(best_model_klasyfikacja)
                    save_model_klas(final_model_klas, "model_klasyfikujacy_pipeline")
                    st.success("Gotowe!")
                    wynik = classify_data(tabela)
                    st.dataframe(wynik,hide_index=True)
                    plot_model_klas(best_model_klasyfikacja, plot='feature', save=True)
                    plot_name = 'Feature Importance.png'
                    plot_image = st.image(plot_name, use_column_width=False)
        else:
            with st.spinner("Analizuję dane. Czekaj...."):
                    setup_reg(data=tabela, target=kolumna, session_id=124)
                    best_model_regresja = compare_models_reg()
                    final_model_reg = finalize_model_reg(best_model_regresja)
                    save_model_reg(final_model_reg, "model_regresji_pipeline")
                    wynik = regres_data(tabela)
                    st.dataframe(wynik,hide_index=True)
                    plot_model_reg(best_model_regresja, plot='feature', save=True)
                    plot_name = 'Feature Importance.png'
                    plot_image = st.image(plot_name, use_column_width=False)
    
        if plot_image:
            with st.spinner("Genreuję opis wykresu. Czekaj..."):
                opis_wykresu= describe_image("Feature Importance.png")
                st.write(opis_wykresu)
    