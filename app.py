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
# Preparation of image for OpenAI
def prepare_image_for_open_ai(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:image/png;base64,{image_data}"
# Best model for classification
def classify_data(df, predicted_column_name="kategoria"):
    model_klasyfikacja= load_model_klas("model_klasyfikujacy_pipeline")
    df= predict_model_klas(model_klasyfikacja, data=df)
    df= df.rename(columns={'prediction_label': predicted_column_name})
    return df
# Best model for regresion
def regres_data(df, predicted_column_name="kategoria"):
    model_regresja= load_model_reg("model_regresji_pipeline")
    df= predict_model_klas(model_regresja, data=df)
    df= df.rename(columns={'prediction_label': predicted_column_name})
    return df
# Checking unique values ​​in target column. and check if number of unique values ​​in target column is less than 5
def has_less_than_5_unique_numbers(tabela, kolumna):
    unique_values = tabela[kolumna].unique()
    return len(unique_values) < 5

# Prompt for OpenAI image-to-text model
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

#v1 - Ability to upload a CSV file
with st.sidebar:
    Plik = st.file_uploader("Wybierz plik CSV z danymi do analizy", type=['csv'])
    if Plik is not None:
        #User defines the separator used in the CSV file
        separator= st.selectbox("Podaj typ separatora użytego w pliku CSV", [";", ",", 'tab', 'spacja'])
        tabela= pd.read_csv(Plik, sep=separator)
        #User Specifies Target Column
        kolumna = st.selectbox("Wybierz kolumnę docelową", tabela.columns)
        #User defines which columns will not be included in the analysis
        st.write("Czy chcesz usunąć z analizy jakieś kolumny?")
        columns_to_del=st.multiselect('Wybierz kolumny do usunięcia',tabela.columns.drop(kolumna))
        if columns_to_del:
            tabela= tabela.drop(columns= columns_to_del)
        tabela= tabela.dropna(how='all')
        #Preparing data for analysis - deleting rows where NaN values ​​occur in 25% of available columns, changing NaN values ​​in columns with the mode value for each column
        num_rows = tabela.shape[0]
        y=min(1, len(columns_to_del))
        num_columns = (tabela.shape[1]-(len(columns_to_del)+y))
        tabela = tabela.dropna(thresh=(tabela.shape[1]*0.75))
        columns_names= tabela.columns.tolist()
        number_of_columns = len(columns_names)
        for number_of_columns in tabela.columns:
            list_of_mode= tabela[columns_names].mode()
            list_of_mode= list_of_mode.dropna()
        for column in tabela.columns:
            tabela[column].fillna(list_of_mode[column].iloc[0], inplace=True)    
        mode_target_column = tabela[kolumna].mode()
        tabela[kolumna].fillna(mode_target_column, inplace=True)
        #Checking if we have enough data to perform the analysis
        if num_rows < 7 * num_columns:
            st.error("Za mała ilość danych do przeprowadzenia analizy")
            st.stop()

#Showing 5 sample lines
if Plik is not None:
    st.write("Losowe wiersze z pliku")
    x=min(5, len(tabela))
    st.dataframe(tabela.sample(x),hide_index=True)
    result_less_5_values = has_less_than_5_unique_numbers(tabela, kolumna)
    st.dataframe(tabela)
    if st.button("Analizuj dane"):
        # If target column has less than 5 unique values we choose classification model
        if result_less_5_values:
            with st.spinner("Analizuję dane. Czekaj...."):
                setup_klas(data=tabela, target=kolumna, session_id=123, ignore_features=columns_to_del)
                best_model_klasyfikacja = compare_models_klas()
                final_model_klas = finalize_model_klas(best_model_klasyfikacja)
                save_model_klas(final_model_klas, "model_klasyfikujacy_pipeline")
                st.success("Gotowe!")
                wynik = classify_data(tabela)
                st.dataframe(wynik,hide_index=True)
                plot_model_klas(best_model_klasyfikacja, plot='feature', save=True)
                plot_name = 'Feature Importance.png'
                plot_image = st.image(plot_name, use_column_width=False)
        # If ​​in target column has more than 5 unique values we choose regression model
        else:
            with st.spinner("Analizuję dane. Czekaj...."):
                setup_reg(data=tabela, target=kolumna, session_id=124, ignore_features=columns_to_del)
                best_model_regresja = compare_models_reg()
                final_model_reg = finalize_model_reg(best_model_regresja)
                save_model_reg(final_model_reg, "model_regresji_pipeline")
                wynik = regres_data(tabela)
                st.dataframe(wynik,hide_index=True)
                plot_model_reg(best_model_regresja, plot='feature', save=True)
                plot_name = 'Feature Importance.png'
                plot_image = st.image(plot_name, use_column_width=False)
        # Generating a description for the most important variables chart   
        if plot_image:
            with st.spinner("Genreuję opis wykresu. Czekaj..."):
                opis_wykresu= describe_image("Feature Importance.png")
                st.write(opis_wykresu)
                