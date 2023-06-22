import streamlit as st
import pandas as pd


from get_explanations import get_shap_values_classification, get_shap_values_regression
from understand_explanations_with_llm import get_explanation_from_genAI
st.title('Explain with GenAI :brain:')
st.text('This app uses Shapley values to explain impact of each feature on xgboost model predictions. All you need is your model and some data')
with st.form("my_form"):
    uploaded_file_1 = st.file_uploader("Upload the data file here with column names in header:point_down:", type="CSV")

    if uploaded_file_1 is not None:
        dataframe = pd.read_csv(uploaded_file_1)
        st.write(dataframe)
    model_name = st.text_input("Type the model URL here :point_down:") #xgb model can be read via a file path or URL
    if model_name is not None:
        model = model_name
        
    problem = st.selectbox(
    'Select problem type:',
    ('Classification', 'Regression'))
    st.write('You selected:', problem)
    # user_input = st.text_input("Columns to use", placeholder="Input a list of columns to use", key='input')
    submit_button = st.form_submit_button(label='Get feature impact')
    if  submit_button:
        if "Classification" in problem:
            shap_result, pred= get_shap_values_classification(dataframe,model)
        else:
            shap_result, pred= get_shap_values_regression(dataframe,model)

        result = get_explanation_from_genAI(shap_result, pred, problem)
        # result = result.split("Prediction")[1]
        st.write(f":point_right: {result} :point_left:")
