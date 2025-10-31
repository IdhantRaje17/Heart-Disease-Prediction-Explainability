import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image

# --- App Configuration ---
st.set_page_config(page_title="Heart Disease Predictor & Explainer", layout="wide")

# --- Model & Explainer Loading ---
@st.cache_resource
def load_model_and_explainer():
    """Loads the saved Random Forest model, columns, and creates a SHAP explainer."""
    try:
        model = joblib.load('heart_disease_model.pkl')
        columns = joblib.load('model_columns.pkl')
        explainer = shap.Explainer(model)
        return model, columns, explainer
    except FileNotFoundError:
        st.error("Model files not found! Please run your Jupyter notebook to save the necessary .pkl files first.")
        return None, None, None

model, columns, explainer = load_model_and_explainer()

# --- Page Title and Description ---
st.title("🩺 Heart Disease Prediction & Explainability Engine")
st.markdown("This app uses a Random Forest model to predict heart disease and provide a full explanation for its decision.")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Data Input")

def get_user_input():
    """Creates sidebar widgets and returns a DataFrame of user inputs."""
    age = st.sidebar.number_input('Age', min_value=20, max_value=80, value=55, help="Patient's age in years.")
    sex_option = st.sidebar.selectbox('Sex', ('Female', 'Male'), index=1, help="Patient's biological sex.")
    sex = 1 if sex_option == 'Male' else 0

    cp_options = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
    cp_selection = st.sidebar.selectbox('Chest Pain Type (cp)', options=list(cp_options.keys()), format_func=lambda x: f"{x}: {cp_options[x]}", index=3)

    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=90, max_value=200, value=130, help="In mm Hg on admission.")
    chol = st.sidebar.number_input('Serum Cholesterol (chol)', min_value=120, max_value=570, value=240, help="In mg/dl.")
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: "True" if x == 1 else "False", index=0)
    restecg = st.sidebar.selectbox('Resting ECG Results (restecg)', (0, 1, 2), index=1)
    thalach = st.sidebar.number_input('Max Heart Rate Achieved (thalach)', min_value=70, max_value=205, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1), format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    oldpeak = st.sidebar.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=6.2, value=1.0, step=0.1, help="Induced by exercise relative to rest.")
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (slope)', (1, 2, 3), index=1)
    ca = st.sidebar.selectbox('Number of Major Vessels (ca)', (0, 1, 2, 3, 4), index=0)
    thal = st.sidebar.selectbox('Thallium Stress Test Result (thal)', (1, 2, 3), index=2)

    data = {
        'age': age, 'sex': sex, 'cp': cp_selection, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

# --- Main App Logic ---
if model is not None:
    st.header("Patient Data")
    user_input_df = get_user_input()
    user_input_df = user_input_df[columns]
    st.dataframe(user_input_df)

    if st.sidebar.button("Predict & Explain", type="primary"):
        st.markdown("---")
        st.header("Prediction & Explanations")
        
        # --- Prediction Results ---
        st.subheader("1. Model Prediction")
        prediction = model.predict(user_input_df)[0]
        probability = model.predict_proba(user_input_df)[0][1] # Probability for class 1 (Heart Disease)

        # UPDATED LOGIC FOR CONFIDENCE DISPLAY
        if prediction == 1:
            st.error(f"**Diagnosis:** Heart Disease (Confidence of Disease: {probability:.2%})")
        else:
            st.success(f"**Diagnosis:** No Heart Disease (Confidence of No Disease: {(1 - probability):.2%})")
        
        st.markdown("---")

        # --- Global Explanation Section ---
        st.subheader("2. Overall Model Importance (Global Explanation)")
        st.markdown("This plot shows the most important features *on average* for all predictions. It provides context for the individual explanation below.")
        try:
            img = Image.open('shap_summary_plot.png')
            st.image(img, caption="SHAP Global Feature Importance (Beeswarm Plot)")
        except FileNotFoundError:
            st.warning("Could not find 'shap_summary_plot.png'. Please run the notebook to generate and save it.")
        
        st.markdown("---")

        # --- Local Explanation Section ---
        st.subheader("3. This Patient's Prediction Explained (Local Explanation)")
        shap_values = explainer(user_input_df)
        
        st.write("**Force Plot:** Shows the push-and-pull of features for this prediction.")
        fig_force = shap.force_plot(explainer.expected_value[1], shap_values.values[:, :, 1], user_input_df, matplotlib=True, show=False)
        st.pyplot(fig_force, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close(fig_force)

        st.write("**Waterfall Plot:** Breaks down how each feature value moves the prediction from the average.")
        fig_waterfall, ax_waterfall = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig_waterfall, bbox_inches='tight', dpi=300, pad_inches=0.2)
        plt.close(fig_waterfall)
    else:
        st.info("Adjust patient data in the sidebar and click 'Predict & Explain'.")

