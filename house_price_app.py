# house_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", layout="wide")

# -------------------------------
# Load model & scaler (once)
# -------------------------------
@st.cache_resource
def load_artifacts(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

MODEL_PATH = r"C:\Users\Wahid\Downloads\best_house_price_model.pkl"
SCALER_PATH = r"C:\Users\Wahid\Downloads\scaler.pkl"

try:
    model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
except Exception as e:
    st.error("Model / scaler load failed. Check the file paths.")
    st.exception(e)
    st.stop()

# -------------------------------
# App Title + instructions
# -------------------------------
st.title("🏠 House Price Prediction")
st.markdown("Fill the form and click **Predict Price**. No prediction will be shown until you press the button.")

# -------------------------------
# Inputs layout
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    area = st.slider("Area (sqft):", min_value=100, max_value=50000, value=1000, step=50)
    bedrooms = st.slider("Bedrooms:", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.slider("Bathrooms:", min_value=1, max_value=10, value=2, step=1)
    stories = st.slider("Stories:", min_value=1, max_value=5, value=2, step=1)

with col2:
    mainroad = st.radio("Main Road?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    guestroom = st.radio("Guestroom?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    basement = st.radio("Basement?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    hotwaterheating = st.radio("Hot Water Heating?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")

with col3:
    airconditioning = st.radio("Air Conditioning?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    parking = st.slider("Parking Spaces:", min_value=0, max_value=5, value=0, step=1)
    prefarea = st.radio("Preferred Area?", options=[1,0], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    furnishingstatus = st.selectbox("Furnishing Status:", options=[0,1,2], index=0,
                                    format_func=lambda x: ["Unfurnished","Semi-Furnished","Furnished"][x])

st.markdown("---")

# -------------------------------
# Button: only run prediction when clicked
# -------------------------------
if st.button("Predict Price"):
    # Feature engineering (same as training)
    area_per_room = area / max(1, (bedrooms + bathrooms))
    bath_bed_ratio = bathrooms / max(1, bedrooms)
    total_rooms = bedrooms + bathrooms
    price_per_sqft = 0
    bathroom_per_sqft = bathrooms / max(1, area)
    bedroom_per_sqft = bedrooms / max(1, area)
    total_rooms_sqft_ratio = total_rooms / max(1, area)

    # Build input DataFrame - ensure column order matches what scaler/model expect
    input_cols = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement',
                  'hotwaterheating','airconditioning','parking','prefarea','furnishingstatus',
                  'area_per_room','bath_bed_ratio','total_rooms','price_per_sqft',
                  'bathroom_per_sqft','bedroom_per_sqft','total_rooms_sqft_ratio']

    input_values = [[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                     basement, hotwaterheating, airconditioning, parking,
                     prefarea, furnishingstatus, area_per_room, bath_bed_ratio,
                     total_rooms, price_per_sqft, bathroom_per_sqft,
                     bedroom_per_sqft, total_rooms_sqft_ratio]]

    input_df = pd.DataFrame(input_values, columns=input_cols)

    # If your model was trained with a 'log_price' column present,
    # keep a placeholder column with the same name (matching training features).
    # If not needed, you can remove the next line.
    input_df['log_price'] = 0

    # Safe NaN handling
    input_df.fillna(0, inplace=True)

    # Debug option: uncomment to print input columns and values to help debug constant-output issues
    # st.write("DEBUG - input_df columns:", list(input_df.columns))
    # st.write("DEBUG - input_df values:", input_df.iloc[0].to_list())

    # Predict with spinner + error handling
    with st.spinner("Scaling input & predicting..."):
        try:
            input_scaled = scaler.transform(input_df)   # may raise if shape mismatch
            predicted_price = model.predict(input_scaled)[0]
            st.subheader("💰 Predicted House Price")
            st.success(f"PKR {predicted_price:,.0f}")
        except Exception as e:
            st.error("Prediction failed. Possible causes: feature-order mismatch between your app and the saved scaler/model.")
            st.exception(e)
            # Optional: show shapes for debugging
            try:
                st.write("Input shape:", input_df.shape)
                # st.write("Scaler expected features:", getattr(scaler, 'n_features_in_', 'unknown'))
            except:
                pass
else:
    st.info("Change input values and click **Predict Price** to see prediction.")
