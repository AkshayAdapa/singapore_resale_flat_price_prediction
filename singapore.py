import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuring the page for Streamlit application
st.set_page_config(
    layout="wide"
)

# Load the pre-trained model
with open("RandomForestRegressor_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

# Load the numerical encoded DataFrame containing the features
df1 = pd.read_csv("encoded_dataset.csv")

# Display image at the top
st.markdown('<div style="display: flex; justify-content: center; align-items: center;"><img src="https://thumbs.dreamstime.com/b/singapore-city-8731712.jpg" width="700"></div><p style="text-align: center; margin: 0;">Singapore City</p>', unsafe_allow_html=True)


# Title
st.markdown("<h1 style='text-align: center; color: #ff6600;'>Singapore Resale Flat Prices Prediction</h1>", unsafe_allow_html=True)

# Navigation buttons
selected = st.radio("Navigation", ["Home", "Predictions"])

# Home tab content
if selected == "Home":
    st.markdown("## Overview")
    st.markdown("Singapore's real estate market is dynamic, with resale flat prices influenced by various factors such as location, size, and lease duration. The 'Singapore Resale Flat Prices Prediction' project aims to leverage machine learning to provide accurate predictions of resale flat prices, aiding both buyers and sellers in making informed decisions.")
    st.markdown("### Approach")
    st.markdown("1. **Data Collection**: Gathered data on past transactions of resale flats in Singapore.")
    st.markdown("2. **Data Preprocessing**: Cleaned and prepared the data for model training.")
    st.markdown("3. **Feature Engineering**: Extracted relevant features such as town, flat type, street name, etc.")
    st.markdown("4. **Model Selection and Training**: Chose a suitable machine learning model and trained it using the prepared data.")
    st.markdown("5. **Model Evaluation**: Evaluated the model's performance using appropriate metrics.")
    st.markdown("6. **Model Deployment**: Deployed the trained model as a user-friendly online application using Streamlit.")
    st.markdown("### Skills Utilized")
    st.markdown("- Python")
    st.markdown("- Pandas")
    st.markdown("- NumPy")
    st.markdown("- Scikit-learn")
    st.markdown("- Streamlit")
    st.markdown("- Machine Learning")
    st.markdown("- Data Preprocessing")
    st.markdown("- Data Visualization")
    st.markdown("### Conclusion")
    st.markdown("The developed model shows promising results in predicting resale flat prices in Singapore. By providing accurate predictions, this tool can be valuable for both buyers and sellers in making informed decisions. However, continuous improvement and updating of the model will be necessary to adapt to changing market trends and dynamics.")

# Predictions tab content
elif selected == "Predictions":
    st.markdown("### Predicting Resale Price (Regression Task) (Accuracy: 97%)")

    # Dropdown options for all features
    town_options = {
        'ANG MO KIO': 1,
        'BEDOK': 2,
        'BISHAN': 3,
        'BUKIT BATOK': 4,
        'BUKIT MERAH': 5,
        'BUKIT TIMAH': 6,
        'CENTRAL AREA': 7,
        'CHOA CHU KANG': 8,
        'CLEMENTI': 9,
        'GEYLANG': 10,
        'HOUGANG': 11,
        'JURONG EAST': 12,
        'JURONG WEST': 13,
        'KALLANG/WHAMPOA': 14,
        'MARINE PARADE': 15,
        'QUEENSTOWN': 16,
        'SENGKANG': 17,
        'SERANGOON': 18,
        'TAMPINES': 19,
        'TOA PAYOH': 20,
        'WOODLANDS': 21,
        'YISHUN': 22,
        'LIM CHU KANG': 23,
        'SEMBAWANG': 24,
        'BUKIT PANJANG': 25,
        'PASIR RIS': 26,
        'PUNGGOL': 27
    }
    flat_type_options = {
        '1 ROOM': 1,
        '2 ROOM': 2,
        '3 ROOM': 3,
        '4.ROOM': 4,
        '5.ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI GENERATION': 7
    }
    street_name_options = {
        'ANG MO KIO AVE 4': 1,
        'ANG MO KIO AVE 8': 2,
        'ANG MO KIO AVE 10': 3,
        'ANG MO KIO AVE 5': 4,
        'ANG MO KIO AVE 3': 5,
        'ANG MO KIO AVE 1': 6,
        'ANG MO KIO AVE 9': 7,
        'ANG MO KIO AVE 6': 8,
        'ANG MO KIO ST 32': 9,
        'ANG_MO_KIO_ST_52': 10,
        'ANG_MO_KIO_ST_21': 11,
        'ANG_MO_KIO_ST_31': 12,
        'BEDOK_RESERVOIR_RD': 13,
        'BEDOK_NTH_ST_3': 14,
        'BEDOK_NTH_RD': 15,
        'CHAI_CHEE_RD': 16,
        'BEDOK_STH_AVE_1': 17,
        'BEDOK_NTH_ST_4': 18,
        'BEDOK_NTH_ST_2': 19,
        'BEDOK_STH_RD': 20,
        'CHAI_CHEE_DR': 21,
        'CHAI_CHEE_AVE': 22,
        'BEDOK_NTH_AVE_1': 23,
        'BEDOK_STH_AVE_3': 24,
        'NEW_UPP_CHANGI_RD': 25,
        'BT_BATOK_ST_31': 26,
        'BT_BATOK_EAST_AVE_6': 27,
        'BT_BATOK_WEST_AVE_9': 28,
        'MARGARET_DR': 29,
        'TAMPINES_ST_61': 30,
        'YISHUN_ST_43': 31
    }
    flat_model_options = {
        'IMPROVED': 1,
        'NEW_GENERATION': 2,
        'MODEL_A': 3,
        'STANDARD': 4,
        'SIMPLIFIED': 5,
        'MODEL_A-MAISONETTE': 6,
        'APARTMENT': 7,
        'MAISONETTE': 8,
        'TERRACE': 9,
        '2-ROOM': 10,
        'IMPROVED-MAISONETTE': 11,
        'MULTI_GENERATION': 12,
        'PREMIUM_APARTMENT': 13,
        'Improved': 14,
        'New Generation': 15,
        'Model A': 16,
        'Standard': 17,
        'Apartment': 18,
        'Simplified': 19,
        'Model A-Maisonette': 20,
        'Maisonette': 21,
        'Multi Generation': 22,
        'Adjoineed flat': 23,
        'Premium Apartment': 24,
        'Terrace': 25,
        'Improved Maisonette': 26,
        'Premium Maisonette': 27,
        '2-room': 28,
        'Model A2': 29,
        'DBSS': 30,
        'Type S1': 31,
        'Type S2': 32,
        'Premium Apartment Loft': 33,
        '3Gen': 34
    }
    block_options = df1["block"].unique()
    floor_area_sqm_options = np.arange(1.0, 501.0)
    lease_commence_date_options = np.arange(1900, 2024)
    remaining_lease_options = df1["remaining_lease"].unique()
    resale_year_options = np.arange(1900, 2024)
    resale_month_options = np.arange(1, 13)
    storey_lower_bound_options = df1["storey_lower_bound"].unique()
    storey_upper_bound_options = df1["storey_upper_bound"].unique()

    # User inputs as dropdown menus
    town = st.selectbox("Town", list(town_options.keys()))
    flat_type = st.selectbox("Flat Type", list(flat_type_options.keys()))
    street_name = st.selectbox("Street Name", list(street_name_options.keys()))
    flat_model = st.selectbox("Flat Model", list(flat_model_options.keys()))
    block = st.selectbox("Block", block_options)
    floor_area_sqm = st.selectbox("Floor Area (Per Square Meter)", floor_area_sqm_options)
    lease_commence_date = st.selectbox("Lease Commence Date", lease_commence_date_options)
    remaining_lease = st.selectbox("Remaining Lease", remaining_lease_options)
    resale_year = st.selectbox("Resale Year", resale_year_options)
    resale_month = st.selectbox("Resale Month", resale_month_options)
    storey_lower_bound = st.selectbox("Storey Lower Bound", storey_lower_bound_options)
    storey_upper_bound = st.selectbox("Storey Upper Bound", storey_upper_bound_options)

    # Submit Button for predicting resale price
    submit_button = st.button("PREDICT RESALE PRICE")

    if submit_button:
        # Combine user inputs into a feature array
        features = np.array([town_options[town], flat_type_options[flat_type], street_name_options[street_name],
                             flat_model_options[flat_model], block,
                             floor_area_sqm, lease_commence_date, remaining_lease,
                             resale_year, resale_month, storey_lower_bound, storey_upper_bound])

        # Predict resale price using the loaded model
        new_pred = loaded_model.predict([features])[0]
        
        # Display the predicted resale price
        st.write('## Predicted resale price:', new_pred)
