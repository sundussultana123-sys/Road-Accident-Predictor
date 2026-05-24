import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Road Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Road Accident Severity Predictor")
st.markdown("""
This app predicts the severity of road accidents based on various factors like weather conditions, 
road conditions, time of day, and vehicle type.
""")

# Initialize session state for storing predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Create synthetic training data (in production, you'd use real data)
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic data
    data = {
        'weather': np.random.choice(['Clear', 'Rainy', 'Foggy', 'Snowy'], n_samples),
        'road_condition': np.random.choice(['Dry', 'Wet', 'Icy', 'Under Construction'], n_samples),
        'light_condition': np.random.choice(['Daylight', 'Dusk', 'Night', 'Dawn'], n_samples),
        'vehicle_type': np.random.choice(['Car', 'Motorcycle', 'Truck', 'Bus'], n_samples),
        'speed_limit': np.random.choice([30, 40, 50, 60, 70, 80], n_samples),
        'num_vehicles': np.random.randint(1, 5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (severity) based on logical rules
    severity_score = (
        (df['weather'] == 'Snowy').astype(int) * 2 +
        (df['weather'] == 'Foggy').astype(int) * 1.5 +
        (df['road_condition'] == 'Icy').astype(int) * 2 +
        (df['light_condition'] == 'Night').astype(int) * 1.5 +
        (df['vehicle_type'] == 'Motorcycle').astype(int) * 1.5 +
        (df['speed_limit'] / 40) +
        (df['num_vehicles'] * 0.5) +
        np.random.random(n_samples) * 2
    )
    
    df['severity'] = pd.cut(severity_score, bins=3, labels=['Minor', 'Moderate', 'Severe'])
    
    # Encode categorical variables
    encoders = {}
    for col in ['weather', 'road_condition', 'light_condition', 'vehicle_type']:
        encoders[col] = LabelEncoder()
        df[col + '_encoded'] = encoders[col].fit_transform(df[col])
    
    # Prepare features
    feature_cols = [col + '_encoded' for col in ['weather', 'road_condition', 'light_condition', 'vehicle_type']] + \
                   ['speed_limit', 'num_vehicles', 'hour', 'day_of_week']
    
    X = df[feature_cols]
    y = df['severity']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model, encoders

# Train the model
with st.spinner('Loading prediction model...'):
    model, encoders = train_model()

# Sidebar for input features
st.sidebar.header("Input Accident Conditions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Environmental Conditions")
    weather = st.selectbox("Weather Condition", ['Clear', 'Rainy', 'Foggy', 'Snowy'])
    road_condition = st.selectbox("Road Condition", ['Dry', 'Wet', 'Icy', 'Under Construction'])
    light_condition = st.selectbox("Light Condition", ['Daylight', 'Dusk', 'Night', 'Dawn'])

with col2:
    st.subheader("Accident Details")
    vehicle_type = st.selectbox("Primary Vehicle Type", ['Car', 'Motorcycle', 'Truck', 'Bus'])
    speed_limit = st.slider("Speed Limit (km/h)", 30, 120, 50, 10)
    num_vehicles = st.number_input("Number of Vehicles Involved", 1, 10, 2)

col3, col4 = st.columns(2)

with col3:
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)

with col4:
    day_of_week = st.selectbox("Day of Week", 
                               ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    day_encoded = day_mapping[day_of_week]

# Predict button
if st.button("🔮 Predict Accident Severity", type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'weather_encoded': [encoders['weather'].transform([weather])[0]],
        'road_condition_encoded': [encoders['road_condition'].transform([road_condition])[0]],
        'light_condition_encoded': [encoders['light_condition'].transform([light_condition])[0]],
        'vehicle_type_encoded': [encoders['vehicle_type'].transform([vehicle_type])[0]],
        'speed_limit': [speed_limit],
        'num_vehicles': [num_vehicles],
        'hour': [hour],
        'day_of_week': [day_encoded]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Store in history
    st.session_state.prediction_history.append({
        'weather': weather,
        'road': road_condition,
        'light': light_condition,
        'vehicle': vehicle_type,
        'prediction': prediction
    })
    
    # Display results
    st.markdown("---")
    st.header("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    severity_colors = {'Minor': '#2ecc71', 'Moderate': '#f39c12', 'Severe': '#e74c3c'}
    
    with col1:
        st.metric("Predicted Severity", prediction, 
                  delta=None)
        st.markdown(f"<div style='background-color: {severity_colors[prediction]}; padding: 20px; border-radius: 10px; text-align: center; color: white; font-size: 24px; font-weight: bold;'>{prediction}</div>", 
                    unsafe_allow_html=True)
    
    with col2:
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Severity': model.classes_,
            'Probability': prediction_proba * 100
        })
        fig = px.bar(prob_df, x='Severity', y='Probability', 
                     color='Severity',
                     color_discrete_map=severity_colors,
                     text='Probability')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Risk Factors")
        risk_factors = []
        if weather in ['Snowy', 'Foggy']:
            risk_factors.append(f"⚠️ Hazardous weather: {weather}")
        if road_condition in ['Icy', 'Under Construction']:
            risk_factors.append(f"⚠️ Poor road: {road_condition}")
        if light_condition in ['Night', 'Dusk']:
            risk_factors.append(f"⚠️ Low visibility: {light_condition}")
        if speed_limit > 70:
            risk_factors.append(f"⚠️ High speed limit: {speed_limit} km/h")
        if num_vehicles > 2:
            risk_factors.append(f"⚠️ Multiple vehicles: {num_vehicles}")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ Low risk conditions detected")

# Prediction history
if st.session_state.prediction_history:
    st.markdown("---")
    st.header("Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(history_df, use_container_width=True)
    
    with col2:
        severity_counts = history_df['prediction'].value_counts()
        fig = px.pie(values=severity_counts.values, 
                     names=severity_counts.index,
                     color=severity_counts.index,
                     color_discrete_map=severity_colors,
                     title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ This is a demonstration app. Always follow traffic rules and drive safely.</p>
    <p>Built with Streamlit | Machine Learning Powered</p>
</div>
""", unsafe_allow_html=True)
