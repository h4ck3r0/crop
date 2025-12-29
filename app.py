import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2. 5rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    . prediction-result {
        background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.8rem;
        margin: 2rem 0;
    }
    .stButton>button {
        background-color: #2E7D32;
        color:  white;
        font-size:  1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    . stButton>button:hover {
        background-color: #1B5E20;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st. cache_resource
def load_models():
    try:
        model, features = joblib.load('best_model.pkl')
        label_encoders = joblib. load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        median_vals = joblib.load('median_values.pkl')
        unique_vals = joblib.load('unique_values. pkl')
        agg_stats = joblib.load('agg_stats.pkl')
        yield_stats = joblib.load('yield_stats.pkl')
        return model, features, label_encoders, scaler, median_vals, unique_vals, agg_stats, yield_stats
    except Exception as e:
        st. error(f"Error loading model files: {e}")
        return None, None, None, None, None, None, None, None

# Function to prepare input data
def prepare_input_data(crop, state, season, crop_year, rainfall, fertilizer, pesticide, area, production,
                       features, label_encoders, median_vals, agg_stats):
    input_dict = {}
    
    crop_avg = agg_stats['crop_avg_yield'].get(crop, median_vals['Crop_Year'])
    state_avg = agg_stats['state_avg_yield'].get(state, median_vals['Crop_Year'])
    season_avg = agg_stats['season_avg_yield'].get(season, median_vals['Crop_Year'])
    
    for col in features:
        if col == 'Crop': 
            input_dict[col] = [label_encoders['Crop']. transform([crop])[0]]
        elif col == 'State': 
            input_dict[col] = [label_encoders['State'].transform([state])[0]]
        elif col == 'Season':
            input_dict[col] = [label_encoders['Season'].transform([season])[0]]
        elif col == 'Crop_Year':
            input_dict[col] = [crop_year]
        elif col == 'Annual_Rainfall': 
            input_dict[col] = [rainfall]
        elif col == 'Fertilizer':
            input_dict[col] = [fertilizer]
        elif col == 'Pesticide':
            input_dict[col] = [pesticide]
        elif col == 'Area':
            input_dict[col] = [area]
        elif col == 'Production':
            input_dict[col] = [production]
        elif col == 'Crop_Avg_Yield':
            input_dict[col] = [crop_avg]
        elif col == 'State_Avg_Yield':
            input_dict[col] = [state_avg]
        elif col == 'Season_Avg_Yield': 
            input_dict[col] = [season_avg]
        elif col == 'Rainfall_per_Fertilizer':
            input_dict[col] = [rainfall / (fertilizer + 1)]
        elif col == 'Area_Production_Ratio':
            input_dict[col] = [area / (production + 1)]
        elif col == 'Total_Input':
            input_dict[col] = [fertilizer + pesticide]
        elif col == 'Fertilizer_Pesticide_Ratio':
            input_dict[col] = [fertilizer / (pesticide + 1)]
    
    return pd. DataFrame(input_dict)

# Header
st.markdown('<p class="main-header">🌾 Crop Yield Prediction</p>', unsafe_allow_html=True)

# Load models
model, features, label_encoders, scaler, median_vals, unique_vals, agg_stats, yield_stats = load_models()

if model is None:
    st.stop()

# Prediction Form
col1, col2 = st. columns([3, 2])

with col1:
    st.markdown("### Enter Details")
    
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            crop = st.selectbox("🌾 Crop", sorted(unique_vals['crops']))
        with col_b:
            state = st.selectbox("📍 State", sorted(unique_vals['states']))
        with col_c:
            season = st.selectbox("📅 Season", sorted(unique_vals['seasons']))
        
        st.markdown("#### Customize Parameters (Optional)")
        use_custom = st.checkbox("Use custom values")
        
        if use_custom:
            col_d, col_e = st.columns(2)
            with col_d:
                crop_year = st.number_input("Crop Year", min_value=1997, max_value=2030, 
                                          value=int(median_vals['Crop_Year']))
                rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, 
                                         value=float(median_vals['Annual_Rainfall']))
                fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0.0, 
                                           value=float(median_vals['Fertilizer']))
            with col_e:
                pesticide = st.number_input("Pesticide (kg/ha)", min_value=0.0, 
                                          value=float(median_vals['Pesticide']))
                area = st.number_input("Area (hectares)", min_value=0.0, 
                                     value=float(median_vals['Area']))
                production = st. number_input("Production (tons)", min_value=0.0, 
                                           value=float(median_vals['Production']))
        else:
            crop_year = int(median_vals['Crop_Year'])
            rainfall = float(median_vals['Annual_Rainfall'])
            fertilizer = float(median_vals['Fertilizer'])
            pesticide = float(median_vals['Pesticide'])
            area = float(median_vals['Area'])
            production = float(median_vals['Production'])
        
        submit_button = st.form_submit_button("🔮 Predict Yield")
    
    if submit_button: 
        with st.spinner("Calculating... "):
            # Prepare input data
            input_data = prepare_input_data(crop, state, season, crop_year, rainfall,
                                          fertilizer, pesticide, area, production,
                                          features, label_encoders, median_vals, agg_stats)
            
            # Make prediction
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Store in session state
            st.session_state['prediction'] = prediction
            st.session_state['crop'] = crop
            st.session_state['state'] = state
            st.session_state['season'] = season
            
            st.success("✅ Prediction Complete!")

with col2:
    st.markdown("### 📊 Result")
    
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        crop = st.session_state['crop']
        state = st.session_state['state']
        season = st.session_state['season']
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-result">
            <div style="font-size: 1rem; margin-bottom: 0.5rem;">Predicted Yield</div>
            <div style="font-size: 3rem; font-weight: bold;">{prediction:.2f}</div>
            <div style="font-size: 1rem;">metric tons per hectare</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"**Crop:** {crop}  \n**State:** {state}  \n**Season:** {season}")
        
        # Download result
        result_df = pd.DataFrame({
            'Crop': [crop],
            'State': [state],
            'Season': [season],
            'Predicted Yield (t/ha)': [prediction]
        })
        
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Result",
            data=csv,
            file_name=f"prediction_{crop}_{state}. csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Fill in the form and click 'Predict Yield'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color:  #666; padding: 1rem;'>
    <p>🌾 Crop Yield Prediction System | © 2025</p>
</div>
""", unsafe_allow_html=True)
