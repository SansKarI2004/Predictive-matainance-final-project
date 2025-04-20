import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the saved model, scaler, and feature names."""
    try:
        model = joblib.load('best_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        # Try to load scaler if it exists
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
        else:
            scaler = None
            
        return model, feature_names, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def load_or_train_model():
    """Load the existing model or train a new one if not available."""
    if not os.path.exists('best_model.pkl'):
        st.warning("Model not found. Training a new model...")
        
        # Start training in a separate process
        import optimized_model_training as model_training
        
        # Add progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading dataset...")
        progress_bar.progress(10)
        
        # Run the model training
        try:
            model_training.main()
            progress_bar.progress(100)
            status_text.text("Model training completed successfully!")
            st.success("Model training completed!")
            
            # Check if model was created
            if os.path.exists('best_model.pkl'):
                return load_model()
            else:
                st.error("Model training completed but model file was not created. Please check logs.")
                return None, None, None
                
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            return None, None, None
    
    return load_model()

def make_prediction(model, feature_names, input_data, scaler=None):
    """
    Make prediction using the trained model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        input_data: Input data for prediction
        scaler: Fitted scaler for numerical features
    
    Returns:
        tuple: Prediction and probability
    """
    try:
        # Convert input data to DataFrame with appropriate column names
        df = pd.DataFrame([input_data])
        
        # For debugging
        st.write("Debug: Creating prediction dataframe with correct features")
        
        # Create a new empty dataframe with columns matching the expected features
        result_df = pd.DataFrame()
        
        # Copy numerical features first
        numeric_features = [
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        for feat in numeric_features:
            if feat in df.columns:
                result_df[feat] = df[feat]
        
        # Add dummy value for 'Machine failure' 
        # (It will be ignored during prediction but needed for column structure)
        result_df['Machine failure'] = 0
        
        # Handle Type column (one-hot encoding)
        if 'Type' in df.columns:
            machine_type = df['Type'].iloc[0]
            # Add the Type columns based on the selected machine type
            result_df['Type_L'] = 1 if machine_type == 'L' else 0
            result_df['Type_M'] = 1 if machine_type == 'M' else 0
            # Type_H is reference category, so it's not included in the model
        
        # Ensure columns match feature_names exactly in order and content
        # Some models might expect Type_H which isn't in feature_names
        if 'Type_H' in feature_names and 'Type_H' not in result_df.columns:
            result_df['Type_H'] = 1 if df['Type'].iloc[0] == 'H' else 0
        
        # For debugging
        st.write("Debug: Feature dataframe created successfully")
        
        # Apply scaling if available
        if scaler is not None:
            try:
                # Order the columns to match what the scaler expects
                scaler_columns = [col for col in scaler.feature_names_in_ if col in result_df.columns]
                if len(scaler_columns) == result_df.shape[1]:
                    result_df = result_df[scaler_columns]  # Ensure correct column order
                    scaled_data = scaler.transform(result_df)
                    st.write("Debug: Scaling applied successfully")
                    
                    # Make prediction with scaled data
                    prediction = model.predict(scaled_data)[0]
                    probability = model.predict_proba(scaled_data)[0][1]
                    return prediction, probability
                else:
                    st.warning("Column mismatch for scaler. Using direct prediction.")
            except Exception as scaling_error:
                st.warning(f"Error during scaling: {scaling_error}. Using direct prediction.")
        
        # Reorder columns to match feature_names
        if feature_names:
            try:
                # Filter to only include the columns that exist in both dataframes
                common_cols = [col for col in feature_names if col in result_df.columns]
                result_df = result_df[common_cols]
                st.write(f"Debug: Reordered columns to {common_cols}")
            except Exception as col_error:
                st.warning(f"Error reordering columns: {col_error}")
        
        # Make prediction with prepared data
        prediction = model.predict(result_df)[0]
        probability = model.predict_proba(result_df)[0][1]
        
        # Remove debug messages in production
        st.write("Debug: Prediction successful")
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error(f"Input data columns: {list(input_data.keys())}")
        st.error(f"Expected feature names: {feature_names}")
        
        # More detailed debugging
        if 'Type' in input_data:
            st.error(f"Machine type: {input_data['Type']}")
            
        # Try rebuilding the model
        st.error("Please click the 'Retrain Model' button at the top of the page and try again.")
        
        return None, None

def display_metrics(model):
    """Display model metrics if available."""
    if os.path.exists('roc_curves.png') and os.path.exists('confusion_matrices.png'):
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.image('roc_curves.png', caption='ROC Curves', use_container_width=True)
        with col2:
            st.image('confusion_matrices.png', caption='Confusion Matrices', use_container_width=True)

def main():
    """Main function for the Streamlit app."""
    st.title("🔧 Predictive Maintenance System")
    st.write("""
    This application predicts machine failures based on operational data using machine learning. 
    Enter the machine's operational parameters to get a prediction on whether it's likely to fail.
    """)
    
    # Add a button to retrain model at the top right
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("⟳ Retrain Model"):
            # Remove existing model files
            for file in ['best_model.pkl', 'feature_names.pkl', 'scaler.pkl', 'roc_curves.png', 'confusion_matrices.png']:
                if os.path.exists(file):
                    os.remove(file)
            st.success("Model files removed. Retraining will start.")
            st.rerun()
    
    # First check if model training is already in progress
    if not os.path.exists('best_model.pkl'):
        # Display a message that model training is about to start
        st.info("Model not found. The system will train a new model now. This may take several minutes to complete.")
        st.warning("Please wait while the model is being trained. The page will update automatically when training is complete.")
        
        # Add a spinner to indicate training is happening
        with st.spinner('Training in progress...'):
            # Load or train model
            model, feature_names, scaler = load_or_train_model()
            
            if model is None:
                st.error("Failed to load or train the model. Please check the logs and try refreshing the page.")
                st.error("If the error persists, there may be an issue with the model training process.")
                # Show common troubleshooting steps
                st.markdown("""
                **Troubleshooting tips:**
                - Try refreshing the page
                - Check if dataset is available and properly formatted
                - Ensure all dependencies are installed correctly
                """)
                return
    else:
        # Model exists, just load it
        model, feature_names, scaler = load_model()
        
        if model is None:
            st.error("Failed to load the existing model. The model file might be corrupted.")
            # Add a button to retrain the model if loading fails
            if st.button("Retrain Model"):
                # Remove existing model files
                for file in ['best_model.pkl', 'feature_names.pkl', 'scaler.pkl']:
                    if os.path.exists(file):
                        os.remove(file)
                # Reload the page to start training
                st.rerun()
            return
    
    # Sidebar - Input parameters
    st.sidebar.header("Machine Parameters")
    
    # Categorical features
    machine_type = st.sidebar.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        help="Select the type of machine (Low, Medium, or High)"
    )
    
    # Numerical features
    st.sidebar.subheader("Operational Parameters")
    air_temperature = st.sidebar.slider("Air Temperature [K]", 295.0, 305.0, 300.0, 0.1)
    process_temperature = st.sidebar.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
    rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1500, 10)
    torque = st.sidebar.slider("Torque [Nm]", 3.0, 77.0, 40.0, 0.1)
    tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 125, 1)
    
    # Optional failure flags
    st.sidebar.subheader("Known Failure Flags (Optional)")
    st.sidebar.write("Note: These flags are not used for prediction, they're just for reference.")
    twf_flag = st.sidebar.checkbox("Tool Wear Failure (TWF)", False)
    hdf_flag = st.sidebar.checkbox("Heat Dissipation Failure (HDF)", False)
    pwf_flag = st.sidebar.checkbox("Power Failure (PWF)", False)
    osf_flag = st.sidebar.checkbox("Overstrain Failure (OSF)", False)
    rnf_flag = st.sidebar.checkbox("Random Failure (RNF)", False)
    
    # Create input data dictionary
    input_data = {
        'Air temperature [K]': air_temperature,
        'Process temperature [K]': process_temperature,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type': machine_type
    }
    
    # Make prediction when button is clicked
    if st.sidebar.button("Predict Failure"):
        prediction, probability = make_prediction(model, feature_names, input_data, scaler)
        
        if prediction is not None:
            # Main panel - Results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Machine Failure Predicted")
                else:
                    st.success("✅ No Failure Predicted")
                
                st.metric("Failure Probability", f"{probability*100:.2f}%")
                
                # Determine risk level
                if probability < 0.3:
                    risk_level = "Low Risk"
                    risk_color = "green"
                elif probability < 0.7:
                    risk_level = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "High Risk"
                    risk_color = "red"
                
                st.markdown(f"Risk Level: <span style='color:{risk_color};font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
            
            with col2:
                # Create probability gauge chart
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Create a simple gauge chart
                ax.pie([probability, 1-probability], colors=['red', 'lightgrey'], 
                      startangle=90, counterclock=False,
                      wedgeprops={'width': 0.3, 'edgecolor': 'w'})
                
                ax.text(0, 0, f"{probability*100:.1f}%", ha='center', va='center', fontsize=20)
                ax.set_title("Failure Probability", pad=20)
                
                # Remove axis
                ax.axis('equal')
                st.pyplot(fig)
            
            # Display input data
            st.subheader("Input Parameters")
            input_df = pd.DataFrame([input_data])
            st.dataframe(input_df)
            
            # Maintenance recommendations
            st.subheader("Maintenance Recommendations")
            if prediction == 1:
                st.markdown("""
                Based on the prediction, we recommend the following actions:
                1. **Schedule maintenance** within the next 24 hours
                2. **Inspect machine components**, especially those related to:
                   - Cooling system (if temperature is high)
                   - Motor and bearings (if rotational speed or torque are abnormal)
                   - Cutting tools (if tool wear is high)
                3. **Prepare replacement parts** to minimize downtime
                """)
            else:
                st.markdown("""
                Based on the prediction, we recommend the following actions:
                1. **Continue regular monitoring**
                2. **Perform routine maintenance** as scheduled
                3. **Document current parameters** for future reference
                """)
    
    # Display metrics section
    display_metrics(model)
    
    # About and information section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This application uses machine learning to predict equipment failures.
    It was trained on the AI4I 2020 Predictive Maintenance Dataset.
    """)

if __name__ == "__main__":
    main()
