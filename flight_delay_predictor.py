import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import os

# 1. Data Loading Function
@st.cache_data
def load_data():
    engine = create_engine(f'mysql+mysqlconnector://root:p%40sskey123@localhost:3306/flights_db')
    query = "SELECT * FROM flight_delay_analysis"
    return pd.read_sql(query, engine)

# 2. Data Preprocessing
def preprocess_data(df):
    # Feature engineering
    df['is_peak_month'] = df['month'].isin([6, 7, 8, 12]).astype(int)
    df['carrier_delay_ratio'] = df['carrier_ct'] / (df['arr_flights'] + 1)
    df['total_delay_causes'] = df[['carrier_ct','weather_ct','nas_ct','security_ct','late_aircraft_ct']].sum(axis=1)
    
    # Target variables
    df['delay_binary'] = (df['arr_del15'] > 0).astype(int)  # Classification target
    df['delay_minutes'] = df['arr_delay']  # Regression target
    
    return df

# 3. Model Training Functions
def train_classification_model(X, y):
    categorical_features = ['carrier', 'airport', 'month']
    numerical_features = ['arr_flights', 'carrier_ct', 'weather_ct', 
                         'nas_ct', 'total_delay_causes', 'is_peak_month',
                         'carrier_delay_ratio']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    return model

def train_regression_model(X, y):
    categorical_features = ['carrier', 'airport', 'month']
    numerical_features = ['arr_flights', 'carrier_ct', 'weather_ct',
                         'nas_ct', 'total_delay_causes', 'is_peak_month',
                         'carrier_delay_ratio']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    return model

# 4. Streamlit Dashboard
def main():
    st.title("✈️ Flight Delay Predictor")
    
    # Load data
    with st.spinner('Loading data...'):
        df = load_data()
        df = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("Model Controls")
    model_type = st.sidebar.radio("Select model type:", 
                                 ["Classification (Delay >15min)", 
                                  "Regression (Delay minutes)"])
    
    # Data exploration section
    st.header("Data Exploration")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x='delay_category', 
                                    title='Flight Delay Categories'))
    with col2:
        st.plotly_chart(px.box(df, y='arr_delay', x='month',
                              title='Delay by Month'))
    
    # Model training section
    st.header("Model Training")
    
    if model_type == "Classification (Delay >15min)":
        X = df.drop(columns=['delay_binary', 'delay_minutes', 'delay_category',
                            'year', 'carrier_name', 'airport_name', 'arr_delay'])
        y = df['delay_binary']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        with st.spinner('Training classification model...'):
            model = train_classification_model(X_train, y_train)
            y_pred = model.predict(X_test)
            
        st.success("Classification Model Trained!")
        st.text(classification_report(y_test, y_pred))
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importances = model.named_steps['classifier'].feature_importances_
        features = (model.named_steps['preprocessor']
                   .transformers_[0][2] + 
                   list(model.named_steps['preprocessor']
                       .transformers_[1][1]
                       .get_feature_names_out()))
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        st.plotly_chart(px.bar(importance_df.sort_values('Importance', ascending=False).head(10),
                              x='Importance', y='Feature', orientation='h'))
        
    else:
        X = df.drop(columns=['delay_binary', 'delay_minutes', 'delay_category',
                            'year', 'carrier_name', 'airport_name', 'arr_delay'])
        y = df['delay_minutes']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        with st.spinner('Training regression model...'):
            model = train_regression_model(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
        st.success(f"Regression Model Trained! MAE: {mae:.2f} minutes")
        
        # Prediction visualization
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.plotly_chart(px.scatter(results, x='Actual', y='Predicted',
                                  title='Actual vs Predicted Delays'))
    
    # Prediction interface
    st.header("Make Predictions")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            carrier = st.selectbox("Airline", df['carrier'].unique())
            airport = st.selectbox("Airport", df['airport'].unique())
            month = st.selectbox("Month", range(1, 13))
        with col2:
            arr_flights = st.number_input("Scheduled Flights", min_value=1)
            carrier_ct = st.number_input("Carrier Delay Count", min_value=0)
            weather_ct = st.number_input("Weather Delay Count", min_value=0)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = pd.DataFrame({
                'carrier': [carrier],
                'airport': [airport],
                'month': [month],
                'arr_flights': [arr_flights],
                'carrier_ct': [carrier_ct],
                'weather_ct': [weather_ct],
                'nas_ct': [0],  # Example value
                'total_delay_causes': [carrier_ct + weather_ct],
                'is_peak_month': [1 if month in [6,7,8,12] else 0],
                'carrier_delay_ratio': [carrier_ct/(arr_flights+1)]
            })
            
            if model_type == "Classification (Delay >15min)":
                proba = model.predict_proba(input_data)[0]
                st.success(f"Probability of delay >15 min: {proba[1]:.2%}")
            else:
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted delay: {prediction:.1f} minutes")

if __name__ == "__main__":
    main()