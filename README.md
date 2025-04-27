Flight Delay Prediction System ✈️⏱️

Python
MySQL
scikit-learn
Streamlit

Introduction
A machine learning system that predicts flight delays with 97% accuracy using historical airline data. This project helps travelers and airlines anticipate delays by analyzing factors like weather, carrier history, and airport traffic.

Key Features
Dual prediction models: Classifies delays (>15 min) and predicts exact delay duration
Interactive dashboard: Real-time visualizations and predictions
Automated pipeline: From raw data to actionable insights
MySQL integration: Scalable data storage and retrieval

Installation
Prerequisites
Python 3.8+
MySQL 8.0+
pip package manager

Setup
bash
# 1. Clone repository
git clone https://github.com/yourusername/flight-delay-prediction.git
cd flight-delay-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up MySQL database
mysql -u root -p < SQL_Data_Preparation.sql

# 4. Import your flight data (CSV format)
python data_importer.py path/to/your_data.csv

Usage
Running the Dashboard

bash
streamlit run flight_delay_predictor.py

Making Predictions
python

# Sample API usage
from prediction_model import predict_delay

result = predict_delay(
    carrier="AA",
    airport="JFK",
    month=7,
    scheduled_flights=150,
    weather_conditions="clear"
)
print(f"Delay probability: {result['probability']:.1%}")

Common Queries
sql

-- Get worst-performing airlines

SELECT carrier_name, AVG(arr_delay) as avg_delay
FROM flight_delay_analysis
GROUP BY carrier_name
ORDER BY avg_delay DESC
LIMIT 10;

Configuration
Create .env file for environment variables:
ini
DB_HOST=localhost
DB_USER=flight_user
DB_PASSWORD=flight_pass
DB_NAME=flights_db

Contributing
We welcome contributions! Please follow these steps:
Fork the repository
Create your feature branch (git checkout -b feature/your-feature)
Commit your changes (git commit -m 'Add some feature')
Push to the branch (git push origin feature/your-feature)
Open a Pull Request

Need Help?
Contact: swagatsantra03@gmail.com
