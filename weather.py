from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from prophet.plot import plot_plotly
import plotly.io as pio
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

class WeatherPredictor:
    def __init__(self):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
    def fetch_weather_data(self, latitude, longitude, start_date, end_date):
        """Fetch historical weather data from Open-Meteo API"""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "precipitation_sum", "windspeed_10m_max"],
            "timezone": "Africa/Nairobi"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Create DataFrame with all weather parameters
            df = pd.DataFrame({
                'ds': pd.to_datetime(data['daily']['time']),
                'y': data['daily']['temperature_2m_max'],
                'precipitation': data['daily']['precipitation_sum'],
                'windspeed': data['daily']['windspeed_10m_max']
            })
            
            return df, None
        except Exception as e:
            return None, str(e)

    def train_and_predict(self, df, forecast_days=30):
        """Train model and generate predictions"""
        try:
            # Fit the model
            self.model.fit(df[['ds', 'y']])
            
            # Create future dates
            future = self.model.make_future_dataframe(periods=forecast_days)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            return forecast, None
        except Exception as e:
            return None, str(e)

    def create_forecast_plot(self, forecast, location_name="Selected Location"):
        """Create an interactive forecast plot"""
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast['ds'][:len(forecast)-30],
            y=forecast['yhat'][:len(forecast)-30],
            name='Historical',
            line=dict(color='blue')
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'][-30:],
            y=forecast['yhat'][-30:],
            name='Forecast',
            line=dict(color='red')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'][-30:],
            y=forecast['yhat_upper'][-30:],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'][-30:],
            y=forecast['yhat_lower'][-30:],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='95% Confidence Interval'
        ))

        fig.update_layout(
            title=f'Temperature Forecast for {location_name}',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return pio.to_html(fig, full_html=False)

# Initialize the predictor
predictor = WeatherPredictor()

# Kenya locations dictionary
KENYA_LOCATIONS = {
    "Nairobi": {"lat": -1.2921, "lon": 36.8219},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    "Kisumu": {"lat": -0.1022, "lon": 34.7617},
    "Nakuru": {"lat": -0.3031, "lon": 36.0800},
    "Eldoret": {"lat": 0.5143, "lon": 35.2698}
}

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_plot = None
    error = None
    weather_data = None
    
    if request.method == "POST":
        try:
            # Get location from form
            location = request.form.get('location', 'Nairobi')
            coords = KENYA_LOCATIONS.get(location)
            
            if not coords:
                raise ValueError("Invalid location selected")
                
            # Calculate dates
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch and process data
            df, error = predictor.fetch_weather_data(
                coords['lat'], 
                coords['lon'], 
                start_date, 
                end_date
            )
            
            if error:
                raise Exception(error)
                
            # Generate forecast
            forecast, error = predictor.train_and_predict(df)
            
            if error:
                raise Exception(error)
                
            # Create plot
            forecast_plot = predictor.create_forecast_plot(forecast, location)
            
            # Prepare current weather data
            weather_data = {
                'current_temp': forecast['yhat'].iloc[-1],
                'temp_trend': 'rising' if forecast['yhat'].diff().tail(7).mean() > 0 else 'falling',
                'confidence': forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
            }
            
        except Exception as e:
            error = str(e)
    
    return render_template(
        "index.html",
        forecast_plot=forecast_plot,
        error=error,
        locations=KENYA_LOCATIONS.keys(),
        weather_data=weather_data
    )

@app.route("/api/forecast/<location>")
def api_forecast(location):
    """API endpoint for getting forecast data"""
    try:
        if location not in KENYA_LOCATIONS:
            return jsonify({"error": "Invalid location"}), 400
            
        coords = KENYA_LOCATIONS[location]
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        df, error = predictor.fetch_weather_data(
            coords['lat'],
            coords['lon'],
            start_date,
            end_date
        )
        
        if error:
            return jsonify({"error": error}), 500
            
        forecast, error = predictor.train_and_predict(df)
        
        if error:
            return jsonify({"error": error}), 500
            
        # Convert forecast to JSON-serializable format
        forecast_data = {
            'dates': forecast['ds'].tail(30).dt.strftime('%Y-%m-%d').tolist(),
            'temperatures': forecast['yhat'].tail(30).round(1).tolist(),
            'upper_bound': forecast['yhat_upper'].tail(30).round(1).tolist(),
            'lower_bound': forecast['yhat_lower'].tail(30).round(1).tolist()
        }
        
        return jsonify(forecast_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
