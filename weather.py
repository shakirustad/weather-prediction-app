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
            
            if not data.get('daily'):
                raise ValueError("No data received from weather API")
                
            df = pd.DataFrame({
                'ds': pd.to_datetime(data['daily']['time']),
                'y': data['daily']['temperature_2m_max'],
                'precipitation': data['daily']['precipitation_sum'],
                'windspeed': data['daily']['windspeed_10m_max']
            })
            
            return df, None
        except requests.RequestException as e:
            return None, f"Failed to fetch weather data: {str(e)}"
        except Exception as e:
            return None, str(e)

    def train_and_predict(self, df, forecast_days=30):
        """Train model and generate predictions"""
        try:
            if df.empty:
                raise ValueError("No data available for training")
            
            # Create a new Prophet instance for each training
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
            
            model.fit(df[['ds', 'y']])
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            return forecast, None
        except Exception as e:
            return None, f"Failed to generate forecast: {str(e)}"
    

    def create_forecast_plot(self, forecast, location_name="Selected Location"):
        """Create an interactive forecast plot"""
        try:
            fig = go.Figure()

            historical_end = len(forecast) - 30
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'][:historical_end],
                y=forecast['yhat'][:historical_end],
                name='Historical',
                line=dict(color='blue')
            ))

            fig.add_trace(go.Scatter(
                x=forecast['ds'][-30:],
                y=forecast['yhat'][-30:],
                name='Forecast',
                line=dict(color='red')
            ))

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
        except Exception as e:
            return None

# Initialize the predictor
predictor = WeatherPredictor()

# Kenya locations dictionary with more accurate coordinates
KENYA_LOCATIONS = {
    "Nairobi": {"lat": -1.2921, "lon": 36.8219},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    "Kisumu": {"lat": -0.1022, "lon": 34.7617},
    "Nakuru": {"lat": -0.3031, "lon": 36.0800},
    "Eldoret": {"lat": 0.5143, "lon": 35.2698}
}

def get_date_range():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

@app.route("/", methods=["GET", "POST"])
# @app.route("/", methods=["GET", "POST"])
def index():
    forecast_plot = None
    error = None
    weather_data = None
    locations = list(KENYA_LOCATIONS.keys())  # Always get locations
    
    if request.method == "POST":
        location = request.form.get('location')
        if not location or location not in KENYA_LOCATIONS:
            error = "Please select a valid location"
            return render_template("index.html", error=error, locations=locations)
            
        coords = KENYA_LOCATIONS[location]
        start_date, end_date = get_date_range()
        
        df, fetch_error = predictor.fetch_weather_data(
            coords['lat'], 
            coords['lon'], 
            start_date, 
            end_date
        )
        
        if fetch_error:
            error = fetch_error
        else:
            forecast, predict_error = predictor.train_and_predict(df)
            
            if predict_error:
                error = predict_error
            else:
                forecast_plot = predictor.create_forecast_plot(forecast, location)
                if forecast_plot is None:
                    error = "Failed to generate forecast plot"
                else:
                    weather_data = {
                        'current_temp': float(forecast['yhat'].iloc[-1]),
                        'temp_trend': 'rising' if forecast['yhat'].diff().tail(7).mean() > 0 else 'falling',
                        'confidence': float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
                    }
    
    return render_template(
        "index.html",
        forecast_plot=forecast_plot,
        error=error,
        locations=locations,  # Always pass locations
        weather_data=weather_data
    )
if __name__ == "__main__":
    app.run(debug=True)
