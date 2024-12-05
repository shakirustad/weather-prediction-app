from flask import Flask, request, render_template
import pandas as pd
from prophet import Prophet
import requests
from prophet.plot import plot_plotly
import plotly.io as pio

app = Flask(__name__)

# Fetch historical weather data
def fetch_weather_data(latitude, longitude, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "timezone": "UTC"
    }
    response = requests.get(url, params=params)
    data = response.json()

    dates = data['daily']['time']
    temps = data['daily']['temperature_2m_max']
    df = pd.DataFrame({'ds': dates, 'y': temps})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    forecast_plot = None
    error = None

    if request.method == "POST":
        try:
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            start_date = request.form['start_date']
            end_date = request.form['end_date']

            # Fetch weather data
            df = fetch_weather_data(latitude, longitude, start_date, end_date)

            # Train Prophet model
            model = Prophet()
            model.fit(df)

            # Make future predictions
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # Generate forecast plot
            fig = plot_plotly(model, forecast)
            forecast_plot = pio.to_html(fig, full_html=False)

        except Exception as e:
            error = str(e)

    return render_template("index.html", forecast_plot=forecast_plot, error=error)

if __name__ == "__main__":
    app.run(debug=True)
