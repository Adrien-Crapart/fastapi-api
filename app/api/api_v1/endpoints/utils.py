from typing import Any, List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query, Response, WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse, FileResponse
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

from app import models, schemas
from app.api import deps
from app.core.celery_app import celery_app
from app.utils import send_test_email

import pytz
import feedparser
from PIL import Image, ImageOps
from io import BytesIO
import qrcode
import httpx
import asyncio
import websockets
from shapely.geometry import Point
import geopandas as gpd
from googlesearch import search
import folium
import internetdownloadmanager as idm
from tempfile import NamedTemporaryFile
import json
import csv
import psycopg2
from forex_python.converter import CurrencyRates
from pint import UnitRegistry
from pathlib import Path

router = APIRouter()

@router.post("/test-celery/", response_model=schemas.Msg, status_code=201)
def test_celery(
    msg: schemas.Msg,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test Celery worker.
    """
    celery_app.send_task("app.worker.test_celery", args=[msg.msg])
    return {"msg": "Word received"}

@router.post("/test-email/", response_model=schemas.Msg, status_code=201)
def test_email(
    email_to: EmailStr,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test emails.
    """
    send_test_email(email_to=email_to)
    return {"msg": "Test email sent"}

@router.get("/current_datetime")
async def get_current_datetime():
    # Define the timezone for UTC Paris
    paris_timezone = pytz.timezone("Europe/Paris")
    
    # Get the current datetime in UTC Paris time zone
    current_datetime = datetime.now(pytz.utc).astimezone(paris_timezone)
    
    # Define the desired datetime format
    datetime_format = "%Y-%m-%d %H:%M:%S %Z%z"
    
    # Format the datetime according to the specified format
    formatted_datetime = current_datetime.strftime(datetime_format)
    
    # Extract day, month, year
    day = current_datetime.day
    month = current_datetime.month
    year = current_datetime.year
    
    # Extract the timestamp
    timestamp = current_datetime.timestamp()
    
    return {
        "day": day,
        "month": month,
        "year": year,
        "timestamp": timestamp,
        "formatted_datetime": formatted_datetime
    }

@router.get("/search-google/")
async def search_google(query: str, num_results: int):
    try:
        # Perform Google search
        search_results = list(
            search(query, num_results=num_results, advanced=True))
        return {"results": search_results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An error occurred during the search.")

@router.get("/health-check")
async def health_check(
    db: Session = Depends(deps.get_sync_session)
    # current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Test health of database instance.
    """
    try:
        db.execute(text("SELECT 1"))
        return {"message": "Database is healthy"}
    except OperationalError as e:
        return HTTPException(status_code=500, detail="Database is not reachable")

# Route to detect news between two dates
@router.get("/detect-news-on-gpu")
async def detect_news():
    try:
        # Parse the RSS feed
        feed = feedparser.parse("https://www.geoportail-urbanisme.gouv.fr/atom/download-feed")
        
        # Check if the feed is valid
        if feed.get('bozo_exception'):
            raise HTTPException(status_code=400, detail="Invalid RSS feed")

        # Extract news items from the feed
        news_items = []
        for entry in feed.entries:
            news_item = {
                "title": entry.title,
                "link": entry.link,
                "updated": entry.updated,
                "summary": entry.summary,
            }
            zip_feed = feedparser.parse(entry.id)
            for zip_entry in zip_feed.entries:
                news_item["zip_file"] = zip_entry.id
            
            news_items.append(news_item)

        return {"news": news_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate_qr")
async def generate_qr_code(data: str = Query(..., description="Data to encode in the QR code")):
    try:
        # Create a QRCode object and add data to it
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create a PIL Image from the QR code
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Save the image as bytes
        img_bytes = BytesIO()
        qr_image.save(img_bytes, format="PNG")
        
        # Get the bytes content
        img_bytes = img_bytes.getvalue()
        
        # Return the QR code image as binary data
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/download_file/{url}/{output}')
async def Downloader(url, output):
    pydownloader = idm.Downloader(
        worker=20, part_size=1024*1024*10, resumable=True)
    return (pydownloader)

@router.post("/json-to-csv/")
async def convert_json_to_csv(json_file: UploadFile):
    """
    Convert JSON data to CSV format.

    **Input:**

    - `json_file`: An uploaded JSON file containing data to be converted.

    **Output:**

    - A downloadable CSV file containing the converted data.
    """
    try:
        # Check if the uploaded file has a JSON extension
        if not json_file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Uploaded file must be in JSON format")

        # Read the JSON content from the uploaded file
        json_data = await json_file.read()

        # Parse the JSON data
        data = json.loads(json_data)

        # Create a temporary CSV file
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as csv_file:
            csv_filename = csv_file.name

            # Write JSON data to the CSV file
            csv_writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(data)

        return {"message": "JSON to CSV conversion successful", "csv_filename": csv_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

c = CurrencyRates()

@router.get("/convert-currency/")
async def convert_currency(from_currency: str, to_currency: str, amount: float):
    """To convert between different currencies.

    **Input:**

    - `from_currency`: Origin currency

    - `to_currency`: Desired currency

    - `amount`: A value at converted

    **Output:**

    - A value convert in json format

    **Example:**

    **GET** `/convert-currency/?from_currency=USD&to_currency=EUR&amount=100`

    `Response: {"converted_amount": 83.67}`
    """
    try:
        result = c.convert(from_currency, to_currency, amount)
        return {"converted_amount": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

ureg = UnitRegistry()

@router.get("/convert-measurement/")
async def convert_measurement(quantity: str, from_unit: str, to_unit: str):
    """To perform conversions between different units of measurement.

    **Input:**

    - `quantity`: Units of mesure

    - `from_unit`: Origin mesure

    - `to_unit`: Desired mesure

    **Output:**

    - A value convert in json format

    **Example:**

    **GET** `/convert-measurement/?quantity=10&from_unit=meter&to_unit=foot`

    `Response: {"converted_value": 32.80839895013123, "converted_unit": "foot"}`
    """
    try:
        input_value = ureg(quantity + from_unit)
        converted_value = input_value.to(to_unit)
        return {"converted_value": converted_value.magnitude, "converted_unit": str(converted_value.units)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# OpenWeatherMap API configuration
API_KEY = "1ea8ffb5be1b1267e3fddb7be2382934"
BASE_URL = "https://api.openweathermap.org/data/3.0"
UNITS = "metric"  # You can change this to "imperial" for Fahrenheit

# Route to get current weather by city name
@router.get("/current-weather")
async def get_current_weather(
    city: str = Query(..., title="City Name", description="Name of the city"),
    country: str = Query(None, title="Country Code", description="Country code (optional)"),
):
    try:
        params = {
            "q": f"{city},{country}" if country else city,
            "appid": API_KEY,
            "units": UNITS,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/weather", params=params)
            response.raise_for_status()
            data = response.json()

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to get weather forecast by city name
@router.get("/weather-forecast")
async def get_weather_forecast(
    city: str = Query(..., title="City Name", description="Name of the city"),
    country: str = Query(None, title="Country Code", description="Country code (optional)"),
):
    try:
        params = {
            "q": f"{city},{country}" if country else city,
            "appid": API_KEY,
            "units": UNITS,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/forecast", params=params)
            response.raise_for_status()
            data = response.json()

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Create a dictionary to store connected WebSocket clients
connected_clients = {}

# WebSocket route for chat
@router.websocket("/chat/{username}")
async def chat_endpoint(
    websocket: WebSocket,
    username: str
):
    # Accept the WebSocket connection
    await websocket.accept()

    # Store the WebSocket client in the connected_clients dictionary
    connected_clients[username] = websocket

    try:
        while True:
            # Receive messages from the client
            message = await websocket.receive_text()
            print(f"Received message from {username}: {message}")

            # Broadcast the message to all connected clients
            for client_username, client_websocket in connected_clients.items():
                if client_username != username:
                    await client_websocket.send_text(f"{username}: {message}")
                    print(f"Sent message to {client_username}: {message}")
    except WebSocketDisconnect:
        # Remove the client from the connected_clients dictionary when they disconnect
        del connected_clients[username]
        await websocket.close()

# from pydantic import BaseModel

# # Define a model for the location and buffer parameters
# class LocationParams(BaseModel):
#     latitude: float
#     longitude: float
#     buffer_meters: float = Query(..., description="Buffer distance in meters")

# # Load your Points of Interest data as a GeoDataFrame (replace 'poi_data.geojson' with your data)
# poi_gdf = gpd.read_file('poi_data.geojson')

# @router.post("/find_nearby_pois", response_model=List[POI])
# async def find_nearby_pois(
#     location: Location,
#     buffer_radius_meters: float = Query(1000.0, description="Buffer radius in meters"),
#     output_format: str = Query("json", description="Output format (json or image)"),
# ):
#     from shapely.geometry import Point

#     user_point = Point(location.longitude, location.latitude)

#     nearby_pois = []
#     for poi in poi_data:
#         poi_point = Point(poi.location.longitude, poi.location.latitude)
#         distance_meters = user_point.distance(poi_point) * 111195
#         if distance_meters <= buffer_radius_meters:
#             nearby_pois.append(poi)

#     # Sort POIs by distance
#     nearby_pois = sorted(nearby_pois, key=lambda x: user_point.distance(Point(x.location.longitude, x.location.latitude)) * 111195)

#     if output_format == "image":
#         # Generate an image with POI markers on a map
#         fig = plt.figure(figsize=(10, 8))
#         ax = plt.axes(projection=ccrs.PlateCarree())

#         # Add a base map (e.g., using cartopy's stock image)
#         ax.stock_img()

#         # Plot the user's location
#         ax.plot(location.longitude, location.latitude, 'bo', markersize=8, transform=ccrs.PlateCarree(), label="User Location")

#         # Plot POIs
#         for poi in nearby_pois:
#             ax.plot(poi.location.longitude, poi.location.latitude, 'ro', markersize=8, transform=ccrs.PlateCarree(), label=poi.name)

#         # Add a scale bar and north arrow
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("bottom", size="5%", pad=0.05)
#         plt.colorbar(ax, cax=cax, orientation="horizontal", label="Scale")

#         ax.annotate('N', xy=(location.longitude, location.latitude), xytext=(location.longitude, location.latitude+0.02), textcoords='data',
#                     arrowprops=dict(arrowstyle='->', lw=1.5), ha='center', fontsize=12)

#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         ax.legend()
        
#         img_byte_map = BytesIO()
#         plt.savefig(img_byte_map, format='png', bbox_inches='tight')
#         img_byte_map.seek(0)
#         image = Image.open(img_byte_map)
#         img_byte_array = BytesIO()
#         image.save(img_byte_array, format="PNG")
#         return Response(content=img_byte_array.getvalue(), media_type="image/png")
#     elif output_format == "json":
#         # Return POIs in JSON format
#         return nearby_pois
#     else:
#         return {"error": "Invalid output_format parameter"}

# Define a list of music files
music_files = [
    "music1.mp3",
    "music2.mp3",
    "music3.mp3",
]

# Initialize the index of the currently playing music
current_music_index = 0

@router.get("/play")
async def play():
    global current_music_index
    current_music = music_files[current_music_index]
    return {"status": "Playing", "music": current_music}

@router.get("/pause")
async def pause():
    return {"status": "Paused"}

@router.get("/stop")
async def stop():
    return {"status": "Stopped"}

@router.get("/next")
async def next():
    global current_music_index
    current_music_index = (current_music_index + 1) % len(music_files)
    return {"status": "Next"}

@router.get("/previous")
async def previous():
    global current_music_index
    current_music_index = (current_music_index - 1) % len(music_files)
    return {"status": "Previous"}

@router.get("/current-music")
async def current_music():
    current_music = music_files[current_music_index]
    return {"current_music": current_music}

@router.get("/stream-music")
async def stream_music():
    current_music = music_files[current_music_index]
    music_path = Path(f"music/{current_music}")
    return FileResponse(music_path)