from typing import Any, List, Optional, Dict
from decouple import config
from pydantic import BaseModel

from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Query, Response, Depends, Form, BackgroundTasks
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import StreamingResponse

from app import crud, models, schemas
from app.api import deps

import geopandas as gpd
import pandas as pd
import json
import tempfile
from databases import Database
from io import BytesIO, StringIO
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
import unidecode
import re
from cachetools import TTLCache
from datetime import datetime, timedelta
import math
import asyncio
import asyncpg
import requests
from pathlib import Path
import aiofiles
import zipfile
from sqlalchemy.ext.asyncio import AsyncSession

# Create a cache with a TTL (time-to-live) of 300 seconds (5 minutes)
cache = TTLCache(maxsize=1000, ttl=300)

router = APIRouter()

# Initialize a global variable to track progress
total_files = 0
completed_files = 0


def clean_column_name(name):
    # Remove accents using unidecode
    name = unidecode.unidecode(name)
    # Replace spaces, parentheses, hyphens, and other special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Convert to lowercase
    name = name.lower()
    # Remove double underscores
    name = re.sub(r'_+', '_', name)
    # Remove final underscore
    name = name.strip('_')
    return name


async def process_single_file(
    file: UploadFile,
    output_format: str,
    delimiter: str,
    encoding: str,
    import_to_db: bool,
    db_name: str,
    schema: str,
    table_name: str,
    replace_data: bool,
    delete_conditions: dict
):
    global completed_files

    # Ensure the output format is lowercase for consistency
    output_format = output_format.lower()

    # Read the uploaded file
    file_extension = file.filename.split('.')[-1]
    file_content = await file.read()

    # Create a temporary file to save the converted content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}', mode='w', encoding='utf-8') as temp_file:
        if output_format == 'json':
            # Convert JSON to JSON (no conversion needed)
            if file_extension == 'json':
                temp_file.write(file_content.decode('utf-8'))

            # Convert CSV or TXT to JSON
            elif file_extension in ['csv', 'txt']:
                df = pd.read_csv(BytesIO(
                    file_content), delimiter=delimiter, encoding='latin-1', low_memory=False)
                df.columns = [clean_column_name(col) for col in df.columns]
                json_data = df.to_json(orient='records', force_ascii=False)
                temp_file.write(json_data)

            # Handle other formats as needed
            else:
                return {"error": "Unsupported conversion"}

        elif output_format == 'csv':
            # Convert JSON to CSV or TXT to CSV
            if file_extension == 'json':
                data = json.loads(file_content)
                df = pd.DataFrame(data)
                df.columns = [clean_column_name(col) for col in df.columns]
                csv_data = df.to_csv(
                    index=False, sep=delimiter, encoding='utf-8')
                temp_file.write(csv_data)

            elif file_extension == 'txt':
                df = pd.read_csv(BytesIO(
                    file_content), delimiter=delimiter, encoding='latin-1', low_memory=False)
                df.columns = [clean_column_name(col) for col in df.columns]
                csv_data = df.to_csv(
                    index=False, sep=delimiter, encoding='utf-8')
                temp_file.write(csv_data)

            # Handle other formats as needed
            else:
                return {"error": "Unsupported conversion"}

        elif output_format == 'gpkg':
            # Convert to GeoPackage (supports CSV, SHP, XLSX)
            if file_extension in ['csv', 'xlsx']:
                df = pd.read_csv(
                    BytesIO(file_content), delimiter=delimiter, encoding=encoding, low_memory=False)
                df.columns = [clean_column_name(col) for col in df.columns]
                gdf = gpd.GeoDataFrame(df)
                gpkg_data = gdf.to_file(BytesIO(), driver='GPKG')
                return StreamingResponse(content=gpkg_data, media_type='application/octet-stream')

            # Handle other formats as needed
            else:
                return {"error": "Unsupported conversion"}

        elif output_format == 'shp':
            # Convert to Shapefile (supports GeoPackage)
            if file_extension == 'gpkg':
                gdf = gpd.read_file(BytesIO(file_content))
                shp_data = gdf.to_file(BytesIO(), driver='ESRI Shapefile')
                return StreamingResponse(content=shp_data, media_type='application/octet-stream')

            # Handle other formats as needed
            else:
                return {"error": "Unsupported conversion"}

        elif output_format == 'xlsx':
            # Convert to XLSX (supports CSV, GeoPackage)
            if file_extension == 'csv':
                df = pd.read_csv(
                    BytesIO(file_content), delimiter=delimiter, encoding=encoding, low_memory=False)
                df.columns = [clean_column_name(col) for col in df.columns]
                xlsx_data = df.to_excel(BytesIO(), index=False)
                return StreamingResponse(content=xlsx_data, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            elif file_extension == 'gpkg':
                gdf = gpd.read_file(BytesIO(file_content))
                xlsx_data = gdf.to_excel(BytesIO(), index=False)
                return StreamingResponse(content=xlsx_data, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            # Handle other formats as needed
            else:
                return {"error": "Unsupported conversion"}

    if import_to_db:
        # Import data into the PostgreSQL database
        try:
            if db_name == 'postgres':
                conn = engine_postgres.connect()
            elif db_name == 'data':
                conn = engine_data.connect()
            if schema and table_name:
                if delete_conditions:
                    delete_query = f'DELETE FROM "{schema}"."{table_name}" WHERE '
                    for field, value in delete_conditions.items():
                        field = clean_column_name(field)
                        delete_query += f'"{field}" = {value} AND '
                    # Remove the trailing "AND "
                    delete_query = delete_query[:-5]
                    conn.execute(delete_query)

                df = pd.read_csv(temp_file.name, delimiter=delimiter,
                                 encoding='utf-8', low_memory=False)
                df.columns = [clean_column_name(col) for col in df.columns]
                df.to_sql(table_name, conn, schema=schema,
                          if_exists='replace' if replace_data else 'append', index=False)

            conn.close()
            # Increment the completed_files counter
            completed_files += 1
            return {"message": "Data imported into PostgreSQL successfully."}
        except Exception as e:
            # Increment the completed_files counter
            completed_files += 1
            return {"error": f"Database import error: {str(e)}"}

    # Serve the temporary file as a response
    response = FileResponse(
        temp_file.name,
        media_type='application/octet-stream',
        headers={
            "Content-Disposition": f"attachment; filename=converted.{output_format}"}
    )

    return response


@router.post("/convert-and-import-file")
async def convert_file(
    files: List[UploadFile] = File(...),
    output_format: str = Form(...),
    delimiter: str = Form(','),
    encoding: str = Form('utf-8'),
    import_to_db: bool = Form(False),
    db_name: str = Form(None),
    schema: str = Form(None),
    table_name: str = Form(None),
    replace_data: bool = Form(False),
    delete_conditions: dict = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    File Conversion and Processing Router

    This router provides endpoints for uploading, processing, and converting files. It supports various input file formats, data manipulation, and optional import to a PostgreSQL database.

    Endpoints:
    - POST /convert-files: Upload and process files asynchronously, with optional database import.

    Parameters:
    - files (List[UploadFile]): A list of uploaded files to be processed.
    - output_format (str): The desired output format for converted files (e.g., 'csv', 'json').
    - delimiter (str): The delimiter used for CSV files.
    - encoding (str): The character encoding of the input files.
    - import_to_db (bool): Whether to import data into a PostgreSQL database.
    - db_name (str): Name of database to import data (only 'postgres','data','localhost')
    - schema (str): The database schema for import.
    - table_name (str): The name of the database table for import.
    - replace_data (bool): Whether to replace existing data in the database table.
    - delete_conditions (dict): Conditions to specify rows for deletion before import.
    - unlimited_varchar (bool): Whether to use unlimited VARCHAR for columns during import.

    Returns:
    - JSON response with a message indicating the processing status.

    Usage:
    1. Upload one or more files using a POST request to /convert-files.
    2. Specify the desired output format, delimiter, and other optional parameters.
    3. Files will be processed asynchronously, and progress is displayed.
    4. Converted files can be downloaded or imported into a PostgreSQL database.

    """
    global total_files, completed_files

    # Initialize progress counters
    total_files = len(files)
    completed_files = 0

    # Create a task to update progress (asyncio)
    async def update_progress():
        while completed_files < total_files:
            await asyncio.sleep(1)  # Update progress every second
            print(f"Progress: {completed_files}/{total_files} files completed")

    # Start the progress update task
    asyncio.create_task(update_progress())

    # Iterate through the list of uploaded files and enqueue tasks for processing
    for file in files:
        # Enqueue a task for each file
        background_tasks.add_task(
            process_single_file,
            file,
            output_format,
            delimiter,
            encoding,
            import_to_db,
            db_name,
            schema,
            table_name,
            replace_data,
            delete_conditions
        )

    # Return a response indicating that files are being processed
    return {"message": f"{len(files)} file(s) are being processed asynchronously."}

@router.get("/geometry-parcel")
async def get_geometry_from_parcel(
    cadastre_id: str,
    session: AsyncSession = Depends(deps.get_async_session)
    ):
    query_session = f"""
            SELECT 
                cadastre_id,
                city_id,
                ST_AsGeoJSON(geom)::json AS geometry  
            FROM
                "cadastre_parcel_dev"."cadastre_parcel_{cadastre_id[0:3] if cadastre_id[0:2] == '97' else cadastre_id[0:2]}"
            WHERE
                cadastre_id = :cadastre_id;
        """
    result_session = await session.execute(query_session, {"cadastre_id": cadastre_id})
    row_session = result_session.fetchall()
    if not row_session:
        raise HTTPException(
            status_code=404, detail="This parcel doesn't exist.")
    
    # Convert the result rows into a list of dictionaries
    geojson_data = []
    for row in row_session:
        geojson_feature = {
            "type": "Feature",
            "properties": {
                "cadastre_id": row.cadastre_id,
                "city_id": row.city_id
            },
            "geometry": row.geometry
        }
        geojson_data.append(geojson_feature)

    return geojson_data

async def configuration_result(session, layer_name):
    query_session = """
        SELECT selected_fields, column_reference, table_reference, layer_name, join_reference, type_reference, used_method, availability, default_reference, gpu_layer_name, gpu_selected_fields
        FROM "references"."layers_configurations"
        WHERE label = :layer_name;
    """
    result_session = await session.execute(query_session, {"layer_name": layer_name})
    row_session = result_session.fetchone()
    if row_session is None:
        raise HTTPException(
            status_code=404, detail="This data doesn't exist or is badly written in the name.")
        
    return row_session

async def availability_result(session, availability, city_id):   
    try:
        query_session = f"""     
            SELECT {availability}
            FROM "data_availability"."data_availability"
            WHERE city_id = :city_id;
        """
        result_session = await session.execute(query_session, {"city_id": city_id})
        row_session = result_session.fetchone()
        if row_session[0] == 'disponible':
            data_availability = {"data_source":"interne", 
                                "availability": "available",
                                "verified":True, 
                                "update": "undefined",
                                "update_verified": "undefined",
                                "gpu_reference":None
                                }
            
        elif row_session[0] == 'non concerné':
            data_availability = {"data_source":"interne", 
                                "availability": "no data",
                                "verified":True, 
                                "update": None,
                                "update_verified": None,
                                "gpu_reference":None
                                }
        else:
            query_session = f"""     
                SELECT existing, existing_verified, update, update_verified, gpu_reference
                FROM "references"."availabilities_gpu"
                WHERE city_id = :city_id
                AND data = :availability;
            """
            result_session = await session.execute(query_session, {"city_id": city_id, "availability": availability})
            row_session = result_session.fetchone()     
            data_availability = {"data_source":"geoportail de l'urbanisme", 
                                "availability": "not available" if row_session[1] is False else "available",
                                "verified": False if row_session[1] is False else row_session[1], 
                                "update": None if row_session[2] is None else row_session[2],
                                "update_verified": None if row_session[3] is None else row_session[3],
                                "gpu_reference": None if row_session[4] is None else row_session[4]
                                }
            
        return data_availability
    
    except:    
        if row_session is None:
            raise HTTPException(
                status_code=404, detail="This data doesn't written correctly or empty, please contact data service.")
        
async def metadata_result(session, columns_reference, table_reference, city_id):
    query_session = f"""
            SELECT {columns_reference}
            FROM "{table_reference.replace('.', '"."')}"
            WHERE city_id = :city_id;
        """
    result_session = await session.execute(query_session, {"city_id": city_id})
    row_session = result_session.fetchone()
    if row_session is None:
        raise HTTPException(
            status_code=404, detail="Any parameters or intersection method is find, please fix the table parameters before retry.")
        
    return row_session

async def intersection_with_inside_tolerance(session, cadastre_id, layer_value, selected_fields, tolerance, field_filter, field_value):
    query_session = f"""
            SELECT DISTINCT ON (t1.cadastre_id {'' if selected_fields == ',' else selected_fields[:selected_fields.find(" as")]})
                t1.cadastre_id,
                t1.geom,
                t1.city_id{selected_fields}    
                CASE
                    WHEN ST_Intersects(ST_Transform(t1.geom, 2154), 
                        ST_Buffer(ST_Transform(t2.geom, 2154), -({tolerance}))) THEN 'auto' 
                    ELSE 'manual'
                END AS mode
            FROM
                "cadastre_parcel_dev"."cadastre_parcel_{cadastre_id[0:3] if cadastre_id[0:2] == '97' else cadastre_id[0:2]}" AS t1
            JOIN
                {layer_value} AS t2
            ON
                ST_Intersects(ST_Transform(t1.geom, 2154), ST_Buffer(ST_Transform(t2.geom, 2154), -0.5))
            WHERE
                t1.cadastre_id = :cadastre_id
                AND t2.{field_filter} = ANY(:field_value);
        """
    result_session = await session.execute(query_session, {"cadastre_id": cadastre_id, "field_value": field_value})
    row_session = result_session.fetchall()
    if row_session is None:
        raise HTTPException(
            status_code=404, detail="Any parameters passed to apply the geotraitment.")
        
    return row_session

async def intersection_with_outside_tolerance(session, cadastre_id, layer_value, selected_fields, field_filter, field_value):
    query_session = f"""
            SELECT DISTINCT ON (t1.cadastre_id {'' if selected_fields == ',' else selected_fields[:selected_fields.find(" as")]})
                t1.cadastre_id,
                t1.city_id{selected_fields}       
                CASE
                    WHEN ST_Intersects(ST_Transform(t1.geom, 2154), 
                        ST_Buffer(ST_Transform(t2.geom, 2154), -(t2.buffer))) THEN 'auto' 
                    ELSE 'manual'
                END AS mode
            FROM
                "cadastre_parcel_dev"."cadastre_parcel_{cadastre_id[0:3] if cadastre_id[0:2] == '97' else cadastre_id[0:2]}" AS t1
            JOIN
                {layer_value} AS t2
            ON
                ST_Intersects(ST_Transform(t1.geom, 2154), ST_Buffer(ST_Transform(t2.geom, 2154), t2.buffer))
            WHERE
                t1.cadastre_id = :cadastre_id
                AND t2.{field_filter} = ANY(:field_value);
        """
    result_session = await session.execute(query_session, {"cadastre_id": cadastre_id, "field_value": field_value})
    row_session = result_session.fetchall()
    if row_session is None:
        raise HTTPException(
            status_code=404, detail="Any parameters passed to apply the geotraitment.")
        
    return row_session

async def intersection_with_two_buffers(session, cadastre_id, layer_value, selected_fields, buffer_auto, buffer_manual):
    query_session = f"""
            SELECT DISTINCT ON (t1.cadastre_id { '' if selected_fields == ',' else selected_fields[:selected_fields.find(" as")]})
                t1.cadastre_id,
                t1.city_id{selected_fields}       
                CASE
                    WHEN ST_Intersects(ST_Transform(t1.geom, 2154), ST_Buffer(ST_Transform(t2.geom, 2154), :buffer_auto)) THEN 'CONCERNÉ' 
                    ELSE 'À PROXIMITÉ' 
                END as response,
                'auto' AS mode
            FROM
                "cadastre_parcel_dev"."cadastre_parcel_{cadastre_id[0:3] if cadastre_id[0:2] == '97' else cadastre_id[0:2]}" AS t1
            JOIN
                {layer_value} AS t2
            ON
                ST_Intersects(ST_Transform(t1.geom, 2154), ST_Buffer(ST_Transform(t2.geom, 2154), :buffer_manual))
            WHERE
                t1.cadastre_id = :cadastre_id;
        """
    result_session = await session.execute(query_session, {"cadastre_id": cadastre_id, "buffer_auto": buffer_auto, "buffer_manual": buffer_manual})
    row_session = result_session.fetchall()
    if row_session is None:
        raise HTTPException(
            status_code=404, detail="Any parameters passed to apply the geotraitment.")
        
    return row_session
    
@router.get("/intersection-server")
async def get_intersection_result(
    cadastre_id: str,
    layer_name: str = Query(..., description="Layer name of database"),
    session: AsyncSession = Depends(deps.get_async_session),
):

    start_time = datetime.now()

    # Define the SQL query for fetching parsed_fields and configuration_result
    config_row = await configuration_result(session, layer_name)
    print(config_row)
    if config_row[0] is not None:
        parsed_fields = json.loads(config_row[0])
        selected_fields = ',\n' + ', '.join([f't2.{key} as {value}' for key, value in parsed_fields.items()]) + ',\n'
    else:
        selected_fields = ','
    if config_row[4] is not None:
        join_value = json.loads(config_row[4])
        columns_reference = config_row[1] + f',{str(list(join_value.keys())[0])}'
    else:
        join_value = None
        columns_reference = config_row[1]
    table_reference = config_row[2]
    layer_value = config_row[3]
    type_reference = config_row[5]
    used_method = config_row[6]
    availability = config_row[7]
    default_reference = config_row[8]
    gpu_layer_value = config_row[9]
    if config_row[10] is not None:
        gpu_parsed_fields = json.loads(config_row[10])
        gpu_selected_fields = ',\n' + ', '.join([f't2.{key} as {value}' for key, value in gpu_parsed_fields.items()]) + ',\n'
    else:
        gpu_selected_fields = ','
    print(used_method,type_reference)

    availability_data = await availability_result(session, availability, cadastre_id[0:5])
    print(availability_data)
    if availability_data['availability'] == 'no data':
        return {
            "intersection_result": [],
            "count_values": 0,
            "layer_name": layer_name,
            "data_source": "interne/ geoportail de l'urbanisme",
            "availability": "no data",
            "verified": None,
            "update": None,
            "update_verified": None,
            "timestamp": datetime.now(),
            "delay": datetime.now() - start_time,
        }
    elif availability_data['gpu_reference'] is not None:
        layer_value = gpu_layer_value
        selected_fields = gpu_selected_fields
        
    metadata =await metadata_result(session, columns_reference, table_reference, cadastre_id[0:5])
    print(metadata)

    tolerance = str(0 if metadata[0] is None else metadata[0])
    parsed_data = json.loads(default_reference)
    buffer_auto = int(0)
    buffer_manual = int(0)
    if 'auto' in parsed_data:
        buffer_auto = int(parsed_data['auto'])
    if 'manual' in parsed_data:
        buffer_manual = int(parsed_data['manual'])

    if availability_data['gpu_reference'] is None and (len(metadata) > 1 and metadata[0] is not None):
        field_filter = str(list(join_value.values())[0])
        field_value = metadata[1].split(',')
    elif availability_data['gpu_reference'] is not None:
        field_filter = "partition"
        field_value = availability_data['gpu_reference'].split(',')
        tolerance = buffer_manual
    else:
        field_filter = "city_id"
        field_value = cadastre_id[0:5].split(',')
    
    if used_method == 'intersection_with_inside_tolerance' and type_reference == 'tolerance':       
        intersection_row = await intersection_with_inside_tolerance(
            session, cadastre_id, layer_value, selected_fields, tolerance, field_filter, field_value
        )
    elif used_method == 'intersection_with_outside_tolerance' and type_reference == 'boolean':
        intersection_row = await intersection_with_outside_tolerance(
            session, cadastre_id, layer_value, selected_fields, field_filter, field_value
        )
    elif used_method == 'intersection_with_two_buffers' and type_reference == 'boolean':
        intersection_row = await intersection_with_two_buffers(
            session, cadastre_id, layer_value, selected_fields, buffer_auto, buffer_manual
        )
    else:
        raise HTTPException(
            status_code=404, detail="Any method is reconize, please unsure if the method exist")

    return {
        "intersection_result": [dict(row) for row in intersection_row],
        "count_values": len(intersection_row),
        "layer_name": layer_name,
        "data_source": availability_data['data_source'],
        "availability": availability_data['availability'],
        "verified": availability_data['verified'],
        "update": availability_data['update'],
        "update_verified": availability_data['update_verified'],
        "timestamp": datetime.now(),
        "delay": datetime.now() - start_time,
    }



@router.get("/leaflet-map", response_class=HTMLResponse)
async def get_leaflet_map():
    with open("app/templates/leaflet_map.html", "r") as file:
        leaflet_html = file.read()
    return leaflet_html

# poi_data = [
#     {
#         "name": "Park 1",
#         "coordinates": {"type": "Point", "coordinates": [43.296368, 5.378113]},
#     },
#     {
#         "name": "Museum 1",
#         "coordinates": {"type": "Point", "coordinates": [43.2971965, 5.3803742]},
#     }
# ]

# class PointOfInterest(BaseModel):
#     name: str
#     coordinates: dict

# @router.get("/poi/")
# async def get_poi(
#     lat: float = Query(..., description="Latitude of the center point"),
#     lon: float = Query(..., description="Longitude of the center point"),
#     radius: float = Query(..., description="Search radius in meters"),
# ) -> List[PointOfInterest]:
#     """
#     Get points of interest (POI) within a specified radius of a given location.
#     """
#     poi_list = []

#     for poi in poi_data:
#         poi_lat, poi_lon = poi["coordinates"]["coordinates"]
#         distance = haversine(lat, lon, poi_lat, poi_lon)

#         if distance <= radius:
#             poi_list.append(PointOfInterest(**poi))

#     return poi_list


# def haversine(lat1, lon1, lat2, lon2):
#     # Radius of the Earth in meters
#     R = 6371000

#     # Convert latitude and longitude from degrees to radians
#     lat1 = math.radians(lat1)
#     lon1 = math.radians(lon1)
#     lat2 = math.radians(lat2)
#     lon2 = math.radians(lon2)

#     # Haversine formula
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     distance = R * c

#     return distance

# @router.get("/update-georisques")
# async def update_georisques_data(
#     clay: bool = Query(False, description="Update clay mouvement ?"),
#     industrial_installations: bool = Query(False, description="Update industrial installations ?"),
#     underground_cavity: bool = Query(False, description="Update underground_cavity ?")
#     ):
#     """
#     Get points of interest (POI) within a specified radius of a given location.
#     """
#     try:
#         deps = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','2A','2B','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','971','972','973','974','976']

#         if clay:
#             # zipfile_url = "https://files.georisques.fr/argiles/AleaRG_Fxx_L93.zip"
#             # response = requests.get(zipfile_url)
#             # print(response.status_code)

#             # # Check if the request was successful (status code 200)
#             # if response.status_code != 200:
#             #     raise HTTPException(status_code=400, detail="Failed to download file.")

#             # Create a temporary directory to store the unzipped files
#             temp_dir = Path("temp/georisques")
#             temp_dir.mkdir(parents=True, exist_ok=True)

#             # # Save the downloaded ZIP file to the temporary directory
#             # zip_file_path = temp_dir / "AleaRG_Fxx_L93.zip"
#             # with open(zip_file_path, "wb") as f:
#             #     f.write(response.content)
#             # print("unzip success")

#             # # # Unzip the file
#             # with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
#             #     zip_ref.extractall(temp_dir)

#             # Read the .shp file from the URL using geopandas
#             shapefile_path = None
#             for file_path in temp_dir.iterdir():
#                 if file_path.is_file() and file_path.suffix == ".shp":
#                     shapefile_path = file_path
#                     print(shapefile_path)
#                     break

#             if shapefile_path is None:
#                 raise HTTPException(status_code=400, detail="No .shp file found in the uploaded ZIP.")

#             gdf = gpd.read_file(shapefile_path)
#             print("data read success")

#             # Change field types to string and rename columns
#             gdf.rename(columns={"DPT": "department_id","NIVEAU":"level","ALEA":"description"}, inplace=True)
#             gdf['department_id'] = gdf['department_id'].astype(str)
#             gdf['level'] = gdf['level'].astype(int)
#             gdf['description'] = gdf['description'].astype(str)
#             print("data changed success")
#             Session = sessionmaker(bind=engine_data)
#             session = Session()
#             gdf["geometry"] = gdf["geometry"].apply(lambda x: WKTElement(x.wkt, srid=4326))
#             gdf.to_sql("clay_mouvement", engine_data, if_exists="replace", index=False, dtype={"geometry": Geometry("GEOMETRY", srid=4326)})
#             session.commit()
#             session.close()

#             print ("Data imported successfully.")

#         if industrial_installations:
#             zipfile_url = "https://mapsref.brgm.fr/wxs/georisques/georisques_dl?&service=wfs&version=2.0.0&request=getfeature&typename=InstallationsClassees&outputformat=SHAPEZIP"
#             response = requests.get(zipfile_url)
#             print(response.status_code)

#             # # Check if the request was successful (status code 200)
#             if response.status_code != 200:
#                 raise HTTPException(status_code=400, detail="Failed to download file.")

#             # Create a temporary directory to store the unzipped files
#             temp_dir = Path("temp/georisques/industrial_installations")
#             temp_dir.mkdir(parents=True, exist_ok=True)

#             # # Save the downloaded ZIP file to the temporary directory
#             zip_file_path = temp_dir / "industrial_installations.zip"
#             with open(zip_file_path, "wb") as f:
#                 f.write(response.content)
#             print("unzip success")

#             # # # Unzip the file
#             with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
#                 zip_ref.extractall(temp_dir)

#             # Read the .shp file from the URL using geopandas
#             shapefile_path = None
#             for file_path in temp_dir.iterdir():
#                 if file_path.is_file() and file_path.suffix == ".shp":
#                     shapefile_path = file_path
#                     print(shapefile_path)
#                     break

#             if shapefile_path is None:
#                 raise HTTPException(status_code=400, detail="No .shp file found in the uploaded ZIP.")

#             gdf = gpd.read_file(shapefile_path)
#             Session = sessionmaker(bind=engine_data)
#             session = Session()
#             gdf["geometry"] = gdf["geometry"].apply(lambda x: WKTElement(x.wkt, srid=4326))
#             gdf.to_sql("industrial_installations", engine_data, if_exists="replace", index=False, dtype={"geometry": Geometry("GEOMETRY", srid=4326)})
#             session.commit()
#             session.close()

#             print ("Data industrial_installations imported successfully.")

#         if underground_cavity:
#             for dep in deps:
#                 if dep not in ('75','78','91','92','93','94','95'):
#                     csv_url = f"https://www.georisques.gouv.fr/webappReport/ws/cavites/departements/{dep}/fiches.csv?"
#                     response = requests.get(csv_url)
#                     if response.status_code != 200:
#                         raise HTTPException(status_code=400, detail="Failed to download file.")

#                     # Create a temporary directory to store the csv files
#                     temp_dir = Path("temp/georisques/underground_cavity")
#                     temp_dir.mkdir(parents=True, exist_ok=True)

#                     # # Save the downloaded ZIP file to the temporary directory
#                     csv_file_path = temp_dir / f"{dep}_underground_cavity.csv"
#                     with open(csv_file_path, "wb") as f:
#                         f.write(response.content)

#                     # Read the CSV file and create a GeoDataFrame
#                     df = pd.read_csv(csv_file_path, sep=';', encoding='utf-8')
#                     geometry = [Point(xy) for xy in zip(df['xouvl2e'], df['youvl2e'])]
#                     gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="IGNF:LAMBE")
#                     gdf = gdf.to_crs("EPSG:4326")

#                     Session = sessionmaker(autocommit=False, autoflush=False, bind=engine_data)
#                     session = Session()
#                     gdf.to_sql(
#                         name='underground_cavity',
#                         con=engine_data,
#                         if_exists='append',
#                         index=False
#                     )
#                     session.commit()
#                     session.close()

#                     print("Data underground_cavity imported successfully.")


#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     # finally:
#     #     # Clean up: remove temporary files and directory
#     #     for file_path in temp_dir.iterdir():
#     #         file_path.unlink()
#     #     temp_dir.rmdir()
