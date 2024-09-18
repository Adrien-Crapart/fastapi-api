from typing import Any, List

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Response
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from app import crud, models, schemas
from app.api import deps

from PIL import Image, ImageOps
from datetime import datetime, timedelta
from io import BytesIO

router = APIRouter()

@router.post("/resize-image/")
async def resize_image(file: UploadFile, width: int = 100, height: int = 100):
    try:
        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Resize the image
        resized_image = image.resize((width, height))
        
        # Save the resized image to a temporary buffer
        output_buffer = BytesIO()
        resized_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crop")
async def crop_image(file: UploadFile, left: int, top: int, right: int, bottom: int):
    try:
        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        
        # Save the cropped image to a temporary buffer
        output_buffer = BytesIO()
        cropped_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rotate")
async def rotate_image(file: UploadFile, degrees: int = 90):
    try:
        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Rotate the image
        rotated_image = image.rotate(degrees)
        
        # Save the rotated image to a temporary buffer
        output_buffer = BytesIO()
        rotated_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mirror")
async def mirror_image(file: UploadFile):
    try:
        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Mirror the image horizontally
        mirrored_image = ImageOps.mirror(image)
        
        # Save the mirrored image to a temporary buffer
        output_buffer = BytesIO()
        mirrored_image.save(output_buffer, format="JPEG")
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/compress")
async def compress_image(file: UploadFile, quality: int = 80):
    """
    Here's how you can call this endpoint with different quality levels:

    * `Low Quality (e.g., quality=30)`: Lower quality, higher compression, smaller file size.
    * `Medium Quality (e.g., quality=80)`: A balance between quality and file size.
    * `High Quality (e.g., quality=100)`: Highest quality, minimal compression, larger file size.

    You can adjust the quality parameter according to your desired compression level when making requests to this endpoint.

    """
    try:
        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Compress the image at the specified quality level
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=quality)
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/change_format")
async def change_image_format(file: UploadFile, format: str = "JPEG"):
    """
    Change format of all images

    `Format`: "JPEG", "PNG", "GIF", "BMP", "TIFF"
    """
    try:
        # Supported image formats
        SUPPORTED_FORMATS = ["JPEG", "PNG", "GIF", "BMP", "TIFF"]

        # Read the uploaded image
        image = Image.open(BytesIO(await file.read()))
        
        # Validate the image format
        format = format.upper()
        if format not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        
        # Change the image format
        output_buffer = BytesIO()
        image.save(output_buffer, format=format)
        output_buffer.seek(0)
        
        return StreamingResponse(content=output_buffer, media_type=f"image/{format.lower()}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
