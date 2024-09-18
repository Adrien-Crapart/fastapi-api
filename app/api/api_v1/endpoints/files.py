from typing import Any, List

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Response, Form
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse

from app import crud, models, schemas
from app.api import deps
import shutil
import os
from datetime import datetime, timedelta
import zipfile
from io import BytesIO

router = APIRouter()


@router.post("/upload/")
async def upload_file(
    upload_file: UploadFile = File(...),
    folder_name: str = "default"
):
    """Upload a file into an existing folder or created folder

    **Input:**

    - `folder_name:` Specify the name of folder

    - `folder_name:` Specify the name of folder

    **Output:**

    - A value to get status in json format
    """
    try:
        path_folder = f"files/{folder_name}"

        # Ensure the "uploads" directory exists
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        # Save the uploaded file to the "uploads" directory
        file_path = os.path.join(path_folder, upload_file.filename)
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return {"message": "File uploaded successfully",
                'filename': file_path,
                'type': upload_file.content_type
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_name}")
async def download_file(
    file_name: str,
    folder_name: str = "default"
):
    """Download a file into an existing folder, caution this file must be existing ! For more information, please get this router : "/files/list_files"

    **Input:**

    - `file_name:` Specify the name of file

    - `folder_name:` Specify the name of folder

    **Output:**

    - Download your file on any format
    """
    try:
        path_folder = f"files/{folder_name}"

        # Ensure the requested file exists in the "uploads" directory
        file_path = os.path.join(path_folder, file_name)
        if os.path.exists(file_path):
            return FileResponse(file_path, headers={"Content-Disposition": f"attachment; filename={file_name}"})
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{file_name}")
async def delete_file(
    file_name: str,
    folder_name: str = "default"
):
    """
    Delete one file in existing folder

    **Examples**

    * `folder_name`: 'images'

    * `file_name`: 'example.jpg'
    """
    try:
        path_folder = f"files/{folder_name}"

        # Ensure the requested file exists in the "uploads" directory
        file_path = os.path.join(path_folder, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_file_history(
    db: Session,
    file_name: str,
    operation: str,
    user: str = "system"
):
    db_file_history = models.FileHistory(
        file_name=file_name,
        operation=operation,
        user=user,
    )
    db.add(db_file_history)
    db.commit()
    db.refresh(db_file_history)


def create_trash_record(db: Session, file_name: str, operation: str, user: str = "system"):
    delete_after = datetime.now() + timedelta(days=14)  # 14 days retention
    db_trash = models.Trash(
        file_name=file_name,
        operation=operation,
        user=user,
        delete_after=delete_after,
    )
    db.add(db_trash)
    db.commit()
    db.refresh(db_trash)


@router.get("/list_files/")
async def list_files(folder_name: str = "default", file_extension: str = Query(None, description="Filter files by extension")):
    """
    Get all files in existing folder

    **Examples**

    * `folder_name`: 'images'

    * `file_extension`: '.jpg'
    """
    try:
        path_folder = f"files/{folder_name}"

        # List all files in the "uploads" directory
        files = os.listdir(path_folder)

        # Filter files by extension if a filter is provided
        if file_extension:
            filtered_files = [
                file for file in files if file.endswith(file_extension)]
        else:
            filtered_files = files

        # Get file information for each file
        file_info_list = [get_file_info(os.path.join(
            path_folder, file_name)) for file_name in filtered_files]

        return {"files": file_info_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_file_info(file_path):
    # Get file size (weight) in bytes
    file_size_bytes = os.path.getsize(file_path)

    # Convert file size to octets, MB, or GB
    if file_size_bytes < 1024:
        file_size = f"{file_size_bytes} octets"
    elif file_size_bytes < 1024 ** 2:
        file_size = f"{file_size_bytes / 1024:.2f} Ko"
    elif file_size_bytes < 1024 ** 3:
        file_size = f"{file_size_bytes / (1024 ** 2):.2f} Mo"
    else:
        file_size = f"{file_size_bytes / (1024 ** 3):.2f} Go"

    path_folder = os.path.dirname(file_path)

    # Get file creation time
    creation_time = datetime.datetime.fromtimestamp(
        os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

    # Get last modified time
    modified_time = datetime.datetime.fromtimestamp(
        os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

    # Get file owner's username (platform-independent)
    file_owner = None
    try:
        file_owner = os.path.basename(os.path.expanduser(file_path))
    except Exception as e:
        pass  # Handle exceptions if unable to get the owner's username

    return {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": file_size,
        "created_date": creation_time,
        "last_modified_date": modified_time,
        "file_folder": path_folder,
        "file_owner": file_owner,
    }


@router.put("/rename/{old_name}/{new_name}")
async def rename_file_or_folder(old_name: str, new_name: str):
    """
    Rename existing folder or file in API Files

    **Examples**

    * `For a folder`: 'images'

    * `For a file`: 'images\example.jpg' (Respect the backslash)
    """
    try:
        path_folder = f"files"

        # Ensure the source file or folder exists in the "uploads" directory
        old_path = os.path.join(path_folder, old_name)
        if os.path.exists(old_path):
            new_path = os.path.join(path_folder, new_name)
            os.rename(old_path, new_path)
            return {"message": f"Renamed '{old_name}' to '{new_name}'"}
        else:
            raise HTTPException(
                status_code=404, detail="Source file or folder not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compress/")
async def compress_files(file_paths: list, output_filename: str = Query(..., description="Name of the compressed file")):
    """
    Export a zipfile from the API files

    `file_paths`: List of files with folder name (e.g : 'images/example.jpg')

    `output_filename`: Name of exported zipfile (e.g : 'data')
    """
    try:
        path_folder = "files"

        # Ensure the "uploads" directory exists
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        # Validate and ensure all specified file paths exist within the "uploads" directory
        valid_paths = []
        for file_path in file_paths:
            full_path = os.path.join(path_folder, file_path)
            if os.path.exists(full_path):
                valid_paths.append(full_path)

        if not valid_paths:
            raise HTTPException(
                status_code=400, detail="No valid files selected for compression")

        # Create a ZIP file in memory
        in_memory_zip = BytesIO()
        with zipfile.ZipFile(in_memory_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for item_path in valid_paths:
                arcname = os.path.relpath(item_path, path_folder)
                zipf.write(item_path, arcname=arcname)

        # Set the Content-Disposition header for downloading the ZIP file
        response = Response(content=in_memory_zip.getvalue())
        response.headers["Content-Disposition"] = f'attachment; filename="{output_filename}.zip"'
        response.media_type = "application/zip"

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export-zip/")
async def export_zip(
    selected_files: list = Form(...)
):
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)

    zip_filename = "exported_files.zip"
    zip_path = os.path.join(export_dir, zip_filename)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for selected_file in selected_files:
            file_path = os.path.join("images", selected_file)
            zipf.write(file_path, os.path.basename(file_path))

    return FileResponse(zip_path)
