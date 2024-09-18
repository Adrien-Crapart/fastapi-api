from typing import Any, List

from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Response, Form, Request
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import StreamingResponse

from app import crud, models, schemas
from app.api import deps

# import cairosvg
import zipfile
from io import BytesIO
from PIL import Image
from docx import Document
import weasyprint
import os
import PyPDF2
from tempfile import NamedTemporaryFile
import subprocess

router = APIRouter()

# Jinja2 templates setup
templates = Jinja2Templates(directory="app/templates")


@router.get("/export-pdf/")
async def export_pdf():
    # Data to populate the template
    data = {"title": "PDF Export Example",
            "last_update": "18-08-2023",
            "update": "25-08-2023"}

    # Render the HTML template with Jinja2
    template = templates.get_template("template.html")
    html_content = template.render(request=None, data=data)

    # Convert the HTML content to a PDF using weasyprint
    pdf = weasyprint.HTML(string=html_content).write_pdf()

    # Define response headers
    headers = {
        "Content-Disposition": "attachment; filename=exported.pdf",
        "Content-Type": "application/pdf",
    }

    # Return the PDF as a StreamingResponse
    return StreamingResponse(BytesIO(pdf), headers=headers)


@router.get("/export/")
async def export(
    format: str = Query(...,
                        description="Export format (pdf, jpg, png, bmp, docx, html, txt, xml)"),
):
    # Data to populate the template
    data = {"title": "Export Example",
            "content": "This is content for the export."}

    # Render the HTML template with Jinja2
    template = templates.get_template("template.html")
    html_content = template.render(request=None, data=data)

    if format == "pdf":
        # Convert the HTML content to a PDF using weasyprint
        pdf = weasyprint.HTML(string=html_content).write_pdf()
        content_type = "application/pdf"
    elif format in ["jpg", "png", "bmp"]:
        # Export the HTML content as an image using wkhtmltoimage
        image_format = format.upper()
        pdf = subprocess.check_output(
            ["wkhtmltoimage", "--format", image_format, "-", "-"], input=html_content.encode())
        content_type = f"image/{format}"
    elif format == "docx":
        # Create a DOCX document and insert the HTML content as a paragraph
        doc = Document()
        doc.add_paragraph(html_content)
        doc_byte_array = BytesIO()
        doc.save(doc_byte_array)
        pdf = doc_byte_array.getvalue()
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif format == "html":
        pdf = html_content.encode("utf-8")
        content_type = "text/html"
    elif format == "txt":
        pdf = html_content.encode("utf-8")
        content_type = "text/plain"
    elif format == "xml":
        pdf = html_content.encode("utf-8")
        content_type = "application/xml"
    else:
        raise HTTPException(status_code=400, detail="Invalid format specified")

    # Define response headers
    headers = {
        "Content-Disposition": f"attachment; filename=exported.{format}",
        "Content-Type": content_type,
    }

    # Return the exported content as a StreamingResponse
    return StreamingResponse(BytesIO(pdf), headers=headers)


@router.post("/merge-pdfs/")
async def merge_pdfs(
    pdf_files: list[UploadFile],
    order: list[int] = None
):
    """Merges a list of PDF files based on specified order priority.

    **Input:**

    - `pdf_files:` A list of uploaded PDF files to be merged.

    - `order:` (Optional) A list of integers representing the order in which PDFs should be merged.

    **Output:**

    - A downloadable PDF file containing the merged PDFs in the specified order.
    """

    try:
        if not pdf_files:
            raise HTTPException(
                status_code=400, detail="No PDF files provided for merging.")

        # Create a list to store the PDF data
        pdf_data = []

        # Read PDF data from the uploaded files
        for pdf_file in pdf_files:
            pdf_data.append(await pdf_file.read())

        # Create a PDF merger object
        pdf_merger = PyPDF2.PdfMerger()

        # If an order is specified, merge PDFs in that order
        if order and len(order) == len(pdf_data):
            for idx in order:
                if idx < 0 or idx >= len(pdf_data):
                    raise HTTPException(
                        status_code=400, detail="Invalid order index.")
                pdf_merger.append(BytesIO(pdf_data[idx]))
        else:
            # Merge PDFs in the order they were uploaded
            for pdf_content in pdf_data:
                pdf_merger.append(BytesIO(pdf_content))

        # Create a temporary merged PDF file
        with NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as merged_pdf_file:
            merged_pdf_filename = merged_pdf_file.name
            pdf_merger.write(merged_pdf_filename)

        return {"message": "PDFs merged successfully", "merged_pdf_filename": merged_pdf_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/divide-pdf/")
async def divide_pdf(
    pdf_file: UploadFile,
    page_ranges: str,
    pdf_names: str,
):
    """
    Divides a PDF into one-page PDFs and allows specifying pages or page ranges for extraction.

    - `pdf_file:` An uploaded PDF file to be divided.

    - `page_ranges:` A string specifying the pages or page ranges to extract (e.g., "1-3,5-7").

    - `pdf_names:` A name specifying after divide, must be the same order of ranges (e.g., "example_name1,example_name2,example_name3")
    """
    try:
        if not pdf_file:
            raise HTTPException(
                status_code=400, detail="No PDF file provided for division.")

        # Read the PDF data from the uploaded file
        pdf_data = await pdf_file.read()

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(BytesIO(pdf_data))
        total_pages = pdf_reader.numPages

        # Parse the specified page ranges
        divided_pdf_data = []

        user_name_list = pdf_names.split(",")
        for idx, page_range in enumerate(page_ranges.split(",")):
            start, end = map(int, page_range.split("-"))
            start = max(1, start)
            end = min(end, total_pages)

            pdf_writer = PyPDF2.PdfFileWriter()

            for page_number in range(start, end + 1):
                pdf_writer.addPage(pdf_reader.getPage(page_number - 1))

            output_pdf = BytesIO()
            pdf_writer.write(output_pdf)

            # Generate a name for the PDF based on the user's name (if available)
            user_name = user_name_list[idx] if idx < len(
                user_name_list) else f"PageRange_{start}-{end}"

            divided_pdf_data.append(
                (f"{user_name}.pdf", output_pdf.getvalue()))

        # Create a temporary directory to store divided PDFs
        with NamedTemporaryFile(delete=False, suffix=".zip") as zip_file:
            with zipfile.ZipFile(zip_file.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename, pdf_data in divided_pdf_data:
                    zipf.writestr(filename, pdf_data)

            zip_file_path = zip_file.name

        def file_iterator(file_path):
            with open(file_path, "rb") as file:
                while chunk := file.read(65536):
                    yield chunk

        # Define the response headers
        response_headers = {
            "Content-Disposition": f'attachment; filename="divided_pdfs.zip"'
        }

        # Return the ZIP archive as a StreamingResponse
        return StreamingResponse(
            content=file_iterator(zip_file_path),
            headers=response_headers,
            media_type="application/zip",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_pages(
    pages: str,
    total_pages: int
):
    page_numbers = []
    ranges = pages.split(",")
    for r in ranges:
        if "-" in r:
            start, end = map(int, r.split("-"))
            # Ensure start and end are within bounds
            start = max(1, start)
            end = min(end, total_pages)
            page_numbers.extend(range(start, end + 1))
        else:
            page_number = int(r)
            if 0 < page_number <= total_pages:
                page_numbers.append(page_number)
    return page_numbers


@router.post("/compress-pdf/")
async def compress_pdf(
    pdf_file: UploadFile,
    max_size_mb: float = 1.0
):
    """Compresses the size of a PDF file while ensuring it does not exceed a specified maximum size.

    Input:

    - `pdf_file:` An uploaded PDF file to be compressed.

    - `max_size_mb:` The maximum size in megabytes that the compressed PDF can be.

    Output:

    - A downloadable compressed PDF file if it meets the specified maximum size criteria.
    """
    try:
        if not pdf_file:
            raise HTTPException(
                status_code=400, detail="No PDF file provided for compression.")

        # Read the PDF data from the uploaded file
        pdf_data = await pdf_file.read()

        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfFileWriter()

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfFileReader(BytesIO(pdf_data))

        # Process each page and add it to the writer
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            pdf_writer.addPage(page)

        # Create a temporary file to store the compressed PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as compressed_pdf:
            pdf_writer.write(compressed_pdf)

        compressed_pdf_path = compressed_pdf.name

        # Check the size of the compressed PDF
        compressed_pdf_size_mb = os.path.getsize(
            compressed_pdf_path) / (1024 * 1024)

        # If the compressed PDF size is larger than the specified max size, raise an exception
        if compressed_pdf_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400, detail=f"Compressed PDF size exceeds {max_size_mb} MB")

        def file_iterator(file_path):
            with open(file_path, "rb") as file:
                while chunk := file.read(65536):
                    yield chunk

        # Define the response headers
        response_headers = {
            "Content-Disposition": f'attachment; filename="compressed.pdf"'
        }

        # Return the compressed PDF as a StreamingResponse
        return StreamingResponse(
            content=file_iterator(compressed_pdf_path),
            headers=response_headers,
            media_type="application/pdf",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
