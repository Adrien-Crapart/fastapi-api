# Create local virtual environment

`python3 -m venv env`

- ## In windows environment

  - `env\Scripts\activate.ps1`

- ## In linux environment

  - `source env/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# Load the api

`uvicorn app.main:app --reload`