FROM python:3.11-slim

WORKDIR /app

# Copy configuration and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application source code
COPY main.py .

# Run the FastAPI app on port 8080 (Cloud Run default)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
