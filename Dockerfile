FROM python:3.10-slim

WORKDIR /app

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and build it
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

# Copy the built frontend to static directory
RUN cp -r dist ../static

# Copy backend code
WORKDIR /app
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/ ./models/
COPY utils/ ./utils/
COPY baselines/ ./baselines/
COPY evaluation/ ./evaluation/
COPY scripts/ ./scripts/
COPY data/ ./data/

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]