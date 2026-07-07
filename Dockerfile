FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY streamlit_app.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
