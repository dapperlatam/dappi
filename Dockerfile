FROM python:3.11.5-bookworm

RUN mkdir -p /home/api

COPY ./source/* /home/api

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /home/api/requirements.txt

#load .env variables
ENV PYTHONPATH /home/api

CMD ["uvicorn", "home.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose the port the app runs on
EXPOSE 8080
