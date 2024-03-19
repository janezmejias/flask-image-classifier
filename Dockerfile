FROM python:3.8-slim

WORKDIR /app

COPY predict.py /app/
COPY requirements.txt /app/
COPY fit_categorical_model.h5 /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "predict:app"]