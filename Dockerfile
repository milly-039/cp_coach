FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY query_coach.py ${LAMBDA_TASK_ROOT}

CMD ["query_coach.lambda_handler"]