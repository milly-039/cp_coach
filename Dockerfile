# 1. Use the official AWS Lambda base image
FROM public.ecr.aws/lambda/python:3.10

# 2. Copy the requirements file into the AWS root folder
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the actual CP Coach brain
COPY query_coach.py ${LAMBDA_TASK_ROOT}

# 5. Point the AWS engine to the file and function
CMD ["query_coach.lambda_handler"]