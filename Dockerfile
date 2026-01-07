FROM python:3.12

ENV PYTHONUNBUFFERED=1

WORKDIR /garbage_app

RUN apt-get update && apt-get install -y  --no-install-recommends  \
    libsndfile1  \
    gcc \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==2.2.2+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt


COPY main.py .
COPY model_garbage.pth .
COPY classes.pth .

EXPOSE 8000

CMD ["uvicorn", "main:garbage_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]