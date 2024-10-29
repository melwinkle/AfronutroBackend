FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install -y ca-certificates
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade certifi
RUN apt-get update && apt-get install -y iputils-ping dnsutils




COPY . .

# Add a script to run migrations and start the server
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["/app/entrypoint.sh"]