#!/bin/sh

# Wait for the database to be ready
echo "Waiting for database..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 0.1
done
echo "Database started"

# Run migrations
echo "Running migrations..."
python manage.py makemigrations
python manage.py migrate

# Start server
echo "Starting server..."
gunicorn core.wsgi:application --bind 0.0.0.0:8000