#!/bin/sh


echo "Stopping all Gunicorn..."
sudo systemctl stop gunicorn.socket
sudo systemctl disable gunicorn.socket
sudo service gunicorn stop

echo "Stopping all Nginx..."
sudo service nginx stop
sudo service redis stop

echo "Stopping all Celery..."
sudo systemctl stop celery
sudo systemctl disable celery
