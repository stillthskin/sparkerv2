#!/bin/bash

# Reload systemd manager configuration (only needed if there's a change to unit files)
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Start and enable gunicorn.socket
echo "Starting and enabling gunicorn socket..."
sudo systemctl stop gunicorn.socket
sudo systemctl start gunicorn.socket
sudo systemctl enable gunicorn.socket

# Restart gunicorn and nginx services
echo "Restarting gunicorn and nginx services..."
sudo service gunicorn stop
sudo service gunicorn start
sudo service nginx stop
sudo service nginx start
sudo service  redis stop
sudo service redis start

# Start and enable celery
echo "Starting and enabling celery..."
sudo systemctl stop celery
sudo systemctl start celery
sudo systemctl enable celery

echo "Script execution complete."
