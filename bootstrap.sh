#!/bin/bash

# Install the latest version of pip
python3 -m pip install --upgrade pip

# Install the latest version of dependencies
pip install --no-cache-dir --upgrade -r requirements.txt

pip install gunicorn


# Add initial log entry when script starts
mkdir -p /app/logs

# Start a background process to update akshare periodically
(
  while true; do
    # Make update interval configurable. Unit is seconds, default is 1h.
    UPDATE_INTERVAL=${AKSHARE_UPDATE_INTERVAL:-3600}

    # Then use this variable
    sleep $UPDATE_INTERVAL

    echo "$(date): Starting akshare update process" >> /app/logs/akshare_updates.log

    # Get the current version of akshare before update
    OLD_VERSION=$(pip show akshare 2>/dev/null | grep "Version" | awk '{print $2}')
    
    # Attempt to upgrade akshare
    pip install --no-cache-dir --upgrade akshare 2>/dev/null
    UPDATE_STATUS=$?
    
    # Get the new version after update attempt
    NEW_VERSION=$(pip show akshare 2>/dev/null | grep "Version" | awk '{print $2}')
    
    if [ $UPDATE_STATUS -eq 0 ]; then
      if [ "$OLD_VERSION" != "$NEW_VERSION" ]; then
        echo "$(date): akshare successfully upgraded from $OLD_VERSION to $NEW_VERSION" >> /app/logs/akshare_updates.log
      fi
    else
      echo "$(date): akshare update failed with status $UPDATE_STATUS" >> /app/logs/akshare_updates.log
    fi
  done
) &

# Init or upgrade database.
flask db upgrade

# run Gunicorn with a specified number of workers and options
gunicorn -w ${GUNICORN_WORKERS:-2} 'app:create_app()' --bind ${FLASK_HOST:-127.0.0.1}:${FLASK_PORT:-8888} --timeout ${GUNICORN_TIMEOUT:-300} --max-requests ${GUNICORN_MAX_REQUESTS:-10} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-5}