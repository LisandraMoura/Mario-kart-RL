#!/bin/bash

# Run our worker process
CUDA_VISIBLE_DEVICES= python ./scripts/worker.py --log-dir /tmp/marioKartLogs --env-id Mario-Kart-Discrete-Luigi-Raceway-v0 --job-name worker --remotes 1 >/tmp/marioKartLogs/a3c.w-`hostname`.out 2>&1 &

# Wait for Xvfb to begin running
until ps aux | grep -i "[x]vfb" > /dev/null; do
    sleep 1
done

# Now that Xvfb is up, connect a VNC server to its display
x11vnc -display :0 -forever -viewonly -rfbport 5900 -shared &

# Tail the worker process output indefinitely
tail -f /tmp/marioKartLogs/a3c.w-`hostname`.out
