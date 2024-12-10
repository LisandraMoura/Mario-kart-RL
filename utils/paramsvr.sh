#!/bin/bash

# Run our parameter server
CUDA_VISIBLE_DEVICES= python ./scripts/worker.py --log-dir /tmp/marioKartLogs --env-id Mario-Kart-Discrete-Moo-Moo-Farm-v0 --job-name ps >/tmp/marioKartLogs/a3c.ps.out 2>&1 &

# Fire up TensorBoard (perhaps should run in it's own container)
tensorboard --logdir /tmp/marioKartLogs --port 12345 >/tmp/marioKartLogs/a3c.tb.out 2>&1 &

# Tail the parameter server process output indefinitely
tail -f /tmp/marioKartLogs/a3c.ps.out
