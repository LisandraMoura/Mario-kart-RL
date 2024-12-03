# Disclaimer

This work has been created solely for academic purposes as part of the "Reinforcement Learning" class. We do not claim ownership of any content presented here. All materials and content are the intellectual property of their respective creators.

# Mario Kart A3C Agent Disclaimer

Forked/branched from the [universe-starter-agent](https://github.com/openai/universe-starter-agent). Several tweaks have been made to allow this agent to work with the [gym-mupen64plus](https://github.com/bzier/gym-mupen64plus) N64 environment (specifically for the MarioKart64 game). This initial set of commits is currently unrefined/hacky/sloppy and needs significant cleanup. However, it is functional at this point.

# Training a Reinforcement Learning Agent to Play Mario Kart 64

## Setup
1. [Clone the Emulator Repository](https://github.com/mendesLet/gym-mupen64plus)
2. Build the Emulator Docker Image

3. Clone this repository
4. Update the Agent's Dockerfile
    
    In the Agent's Dockerfile, replace the FROM statement with:
    ```bash
    FROM bz/gym-mupen64plus:0.0.5
    ```

5. Build the Agent Docker Image

    ```bash
    ./docker/build.sh
    ```
    - If the build doesn't work make sure the .sh file is executable ```chmod +x <file_name>```

6. Create a ```.env``` file at the ```./utils``` of the Agent repository with the following variables:

    ```bash
    IMAGE_SPEC=bz/mario-kart-a3c:0.0.5
    LOCAL_ROM_PATH=/path/to/your/rom
    ```

## Training

1. Start the training process using Docker Compose:

    ```bash
    docker compose -f docker/docker-compose.yml --env-file utils/.env -p mario-kart-agent up -d
    ```
    - You can close the docker by using the down command ```docker compose -f docker/docker-compose.yml --env-file utils/.env -p mario-kart-agent down```
    - To monitor the training progress and view TensorBoard, access ```http://localhost:12345``` in your web browser.

## Visualization

1. [Clone the noVNC Repository](https://github.com/novnc/noVNC)
2. Connect to the VNC Server

    ```bash
    ./utils/novnc_proxy --vnc localhost:32775
    ```
    - Replace ```32775``` with the actual port exposed by your Docker container. This will open a web interface where you can visually observe the agent playing Mario Kart 64.

## Checkpoints

1. Check for the location of your logs using `docker volume ls`, you should see a volume named `mario-kart-agent_mklogs`
2. Run `docker volume inspect mario-kart-agent_mklogs` and locate the folder it is in
- If you want to start from scratch, just delete the volume. The code should automatically use the latest weights in the volume to resume training when you use docker compose up
