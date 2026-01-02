#!/bin/bash
# Auto-start 2048-bot training script

# Wait for internet connection
echo "Waiting for internet connection..."
while ! ping -c 1 google.com > /dev/null 2>&1; do
    sleep 2
done
echo "Internet connection established!"

# Wait a bit more for system to fully initialize
sleep 5

# Set up environment
export DISPLAY=:0
export GAME_RASPBERRY_PI=true

# Go to project directory
cd /home/nkasmanoff/Desktop/2048-bot

# Activate virtual environment (can deal with wrong repo another day)
source /home/nkasmanoff/Desktop/slither-bot/.venv/bin/activate

# Start training
echo "Starting 2048-bot PPO training with wandb logging..."
python train.py --algorithm ppo --episodes 1000 --wandb --wandb-project 2048-raspberry-pi