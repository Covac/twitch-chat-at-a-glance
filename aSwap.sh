# This is a bandage fix to circumvent low RAM on fly.io free tier
fallocate -l 256M /swapfile
chmod 0600 /swapfile
mkswap /swapfile
# Default vm.swappiness is on 60
# Enable swap
swapon /swapfile
python3 TwitchChatAtAGlance.py