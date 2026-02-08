source ~/.zshrc
source /home/astar/dart_ws/devel/setup.sh
sudo nmcli device set enp100s0 managed no
sudo ip addr add 192.168.1.100/24 dev enp100s0
