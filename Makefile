.PHONY: start shutdown server client

# Environment paths
CONDA_BASE   = /home/hsb/miniforge3
CONDA_INIT   = $(CONDA_BASE)/etc/profile.d/conda.sh
SERVER_ENV   = py310
CLIENT_ENV   = lerobot_py310
ROS_SETUP    = /opt/ros/humble/setup.bash

# Common shell preamble: load conda + scrub firewall + bypass proxy for the robot.
PREAMBLE = source $(CONDA_INIT) && \
	{ sudo ufw disable 2>/dev/null || true; } && \
	export NO_PROXY=10.192.1.2,localhost,127.0.0.1 && \
	export no_proxy=10.192.1.2,localhost,127.0.0.1

start:
	@echo "=== Tron2 warmup: [0]*14 -> WP3 ==="
	bash -c '$(PREAMBLE) && \
		conda activate $(SERVER_ENV) && \
		source $(ROS_SETUP) && \
		python start.py'

shutdown:
	@echo "=== Tron2 shutdown: WP3 -> [0]*14 ==="
	bash -c '$(PREAMBLE) && \
		conda activate $(SERVER_ENV) && \
		source $(ROS_SETUP) && \
		python shutdown.py'

server:
	@echo "=== Tron2 server (py310 + ROS2) ==="
	bash -c '$(PREAMBLE) && \
		conda activate $(SERVER_ENV) && \
		source $(ROS_SETUP) && \
		python server.py'

client:
	@echo "=== Tron2 client (lerobot_py310) ==="
	bash -c '$(PREAMBLE) && \
		conda activate $(CLIENT_ENV) && \
		python client.py'
