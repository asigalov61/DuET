FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core libgl1 libegl1 libopengl0 libnss3 libasound2 libjack-jackd2-0 libxkbcommon0 libxcb-cursor0 libdbus-1-3 libpulse0 libxcomposite1 libxcursor1 libxi6 libxrandr2 libxtst6 libxkbcommon-x11-0 libxdamage1 libxrender1 libxss1 xvfb xauth && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir '.[audio]' gradio==5.49.1 spaces 'setuptools<81'
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
CMD ["python", "demo/app.py"]
