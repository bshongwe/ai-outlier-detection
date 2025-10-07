import os

bind = f"0.0.0.0:{os.getenv('API_PORT', 8000)}"
workers = int(os.getenv('WORKERS', 4))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = 100
timeout = int(os.getenv('TIMEOUT', 60))
keepalive = 5
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()