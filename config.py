import os

class ServerConfig:#Production
    DEBUG = False
    TESTING = False
    HOST = '0.0.0.0'
    PORT = 8080

class Development(ServerConfig):
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000

class BotConfig:
    SECRET_KEY = os.getenv("SECRET")
    CLIENT_ID = os.getenv("CLIENT_ID")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
