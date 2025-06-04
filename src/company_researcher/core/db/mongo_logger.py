from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from datetime import datetime

class MongoLogger:
    def __init__(self):
        uri = os.getenv("MONGO_URI")
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client["company_research"]
        self.collection = self.db["research_logs"]

    def log_result(self, company_name: str, company_url: str, result: dict):
        self.collection.insert_one({
            "company_name": company_name,
            "company_url": company_url,
            "result": result,
            "timestamp": datetime.utcnow()
        })