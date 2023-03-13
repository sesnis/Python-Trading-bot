from pymongo import MongoClient
import logging
from mongolog.handlers import MongoHandler
from dotenv import dotenv_values

config = dotenv_values(".env")

def connect_db():
    try:
        client = MongoClient(config["DB_URL"])
        db = client['test']

        global collection
        collection = db['logging']

        print("Izveidots savienojums ar datubāzi")
    except Exception:
        print("Nevarēja izveidot savienojumu ar datubāzi")

def insert_data(data):
    result = collection.insert_one(data)