from pymongo import MongoClient
import gridfs
import os
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["legal_cases"]

fs = gridfs.GridFS(db)


def upload_case_file(case_id, file_bytes):

    file_id = fs.put(file_bytes, filename=f"{case_id}.pdf")

    return str(file_id)


def get_case_file(file_id):

    file = fs.get(ObjectId(file_id))

    return file.read()