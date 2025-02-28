import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase

MONGO_DETAILS = "mongodb://matchingproject:matchingproject@mongodb"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client['matchingdb']

def get_database() -> AsyncIOMotorDatabase:
    return database
