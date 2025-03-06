# Block 2: Imports and Configuration
import os
import json
import ast
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import core_schema
from bson import ObjectId
from typing import Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from transformers import BertTokenizer, BertModel
from fastapi.encoders import jsonable_encoder  # Add this import

# Allow running async code in Jupyter
import nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Resume Filtering API is running!"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection setup
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "resume_matcher_db"

# Initialize MongoDB client
client = AsyncIOMotorClient(MONGODB_URL)
database = client[DATABASE_NAME]

# Dependency to get the database instance
async def get_database() -> AsyncIOMotorDatabase:
    return database

# Block 3: Pydantic Models
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

class ResumeMatchedModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    id_resume: str = Field(...)
    job_index: int = Field(...)
    degree_matching: float = Field(...)
    major_matching: float = Field(...)
    skills_semantic_matching: float = Field(...)
    matching_score: float = Field(...)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class JobExtractedModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    minimum_degree_level: str = Field(...)
    acceptable_majors: List[str] = Field(...)
    skills: List[str] = Field(...)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Block 4: Model Initialization
# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Placeholder for Gemini 1.5 Pro model initialization
def load_gemini_model():
    # Load and return the Gemini 1.5 Pro model
    # Example: return GeminiModel.from_pretrained('gemini-1.5-pro')
    pass

gemini_model = load_gemini_model()

# Block 5: Utility Functions
def transform_dataframe_to_json(dataframe: pd.DataFrame) -> str:
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)
    return json_data

def modifying_type_resume(resumes: pd.DataFrame) -> pd.DataFrame:
    try:
        for i in range(len(resumes["degrees"])):
            resumes["degrees"][i] = ast.literal_eval(resumes["degrees"][i])
        for i in range(len(resumes["skills"])):
            resumes["skills"][i] = ast.literal_eval(resumes["skills"][i])
        logger.info("Resume data modified successfully.")
        return resumes
    except Exception as e:
        logger.error(f"Error in modifying_type_resume: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing resumes: {e}")

def modifying_type_job(jobs: pd.DataFrame) -> pd.DataFrame:
    try:
        for i in range(len(jobs["Skills"])):
            jobs["Skills"][i] = ast.literal_eval(jobs["Skills"][i])
        logger.info("Job data modified successfully.")
        return jobs
    except Exception as e:
        logger.error(f"Error in modifying_type_job: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing jobs: {e}")

# Block 6: Request Models
class ExtractionRequest(BaseModel):
    degrees_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/degrees.jsonl')
    majors_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/majors.jsonl')
    skills_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/skills.jsonl')
    jobs_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/job descriptions.csv')

class MatchingRequest(BaseModel):
    labels_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/labels.json')
    job_desc_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/job_description_by_spacy.csv')
    resumes_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/resumes_by_spacy.csv')

# Block 7: API Endpoints
@app.post("/extraction")
async def extraction(request: ExtractionRequest, database: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Extracts job information and stores it in the database.
    """
    try:
        logger.info("Starting extraction process...")
        logger.info(f"Request payload: {request}")  # Debug log: Print the request payload
        
        if not os.path.exists(request.jobs_path):
            logger.error(f"Jobs file not found: {request.jobs_path}")  # Debug log: Print the missing file path
            raise HTTPException(status_code=400, detail="Jobs file not found.")
        
        jobs = pd.read_csv(request.jobs_path, index_col=0)
        logger.info(f"Loaded jobs data: {jobs.shape[0]} rows")
        jobs = jobs[['Qualifications']]
        
        logger.info("Extracting entities...")  # Debug log: Indicate entity extraction is starting
        job_extraction = JobInfoExtraction(request.skills_patterns_path, request.majors_patterns_path, request.degrees_patterns_path, jobs)
        jobs = job_extraction.extract_entities(jobs)
        logger.info("Entities extracted successfully.")
        
        for i, row in jobs.iterrows():
            minimum_degree_level = jobs['Minimum degree level'][i]
            acceptable_majors = jobs['Acceptable majors'][i]
            skills = jobs['Skills'][i]

            job_extracted = JobExtractedModel(
                minimum_degree_level=minimum_degree_level if minimum_degree_level else '',
                acceptable_majors=acceptable_majors if acceptable_majors else [],
                skills=skills if skills else []
            )
            job_extracted = jsonable_encoder(job_extracted)
            await database.get_collection("jobsextracted").insert_one(job_extracted)
        
        jobs_json = transform_dataframe_to_json(jobs)
        logger.info("Extraction process completed successfully.")
        return jobs_json
    except Exception as e:
        logger.error(f"Error in extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Block 8: Server Startup
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Block 9: Test the Server
import requests

# Define the request payload
payload = {
    "degrees_patterns_path": "C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/degrees.jsonl",
    "majors_patterns_path": "C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/majors.jsonl",
    "skills_patterns_path": "C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/skills.jsonl",
    "jobs_path": "C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/job descriptions.csv"
}

# Make a POST request to the /extraction endpoint
# response = requests.post("http://localhost:8000/extraction", json=payload)

# Print the response
# print(response.status_code)
# print(response.json())

# Block 10: Verify the Database
from motor.motor_asyncio import AsyncIOMotorClient

# Connect to MongoDB
client = AsyncIOMotorClient("mongodb://localhost:27017")
database = client["resume_matcher_db"]
collection = database["jobsextracted"]

# Query the collection
async def fetch_data():
    documents = await collection.find().to_list(length=10)
    return documents

# Run the query
import asyncio
data = asyncio.run(fetch_data())
print(data)

import pandas as pd
import json
import re

class JobInfoExtraction:
    def __init__(self, skills_patterns_path: str, majors_patterns_path: str, degrees_patterns_path: str, jobs: pd.DataFrame):
        self.skills_patterns = self.load_patterns(skills_patterns_path)
        self.majors_patterns = self.load_patterns(majors_patterns_path)
        self.degrees_patterns = self.load_patterns(degrees_patterns_path)
        self.jobs = jobs

    def load_patterns(self, path: str) -> List[str]:
        with open(path, 'r') as file:
            patterns = [line.strip() for line in file]
        return patterns

    def extract_entities(self, jobs: pd.DataFrame) -> pd.DataFrame:
        jobs['Minimum degree level'] = jobs['Qualifications'].apply(self.extract_degree)
        jobs['Acceptable majors'] = jobs['Qualifications'].apply(self.extract_majors)
        jobs['Skills'] = jobs['Qualifications'].apply(self.extract_skills)
        return jobs

    def extract_degree(self, text: str) -> str:
        for pattern in self.degrees_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        return ''

    def extract_majors(self, text: str) -> List[str]:
        majors = []
        for pattern in self.majors_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                majors.append(pattern)
        return majors

    def extract_skills(self, text: str) -> List[str]:
        skills = []
        for pattern in self.skills_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(pattern)
        return skills