from fastapi import FastAPI, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
import pandas as pd
import json
from services.JobInfoExtraction import JobInfoExtraction
from services.Rules import Rules
from source.db_helpers.db_connection import get_database
from source.schemas.matched_resume import ResumeMatchedModel
from source.schemas.jobextracted import JobExtractedModel
import ast
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from bson import ObjectId
from pydantic import BaseModel, Field
import os
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_json_schema__(cls, schema):
        schema.update(type="string")
        return schema

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class ExampleClass:
    @classmethod
    def __get_pydantic_json_schema__(cls, schema):
        schema.update(type="string")
        return schema

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        # validation logic
        pass

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Placeholder for Gemini 1.5 Pro model initialization
# Assuming you have a function to load the Gemini 1.5 Pro model
def load_gemini_model():
    # Load and return the Gemini 1.5 Pro model
    pass

gemini_model = load_gemini_model()

def transform_dataframe_to_json(dataframe: pd.DataFrame) -> str:
    """
    Transforms a pandas DataFrame to a JSON string.
    """
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)
    return json_data

def modifying_type_resume(resumes: pd.DataFrame) -> pd.DataFrame:
    """
    Modifies the types of the 'degrees' and 'skills' columns in the resumes DataFrame.
    """
    for i in range(len(resumes["degrees"])):
        resumes["degrees"][i] = ast.literal_eval(resumes["degrees"][i])
    for i in range(len(resumes["skills"])):
        resumes["skills"][i] = ast.literal_eval(resumes["skills"][i])
    return resumes

def modifying_type_job(jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Modifies the types of the 'Skills' column in the jobs DataFrame.
    """
    for i in range(len(jobs["Skills"])):
        jobs["Skills"][i] = ast.literal_eval(jobs["Skills"][i])
    return jobs

class ExtractionRequest(BaseModel):
    degrees_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/degrees.jsonl')
    majors_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/majors.jsonl')
    skills_patterns_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/skills.jsonl')
    jobs_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/job descriptions.csv')

class MatchingRequest(BaseModel):
    labels_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/labels.json')
    job_desc_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/job_description_by_spacy.csv')
    resumes_path: str = Field(default='C:/Users/Moon/Downloads/Job-Resume-Matching-master/Resources/data/resumes_by_spacy.csv')

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extraction")
async def extraction(request: ExtractionRequest, database: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Extracts job information and stores it in the database.
    """
    try:
        jobs = pd.read_csv(request.jobs_path, index_col=0)
        jobs = jobs[['Qualifications']]
        job_extraction = JobInfoExtraction(request.skills_patterns_path, request.majors_patterns_path, request.degrees_patterns_path, jobs)
        jobs = job_extraction.extract_entities(jobs)
        for i, row in jobs.iterrows():
            minimum_degree_level = jobs['Minimum degree level'][i]
            acceptable_majors = jobs['Acceptable majors'][i]
            skills = jobs['Skills'][i]

            job_extracted = JobExtractedModel(minimum_degree_level=minimum_degree_level if minimum_degree_level else '',
                                              acceptable_majors=acceptable_majors if acceptable_majors else [],
                                              skills=skills if skills else [])
            job_extracted = jsonable_encoder(job_extracted)
            await database.get_collection("jobsextracted").insert_one(job_extracted)
        jobs_json = transform_dataframe_to_json(jobs)
        return jobs_json
    except Exception as e:
        logger.error(f"Error in extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/matching")
async def matching(request: MatchingRequest, database: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Matches resumes to job descriptions and stores the results in the database.
    """
    try:
        with open(request.labels_path) as fp:
            labels = json.load(fp)
        jobs = pd.read_csv(request.job_desc_path, index_col=0)
        resumes = pd.read_csv(request.resumes_path, index_col=0)
        resumes = modifying_type_resume(resumes)
        jobs = modifying_type_job(jobs)
        rules = Rules(labels, resumes, jobs)
        job_indexes = [0, 1, 2, 3, 4]
        resumes_matched_jobs = pd.DataFrame()
        for job_index in job_indexes:
            resumes_matched = rules.matching_score(resumes, jobs, job_index)
            resumes_matched_jobs = resumes_matched_jobs.append(resumes_matched)

            # adding matched resumes to database
            for i, row in resumes_matched.iterrows():
                id_resume = resumes_matched['_id'][i]
                degree_matching = float(resumes_matched['Degree job ' + str(job_index) + ' matching'][i])
                major_matching = float(resumes_matched['Major job ' + str(job_index) + ' matching'][i])
                skills_semantic_matching = float(resumes_matched['Skills job ' + str(job_index) + ' semantic matching'][i])
                matching_score = float(resumes_matched['matching score job ' + str(job_index)][i])
                matched_resume = ResumeMatchedModel(id_resume=id_resume if id_resume else '',
                                                    job_index=job_index if job_index else 0,
                                                    degree_matching=degree_matching if degree_matching else 0,
                                                    major_matching=major_matching if major_matching else 0,
                                                    skills_semantic_matching=skills_semantic_matching
                                                    if skills_semantic_matching else 0,
                                                    matching_score=matching_score if matching_score else 0)
                matched_resume = jsonable_encoder(matched_resume)
                await database.get_collection("matches").insert_one(matched_resume)

        resumes_matched_json = transform_dataframe_to_json(resumes_matched_jobs)
        return resumes_matched_json
    except Exception as e:
        logger.error(f"Error in matching endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/top_resumes")
async def top_resumes(database: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Retrieves the top 5 resumes with the highest matching scores.
    """
    try:
        top_resumes = database.matches.find().sort("matching_score", -1).limit(5)
        result = []
        top_resumes = await top_resumes.to_list(None)
        for x in top_resumes:
            result.append(x)
        top_resumes_json = jsonable_encoder(result)
        return top_resumes_json
    except Exception as e:
        logger.error(f"Error in top_resumes endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")