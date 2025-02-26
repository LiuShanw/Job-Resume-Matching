# Code Citations

## License: unknown
https://github.com/amiradridi/Job-Resume-Matching/tree/49680ad1be68e0ff144e03f628dddc13ba944cc3/main.py

```
transforms the dataframe into json
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)
    return json_data

app = FastAPI()

@app.get("/extraction")
async
```


## License: unknown
https://github.com/famousyub/jobmatcherz/tree/6d13d46d36b385044e7b8a093b0308aa73b50b87/jobresume/main.py

```
', index_col=0)
    jobs = jobs[['Qualifications']]
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    jobs = job_extraction.extract_entities(jobs)
    for i, row in jobs.iterrows():
        minimum_degree_level = jobs['Minimum degree level
```


## License: unknown
https://github.com/amiradridi/Job-Resume-Matching/tree/49680ad1be68e0ff144e03f628dddc13ba944cc3/services/Rules.py

```
= ast.literal_eval(resumes["degrees"][i])
    for i in range(len(resumes["skills"])):
        resumes["skills"][i] = ast.literal_eval(resumes["skills"][i])
```


## License: Apache_2_0
https://github.com/ha5minh2duc/Thesis-Job-Resume-Matching-Shorlister-and-Latent-Topics-Mining/tree/2a069a9df7fca284ab69062cd9aa93c765dfb60a/Job_resume_matching/matching.py

```
def modifying_type_resume(resumes):
    for i in range(len(resumes["degrees"])):
        resumes["degrees"][i] = ast.literal_eval(resumes["degrees"][i])
    for i in range(len(resumes[
```

