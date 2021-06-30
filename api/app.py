
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
# Implement a /predict endpoint
def get_pop(title,original_title,release_date,duration_min,description,budget,original_language,status,number_of_awards_won,number_of_nominations,has_collection,all_genres,top_countries,number_of_top_productions,available_in_english):
    model = joblib.load("model.joblib")
    X = [[original_title,
          title,
          release_date,
          duration_min,
          description,
          budget,
          original_language,
          status,
          number_of_awards_won,
          number_of_nominations,
          has_collection,
          all_genres,
          top_countries,
          number_of_top_productions,
          available_in_english]]
    columns=['original_title',
            'title',
            'release_date',
            'duration_min',
            'description',
            'budget',
            'original_language',
            'status',
            'number_of_awards_won',
            'number_of_nominations',
            'has_collection',
            'all_genres',
            'top_countries',
            'number_of_top_productions',
            'available_in_english']
    #print(columns)
    Xtopred = pd.DataFrame(X,columns=columns)
    prediction = model.predict(Xtopred)[0]
    return {'title':title,'popularity':prediction}
