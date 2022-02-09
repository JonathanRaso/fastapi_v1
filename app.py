# # 1. Library imports
# import uvicorn
# from fastapi import FastAPI

# # 2. Create the app object
# app = FastAPI()

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     '''
#     This is a first docstring.
#     '''
#     return {'message': 'Hello, stranger'}

# # 4. Route with a single parameter, returns the parameter within a message
# #    Located at: http://127.0.0.1:8000/AnyNameHere
# @app.get('/{name}')
# def get_name(name: str):
#     '''
#     This is a second docstring.
#     '''
#     return {'message': f'Hello, {name}'}

# # 5. Run the API with uvicorn
# #    http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc to see the doc page
# #    uvicorn app:app --reload  --> Run this command to start server
# #    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)


########################################

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Model import IrisModel, IrisSpecies

# 2. Create app and model objects
app = FastAPI()
model = IrisModel()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)