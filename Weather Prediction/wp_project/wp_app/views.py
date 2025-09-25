from django.shortcuts import render
import pandas as pd
import os
import joblib
from . models import History
path=os.path.dirname(__file__)
model=joblib.load(open(os.path.join(path,'weather2.pkl'),'rb'))
# Create your views here.
def index(req):
    return render(req,'index.html')

def prediction(req):
    if req.method == 'POST':
        # Get and convert user inputs to float
        precipitation = float(req.POST['precipitation'])
        max_temp = float(req.POST['max_temp'])
        min_temp = float(req.POST['min_temp'])
        wind_speed = float(req.POST['wind_speed'])
        # Define features list in the order your ML model expects
        features = ["precipitation", "max_temp", "min_temp", "wind_speed"]
        # Create DataFrame for prediction
        user_input = [precipitation, max_temp, min_temp, wind_speed]
        input_df = pd.DataFrame([user_input], columns=features)
        # Predict weather condition
        result = model.predict(input_df)[0]
        # Save in history model
        his = History(
            precipitation=precipitation,
            max_temp=max_temp,
            min_temp=min_temp,
            wind_speed=wind_speed,
            res=result)
        his.save()
        return render(req, "prediction.html", {"res": result})
    return render(req, "prediction.html")

def history(req):
    his=History.objects.all()
    return render(req,"history.html",{"his":his})

