import xgboost
import pandas as pd

def train_iris():
    data = pd.read_csv('./train_data/iris.csv')
    map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    for i in range(data.shape[0]):
        data['Species'].iloc[i]= map[data['Species'].iloc[i]]
    data['Species'] = data['Species'].astype("int")
    X, y = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']]

    model = xgboost.XGBClassifier()
    model.fit(X, y)
    model.save_model("iris_model.json")
    
def train_flights():
    df = pd.read_csv('./train_data/flights.csv')[0:3000]
    X,y = df[['DayofMonth', 'DayOfWeek', 'OriginAirportID','DestAirportID', 'DepDelay']],df[['ArrDelay']]
    model = xgboost.XGBRegressor()
    model.fit(X, y)
    model.save_model("flight_model.json")
train_flights()
train_iris()