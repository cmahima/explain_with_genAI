import xgboost as xgb
import numpy as np
import shap


def get_shap_values_classification(X, model):
    print(model)
    model2 = xgb.Booster()
    model2.load_model(model)
    # X = pd.read_csv(X)
    Xd= xgb.DMatrix(X)
    pred= model2.predict(Xd, output_margin=True)
    explainer = shap.TreeExplainer(model2)
    shap_values = explainer.shap_values(Xd)
    # print("here", np.abs(np.array(shap_values).sum(1) + explainer.expected_value - pred).max())
    cols = X.columns
    shap_values= np.array(shap_values)
    m,n,p = shap_values.shape
    shap_values=shap_values.reshape(n,m,p)
    prediction = np.argmax(pred,1)
    feat_shap_values = shap_values[0][prediction]
    # print(shap_values[0].sum(1)+ explainer.expected_value) #should be equal to prediction
    return {f'{cols[i]}':f'{feat_shap_values[0][i]}' for i in range(feat_shap_values.shape[1])}, prediction[0]

def get_shap_values_regression(X, model):
    model2 = xgb.Booster()
    # model2.load_model("flight_model.json")
    model2.load_model(model)
    # X = pd.read_csv(X)
    Xd= xgb.DMatrix(X)
    pred= model2.predict(Xd, output_margin=True)
    explainer = shap.TreeExplainer(model2)
    shap_values = explainer.shap_values(Xd)
    # print(shap_values.sum(1)+ explainer.expected_value) #should be equal to prediction
    cols = X.columns
    shap_values= np.array(shap_values)
    return {f'{cols[i]}':f'{shap_values[0][i]}' for i in range(shap_values.shape[1])}, pred[0]
