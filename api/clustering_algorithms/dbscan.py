#Import necessary libraries
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
from sklearn import metrics
from sklearn.cluster import DBSCAN
import json


@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

def dbscan_cluster(X, eps, min_samples):
    
    model = DBSCAN(eps, min_samples)
    # fit model and predict clusters
    y_db = model.fit_predict(X)
    silhouette_score=metrics.silhouette_score(X, y_db)

    #pickle.dump(y_db,open("ml_model/dbscan_result.pkl", "wb"))

    return y_db, silhouette_score

@api_view(["POST"])
def get_dbscan(request):
    try:

        data = json.loads(request.body)
        #print("data", data)
         # data contains three fileds: eps: epsilon, min_samples : minimum number of elements within epsilon distance
         # train: matrix(size: m*n, m = # row, n = # column)
        eps = data['eps']
        min_samples = data['min_samples']
        train_data = data['train']
        if eps is not None:
            #Datapreprocessing Convert the values to float
            eps = float(eps)
            #print("eps",eps)
            min_samples = int(min_samples)
            #print("min_samples",min_samples)
            train_data = np.array(train_data)
            #print("train_data",train_data)
            y_db, silhouette_score = dbscan_cluster(train_data,eps, min_samples)
            result = {
                'error' : '0',
                'message' : 'Successfull',
                'y_db' : y_db.reshape(-1,1),
                'silhouette_score' : silhouette_score
            }
        else:
            result = {
                'error' : '1',
                'message': 'Invalid Parameters'                
            }
    except Exception as e:
        result = {
            'error' : '2',
            "message": str(e)
        }
    return Response(result)