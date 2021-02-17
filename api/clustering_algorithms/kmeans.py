from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import json

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

def kmeans_cluster(X, n_clusters):

    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    silhouette_score=metrics.silhouette_score(X, y_kmeans)
    
    return y_kmeans, kmeans.inertia_, silhouette_score

@api_view(["POST"])
def get_kmeans(request):
    try:
        # load request from json
        data = json.loads(request.body)
        #print("data", data)        
        # data contains two fileds: k: number of cluster, train: matrix(size: m*n)
        k = data['k']
        train_data = data['train']
        #train_data = request.GET.get('data')
        if k is not None:
            # Datapreprocessing Convert the values to float
            k = int(k)
            #print("k",k)
            train_data = np.array(train_data)
            train_data = list(filter(any,train_data))
            # Filtering the rows which contains None 
            train_data = [list(filter(None, lst)) for lst in train_data]
            #print("train_data",train_data)
            y_kmeans, ssd_kmeans, silhouette_score = kmeans_cluster(train_data,k)
            result = {
                'error' : '0',
                'message' : 'Successfull',
                'y_kmeans' : y_kmeans.reshape(-1,1),
                'ssd' : ssd_kmeans,
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
