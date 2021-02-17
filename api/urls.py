from django.urls import path
from api.clustering_algorithms import kmeans
from api.clustering_algorithms import dbscan
from api.clustering_algorithms import agglomerative

urlpatterns = [
    path('kmeans', kmeans.get_kmeans),
    path('db_scan', dbscan.get_dbscan),
    path('agglomerative', agglomerative.get_agglomerative)
]