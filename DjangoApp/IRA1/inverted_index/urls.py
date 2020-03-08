from django.urls import path, include
from . import views

inverted_index = 'inverted_index'

urlpatterns = [
    path('test/', views.Test.as_view(), name='test'),
    path('indexer/', views.Indexer.as_view(), name='indexer'),
    path('query/', views.QueryEngine.as_view(), name='query_engine'),
    path('docs/<int:doc_id>', views.DocumentRetreival.as_view(), name='document_reteival'),
]