from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views import View
from django.middleware.csrf import get_token
from django.views import View
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from .helpers import build_index, get_boolean_query
from .models import InvertedIndexModel

class Test(View):
    def get(self, request):
        return JsonResponse({'status':True, 'message':'Server is up'}, status=200)

  
class Indexer(View):
    def get(self, request):

        return JsonResponse({'status':True, 'message':'Indexer Status'}, status=200)

    # @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        status = build_index()
        return JsonResponse({'status':status, 'message':'Starting Indexing'}, status=200)

class QueryEngine(View):
    def get(self, request):
        query_options = [
            ('Boolean Query', 0),
            ('Phrasal Query', 1),
            ('Positional Query', 2)
        ]
        return JsonResponse({'status':True, 'message':'Query Engine Status', 'options':query_options}, status=200)

    # @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        try:
            if request.post['query_type'] == 'Boolean Query':
                result = get_boolean_query(request.post['query'])
            elif request.post['query_type'] == 'Phrasal Query':
                result = 0
            elif request.post['query_type'] == 'Positional Query':           
                result = 0 
            return JsonResponse({'status':True, 'message':'Query Result', 'result':result}, status=200)

        except BaseException as e:
            return JsonResponse({'status':False, 'message':e, 'result' : ''}, status=401)