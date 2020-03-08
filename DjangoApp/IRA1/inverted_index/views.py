from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views import View
from django.middleware.csrf import get_token
from django.views import View
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
import os
from .helpers import PostingList, build_index, get_boolean_query, get_phrasal_query, get_proximity_query
from .models import InvertedIndexModel
FILE_PATH = os.path.dirname(__file__) + '../../data/' + 'Trump Speechs/speech_'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class Test(View):
    def get(self, request):
        return JsonResponse({'status':True, 'message':'Server is up'}, status=200)


class DocumentRetreival(View):
    def get(self, request, doc_id):
        content = 'ERROR Reading Data'
        try:
            BASE_URL = os.path.join(BASE_DIR, 'IRA1\static')
            with open(f'{BASE_URL}/speech_{doc_id}.txt', 'r') as file:
                content = file.read()
            return HttpResponse(content,content_type='text/plain', status=200)
        except BaseException as e:
            return JsonResponse({'status':False, 'message':str(e), 'result' : ''}, status=400)
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
            ('Proximity Query', 2)
        ]
        return JsonResponse({'status':True, 'message':'Query Engine Status', 'options':query_options}, status=200)

    # @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        try:
            result = []
            print(request.POST)
            if request.POST['query'] == '':
                raise ValueError('Invalid Query')
            if request.POST['query_type'] == 'Boolean Query':
                result = get_boolean_query(request.POST['query'])
            elif request.POST['query_type'] == 'Phrasal Query':
                result = get_phrasal_query(request.POST['query'])
            elif request.POST['query_type'] == 'Proximity Query':           
                result = get_proximity_query(request.POST['query'])
            
            print(type(result))
            print(result)
            if(len(result) == 0):
                raise ValueError('Term not in any Documents.')
            if isinstance(result, set):
                return JsonResponse({'status':True, 'message':'Query Result', 'result':list(result), 'type':'set', 'docs':list(result)}, status=200)
            if isinstance(result, PostingList):
                
                res = result.occurrance
                doc_ids = list(map(lambda pos: pos,result.occurrance.keys()))

                return JsonResponse({'status':True, 'message':'Query Result', 'result':res, 'type':'PostingList', 'docs':doc_ids}, status=200)
            else:
                return JsonResponse({'status':True, 'message':'Something Went Wrong', 'result':list(result), 'type':'Unknown'}, status=200)
            

        except BaseException as e:
            print(e)
            return JsonResponse({'status':False, 'message':str(e), 'result' : ''}, status=400)