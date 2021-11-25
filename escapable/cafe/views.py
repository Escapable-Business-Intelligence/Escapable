from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse

from .models import Cafe
from .models import Thema
from .models import Review

import pandas as pd

def index(request):

    return render(request, 'index.html')

def detail(request, cafe_index):
    return HttpResponse("You're looking at cafe %s." % cafe_index)

def recommend(request):
    cafe_list = Cafe.objects.all()

    context = {
        'cafe_list': cafe_list,
    }

    return render(request, 'recommend.html', context)

@csrf_exempt
def selectThema(request):
    cafeName = json.loads(request.body)
    
    thema_list = Thema.objects.filter(cafe_name=cafeName)
    thema_list = list(thema_list.values())

    return JsonResponse(thema_list, safe=False)


def test(request):
  review_list = Review.objects.all()
  matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])
  review_matrix = matrix.pivot_table('user_rate', index='user_nickname', columns='thema_name', aggfunc='first') # 사용자-테마 행렬

  print(review_matrix)

  return HttpResponse(review_matrix)
