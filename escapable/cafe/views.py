from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse

from .models import Cafe
from .models import Thema

def index(request):

    return render(request, 'index.html')

def detail(request, cafe_index):
    return HttpResponse("You're looking at cafe %s." % cafe_index)

def recommend(request):

    return render(request, 'recommend.html')

def recommend1(request):
    cafe_list = Cafe.objects.all()

    context = {
        'cafe_list': cafe_list,
    }

    return render(request, 'recommend1.html', context)

def recommend2(request):
    cafe_list = Cafe.objects.all()

    context = {
        'cafe_list': cafe_list,
    }

    return render(request, 'recommend2.html', context)

@csrf_exempt
def selectThema(request):
    cafeName = json.loads(request.body)
    
    thema_list = Thema.objects.filter(cafe_name=cafeName)
    thema_list = list(thema_list.values())

    return JsonResponse(thema_list, safe=False)

@csrf_exempt
def recommendThema1(request):
    data_list = json.loads(request.body)

    print(data_list)

    #머신러닝 처리 -> 추천 테마 데이터 받아오기

    context = {
        'result': data_list,
    }

    return JsonResponse(context, safe=False)

@csrf_exempt
def recommendThema2(request):
    data_list = json.loads(request.body)

    #머신러닝 처리 -> 추천 테마 데이터 받아오기

    context = {
        'result': data_list,
    }

    return JsonResponse(context, safe=False)
