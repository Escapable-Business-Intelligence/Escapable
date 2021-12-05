from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('thema/<int:thema_index>/', views.thema, name='thema'),
    path('recommend/', views.recommend, name='recommend'),
    path('recommend1/', views.recommend1, name='recommend1'),
    path('recommend2/', views.recommend2, name='recommend2'),
    path('selectThema', views.selectThema, name='selectThema'),
    path('recommendThema1', views.recommendThema1, name='recommendThema1'),
    path('recommendThema2', views.recommendThema2, name='recommendThema2'),
]