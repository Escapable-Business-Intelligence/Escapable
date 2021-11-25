from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<int:cafe_index>/', views.detail, name='detail'),
    path('recommend/', views.recommend, name='recommend'),
    path('selectThema', views.selectThema, name='selectThema'),
    path('test/', views.test, name='test'),
]