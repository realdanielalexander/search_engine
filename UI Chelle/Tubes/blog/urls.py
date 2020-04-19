from django.urls import path
# from fontawesome.fields import IconField
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('query/', views.query, name="query"),
    path('toprank/', views.toprank, name="toprank"),
]
