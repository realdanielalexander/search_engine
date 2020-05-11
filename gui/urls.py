# from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='query'),
    path('tfidf/', views.toprank, name='tfidf'),
    path('query-ascending/', views.home_ascending, name='query-ascending'),
    path('tfidf-ascending/', views.toprank_ascending, name='tfidf-ascending'),
    path('ajax/more/', views.button_build_index, name='build-index'),
    path('ajax/build-index-tf-idf/',
         views.button_build_index_tf_idf, name='build-index'),
    path('ajax/build-index-language-model/',
         views.button_build_index_language_model, name='build-index'),
]
