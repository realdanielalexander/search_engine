from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.

def home(request):
    return render(request, 'blog/Index.html')

def query(request):
    return render(request, 'blog/html/Query.html')

def toprank(request):
    return render(request, 'blog/html/TopRank.html')
