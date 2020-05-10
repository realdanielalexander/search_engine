from django import forms

class SearchForm(forms.Form):   
    query = forms.CharField(label='query', max_length=100)
    limit = forms.IntegerField(label='limit')
    lambdaa = forms.FloatField(label='lambdaa')

class SearchForm2(forms.Form):   
    query = forms.CharField(label='query', max_length=100)