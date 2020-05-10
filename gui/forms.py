from django import forms

METHOD_CHOICES = [
    ('and', 'AND (Default)'),
    ('or', 'OR'),
    ('not', 'NOT')
]

DISPLAY_CHOICES = [
    ('original', 'Original'),
    ('clean', 'Clean')
]

CORRECTION_CHOICES = [
    ('soundex', 'Soundex'),
    ('levenshtein', 'Levenshtein'),
]


class SearchForm(forms.Form):
    query = forms.CharField(label='query', max_length=100)
    method = forms.CharField(
        label='method', widget=forms.RadioSelect(choices=METHOD_CHOICES))
    correction = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=CORRECTION_CHOICES)
    display = forms.CharField(
        label='display', widget=forms.RadioSelect(choices=DISPLAY_CHOICES))
    threshold = forms.IntegerField(label='threshold')
