from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('wild', wild_page, name='wild'),
    path('detection', detection, name='detection'),
]