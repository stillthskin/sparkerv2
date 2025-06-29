from django.urls import path,include
from . import views

urlpatterns = [
    
    path('', views.login_view, name='login') , 
    path('login', views.login_view, name='login') , 
    path('home/', views.home, name='home') ,
    # path('train_model/', views.train_model, name='train_model'),
]