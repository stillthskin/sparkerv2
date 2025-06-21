from django.shortcuts import render
from django.http import JsonResponse,request
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
# from .NModel import AI_Model
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.core.cache import cache
from django.contrib.auth.models import User


# Create your views here.

# @csrf_exempt
# def train_model():
#     model=AI_Model(request)
    
#     if request.method == 'POST':
#         res=model.train_model()
#         return JsonResponse({'status': 'success', 'message': res})
#     return JsonResponse({'status': 'error', 'message': 'Invalid request'})



def login_view(request):
    if request.method == 'POST':
        # Get the form data
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # If authentication is successful, log the user in
            login(request, user)
            return redirect('home')  # Redirect to home or any success page
        else:
            # If authentication fails, show an error message
            messages.error(request, "Invalid username or password")

    return render(request, 'login.html')

@login_required(login_url='/login/')  # Redirects to this URL if not logged in
def home(request):
    assets = cache.get('current_assets', [])
    
    # Optionally add cache status flag
    cache_status = {
        'is_fallback': len(assets) > 0 and assets[0]['symbol'] == 'ETH' and assets[0]['free'] == 200.0,
        'timestamp': cache.get('current_assets_timestamp')
    }

    open_positions = cache.get('open_positions', {})
    
    # Convert to list of dictionaries for easier templating
    positions_list = [
        {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'quantity': position['quantity']
        } for symbol, position in open_positions.items()
    ]

    contec = {
        'positions_list': positions_list,
        'assets': assets,
        'cache_status': cache_status
    }
    
    return render(request, 'index2.html', context=contec)


