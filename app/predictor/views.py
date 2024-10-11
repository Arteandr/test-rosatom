from django.shortcuts import render
from .models import Review
from .ai import clf, reg

def get_all():
    reviews = Review.objects.all().order_by('-id')
    all_reviews = []
    for review in reviews:
        all_reviews.append({'rating': review.rating, 'sentiment': review.sentiment, 'name': review.name, 'text': review.text})
        
    return all_reviews
    

def index(req):
    if req.method == 'POST':
        review_text = req.POST['review']
        Review.objects.create(text=review_text, rating=reg.predict(review_text), sentiment=clf.predict(review_text), name=req.POST['name'])
        return render(req, 'predictor/index.html',{'reviews': get_all() })
        
    return render(req, 'predictor/index.html',{'reviews': get_all() })