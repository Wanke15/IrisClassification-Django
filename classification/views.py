from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt,csrf_protect

import urllib
# If you are using Python 3+, import urllib instead of urllib2

import json 

from pyecharts import Bar

from sklearn.externals import joblib
import numpy as np
svc = joblib.load('./classification/static/model/iris_svc.pkl') 

@csrf_exempt
def index(request):
    data = np.zeros((1, 4))
    res_class = ''
    echart = ''
    #result = ''
    class_dict = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}
    if request.method == 'POST':
        data[0][0] = float(request.POST['Col1'])
        data[0][1] = float(request.POST['Col2'])
        data[0][2] = float(request.POST['Col3'])
        data[0][3] = float(request.POST['Col4'])
        result = classify(data)
       
        return render(request, 'index.html', {'res_class': class_dict[result[0]]})
    else:
    	return render(request, 'index.html', {'res_class': ''})


def classify(data):
	       
	
	result = svc.predict(data)    
	
	print(result)
	result_proba = svc.predict_proba(data) 
	bar =Bar("每种分类结果的概率")
	bar.add("分类结果", ["Iris-setosa", "Iris-versicolor", "Iris-virginica"], 
	        [result_proba[0][0], result_proba[0][1], 
	         result_proba[0][2]]
	       )
	bar.show_config()
	bar.render("./classification/templates/echarts.html")
	return result