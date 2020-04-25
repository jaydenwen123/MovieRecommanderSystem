'''
Created on 2019年2月20日

@author: wenxiaofei-PC
'''
from django.http import request,HttpResponse

def test(request):
    return HttpResponse("hello world django!!")