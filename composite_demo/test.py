import time
import requests

# 定义装饰器
def time_calc(func):
    print("============%s" % str(func))
    def wrapper(*args, **kargs):        
        start_time = time.time()        
        f = func(*args,**kargs)        
        exec_time = time.time() - start_time        
        return f    
    return wrapper   
    
# 使用装饰器
@time_calc    
def add(a, b):
    return a + b
    
@time_calc
def sub(a, b):    
    return a - b


resp = requests.get(f"https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key=4c2c0cde277e830eee150de7d9aa64eb")
resp.raise_for_status()
resp = resp.json()
ret = {k: v for k, v in resp.items()}
print(str(ret))