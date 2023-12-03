from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaeModel
import uvicorn
import json
import requests

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

url = 'http://localhost:8000'
headers = {'Content-Type':'application/json'}

class weather(BaseModel):
    gender: int
    age: int
    course: int
    year: int
    cgpa: int
    marital_status: int
    depression: int
    anxiety: int
    panic_attack: int
    treatment: int

@app.get("/")
def home(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@app.post('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request, gender: int = Form(...), age: int = Form(...), course: int = Form(...), year: int = Form(...), cgpa: int = Form(...), marital_status: int = Form(...),depression: int = Form(...),anxiety: int = Form(...),panic_attack: int = Form(...), treatment: int = Form(...)):
    inference_data = [int(gender), int(age), int(course), int(year), int(cgpa), int(marital_status),int(depression),int(anxiety),int(panic_attack),int(treatment),]
    inference_data = json.dumps({"data": [inference_data]})
    r = requests.post(url, data=inference_data, headers=headers)
    result = r.content
    res = str(r.content)
    #print(res[-5])

    if (res[-5] == 1):
        result = "No rain"
    else:
        result = "Rain"
   
    return templates.TemplateResponse("index.html", {"request": request, "result":result})



