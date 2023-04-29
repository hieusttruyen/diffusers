from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
from fastapi.responses import FileResponse, StreamingResponse,JSONResponse
from PIL import Image
import functools
from os.path import exists
from render import render_cpu, render_gpu, render,render_x4
import torch
import requests
from typing import Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import traceback
from pydantic import BaseModel
import json
from io import BytesIO

import diffusersapi.app as user_src 
import subprocess
# from sqlalchemy.orm import Session
# import crud, models, schemas
# from database import SessionLocal, engine

# models.Base.metadata.create_all(bind=engine)

# Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


def get_bytes_value(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr.read()


class Item2(BaseModel):
    modelInputs : Dict[str, Any]
    callInputs: Dict[str, Any]

class Item(BaseModel):
    prompt: str
    negative_prompt: str


# lora_path = "D:\AI\web\lora\hipoly3DModelLora_v20"
# model_name = "Hius/DreamFul-V2"
# # ddim = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
# model = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None,  torch_dtype=torch.float16,)
# model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.get("/users/", response_model=list[schemas.User])
# def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
#     users = crud.get_users(db, skip=skip, limit=limit)
#     return users


# @app.get("/renderx4")
# def read_renderx4():
#     render_x4()
#     return "done"



@app.get('/init')
def init():
    user_src.init()
@app.get('/healthcheck')
def healthcheck():
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return JSONResponse({"state": "healthy", "gpu": gpu})

@app.post("/render")
async def inference(Item2: Item2):
    try:
        all_inputs = Item2.dict()
    except:
        all_inputs = Item2.json()
  
    # print(type(all_inputs))
    # return
    call_inputs = all_inputs.get("callInputs", None)
    print(call_inputs)
    # return
    # stream_events = call_inputs and call_inputs.streamEvents != 0
    stream_events=0
    streaming_response = None
    # if stream_events:
    #     streaming_response = await request.respond(content_type="application/x-ndjson")

    try:
        output = await user_src.inference(all_inputs, streaming_response)

        
    except Exception as err:
        output = {
            "$error": {
                "code": "APP_INFERENCE_ERROR",
                "name": type(err).__name__,
                "message": str(err),
                "stack": traceback.format_exc(),
            }
        }

    if stream_events:
        await streaming_response.send(json.dumps(output) + "\n")
    else:
        return JSONResponse(output)

@app.post("/generate_image")
def generate_image(item: Item, request: Request):
   

    image = render(data=item, model=model, lora_path=lora_path)
    if image is not None:
        return {"url_img": str(request.base_url) + "image/test.png"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    
@app.get("/generate_image")
def generate_image( request: Request):
   

    image = render(model=model, lora_path=lora_path)
    if image is not None:
        return {"url_img": str(request.base_url) + "image/test.png"}
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.get("/")
async def root():
    return {"hello": "world"}


@app.get("/load_models")
async def read_load_models():
    repo_id = "Hius/DreamFul-V2"
    pipe = DiffusionPipeline.from_pretrained(repo_id)
    print(pipe)
    return {"hello": "pipe"}


@app.get("/render")
async def read_render(prompt: str):
    model_id = "Hius/DreamFul-V2"
    render_cpu(model_id, prompt, height=768, num_inference_steps=30)

    return FileResponse(f"test.png")


@app.get("/image/{img}")
async def read_image(img: str):
    file_exists = exists(img)
    if file_exists:
        return FileResponse(img)
    else:
        raise HTTPException(status_code=404, detail="Item not found")
