# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from sanic import Sanic, response
# from sanic.reloader import Reloader
from sanic_ext import Extend
import subprocess
import diffusersapi.app as user_src
import traceback
import os
import json

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("my_app")
server.config.CORS_ORIGINS = os.getenv("CORS_ORIGINS") or "*"
Extend(server)


# Healthchecks verify that the environment is correct on Banana Serverless
@server.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})


# Inference POST handler at '/' is called for every http call from Banana
@server.route("/", methods=["GET"])
async def inference(request):
    try:
        all_inputs = response.json.loads(request.json)
    except:
        all_inputs = request.json

    all_inputs = {
            "modelInputs": {
                "prompt": "a girl",
                "num_inference_steps": 1,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 0,
            },
            "callInputs": {
                "MODEL_ID": "DreamFul-V2",
                "PIPELINE": "StableDiffusionPipeline",
                "SCHEDULER": "LMSDiscreteScheduler",
                "safety_checker": True,
            },
        }
    call_inputs = all_inputs.get("callInputs", None)
    stream_events = call_inputs and call_inputs.get("streamEvents", 0) != 0

    streaming_response = None
    if stream_events:
        streaming_response = await request.respond(content_type="application/x-ndjson")

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
        return response.json(output)


if __name__ == "__main__":
    server.run()
