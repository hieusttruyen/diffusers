import time
import os, fnmatch
import torch
from diffusers import (
    DiffusionPipeline,
    pipelines as diffusers_pipelines,
)

HOME = os.path.expanduser("~")
# MODELS_DIR = os.path.join(HOME, ".cache", "huggingface\hub")
MODELS_DIR = "./models/"

_pipelines = {}
_availableCommunityPipelines = None


def listAvailablePipelines():
    return (
        list(
            filter(
                lambda key: key.endswith("Pipeline"),
                list(diffusers_pipelines.__dict__.keys()),
            )
        )
        + availableCommunityPipelines()
    )


def availableCommunityPipelines():
    global _availableCommunityPipelines
    if not _availableCommunityPipelines:
        _availableCommunityPipelines = list(
            map(
                lambda s: s[0:-3],
                fnmatch.filter(os.listdir("diffusers/examples/community"), "*.py"),
            )
        )

    return _availableCommunityPipelines


def clearPipelines():
    """
    Clears the pipeline cache.  Important to call this when changing the
    loaded model, as pipelines include references to the model and would
    therefore prevent memory being reclaimed after unloading the previous
    model.
    """
    global _pipelines
    _pipelines = {}


def getPipelineForModel(pipeline_name: str, model, model_id):
    """
    Inits a new pipeline, re-using components from a previously loaded
    model.  The pipeline is cached and future calls with the same
    arguments will return the previously initted instance.  Be sure
    to call `clearPipelines()` if loading a new model, to allow the
    previous model to be garbage collected.
    """
    pipeline = _pipelines.get(pipeline_name)
    if pipeline:
        print('HEHE')
        return pipeline

    start = time.time()


    # if not os.path.isdir(model_dir):
    #     model_dir = None

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        # revision=revision,
        # torch_dtype=torch_dtype,
        torch_dtype=torch.float16,
        # custom_pipeline="./diffusers/examples/community/" + pipeline_name + ".py",
        # local_files_only=True,
        **model.components,
    )

    if pipeline:
        print(pipeline)
        _pipelines.update({pipeline_name: pipeline})
        diff = round((time.time() - start) * 1000)
        print(f"Initialized {pipeline_name} for {model_id} in {diff}ms")
        return pipeline
