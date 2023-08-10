import os
from os import path
import importlib.metadata
from typing import BinaryIO, Union

import numpy as np
import ffmpeg
from fastapi import FastAPI, File, UploadFile, Query, applications, Depends, Body
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from whisper import tokenizer

import requests

ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")
if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import transcribe, language_detection
else:
    from .openai_whisper.core import transcribe, language_detection

SAMPLE_RATE=16000
LANGUAGE_CODES=sorted(list(tokenizer.LANGUAGES.keys()))

projectMetadata = importlib.metadata.metadata('whisper-asr-webservice')
app = FastAPI(
    title=projectMetadata['Name'].title().replace('-', ' '),
    description=projectMetadata['Summary'],
    version=projectMetadata['Version'],
    contact={
        "url": projectMetadata['Home-page']
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        "url": projectMetadata['License']
    }
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")
    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )
    applications.get_swagger_ui_html = swagger_monkey_patch

@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/asr", tags=["Endpoints"])
def asr(
    payload: dict = Body(...),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps: bool = Query(
        default=False,
        description="World level timestamps",
        include_in_schema=(True if ASR_ENGINE == "faster_whisper" else False)
    )
):
    audio_url = payload["inputs"]["audio"][0]
    language = payload["inputs"]["language"][0]
    
    audio_data = get_audio_from_url(audio_url)
    result = transcribe(load_audio(audio_data, encode), task, language, None, word_timestamps, output)
    
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            'Asr-Engine': ASR_ENGINE,
            'Content-Disposition': f'attachment; filename="result.{output}"'
        })

def get_audio_from_url(url: str) -> bytes:
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def load_audio(data: bytes, encode=True, sr: int = SAMPLE_RATE) -> np.ndarray:
    if encode:
        try:
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=data)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = data

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
