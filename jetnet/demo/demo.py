from argparse import ArgumentParser

import os
import asyncio
import cv2
import uvicorn
import threading
import PIL.Image
import json
import socketio
from typing import Union

from pydantic import BaseModel, PrivateAttr
from jetnet.config import Config
from jetnet.utils import import_object
from jetnet.image import Image

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from jetnet.utils import import_object
from jetnet.classification import ClassificationModelConfig
from jetnet.detection import DetectionModelConfig
from jetnet.text_detection import TextDetectionModelConfig
from jetnet.pose import PoseModelConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class DemoConfig(Config):

    model_config: Config
    host: str = "0.0.0.0"
    port: int = 8000
    camera_device: int = 0
    exclude_image: bool = False


class Demo(object):

    def __init__(self, cfg: DemoConfig):
        self._running = False
        self._thread = None
        self._model = cfg.model_config.build()
        self._cfg = cfg

    def run(self):
        self._sio = socketio.AsyncServer(async_mode="asgi")
        self._sio_app = socketio.ASGIApp(self._sio)
        self._app = Starlette(
            debug=True, 
            routes=self.routes(), 
            on_startup=[self._start_inference],
            on_shutdown=[self._stop_inference]
        )
        uvicorn.run(self._app, host=self.host, port=self.port)

    @property
    def host(self) -> str:
        return self._cfg.host

    @property
    def port(self) -> int:
        return self._cfg.port

    @property
    def camera_device(self) -> int:
        return self._cfg.camera_device

    @property
    def exclude_image(self):
        return self._cfg.exclude_image

    def routes(self):
        routes = [
            Route('/', self.index),
            Mount("/ws", self._sio_app),
        ]
        return routes

    def _start_inference(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_inference)
        self._thread.start()

    def _stop_inference(self):
        self._running = False
        self._thread.join()
        self._thread = None
        
    def _run_inference(self):

        loop = asyncio.new_event_loop()
        camera = cv2.VideoCapture(self.camera_device)
        re, image = camera.read()

        while self._running and re:
            image_jpeg = bytes(
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image(PIL.Image.fromarray(image))
            output = self.model(image)
            if not self.exclude_image:
                loop.run_until_complete(self._sio.emit("image", image_jpeg))
            loop.run_until_complete(self._sio.emit("output", output.json()))

            re, image = camera.read()

    def index(self, request):
        raise NotImplementedError

    @property
    def model(self):
        return self._model

def register_args(parser):
    parser.add_argument('model_config', type=str)
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--camera_device', type=int, default=0)
    parser.add_argument('--exclude_image', action="store_true")


def run_args(args):
    from .classification import ClassificationDemoConfig
    from .detection import DetectionDemoConfig
    from .text_detection import TextDetectionDemoConfig
    from .pose import PoseDemoConfig

    model_config = import_object(args.model_config)
    
    if issubclass(model_config.__class__, ClassificationModelConfig):
        demo_config = ClassificationDemoConfig(model_config=model_config, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model_config.__class__, DetectionModelConfig):
        demo_config = DetectionDemoConfig(model_config=model_config, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model_config.__class__, TextDetectionModelConfig):
        demo_config = TextDetectionDemoConfig(model_config=model_config, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model_config.__class__, PoseModelConfig):
        demo_config = PoseDemoConfig(model_config=model_config, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)

    demo = demo_config.build()

    demo.run()


def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)

if __name__ == "__main__":
    main()