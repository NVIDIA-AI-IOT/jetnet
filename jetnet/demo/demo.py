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
from jetnet.utils import import_object
from jetnet.image import Image

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from jetnet.utils import import_object
from jetnet.classification import ClassificationModel
from jetnet.detection import DetectionModel
from jetnet.text_detection import TextDetectionModel
from jetnet.pose import PoseModel
from pydantic import BaseModel, PrivateAttr
from typing import Any

dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class Demo(BaseModel):

    model: Any
    host: str = "0.0.0.0"
    port: int = 8000
    camera_device: int = 0
    exclude_image: bool = False

    _running = PrivateAttr(default=False)
    _thread = PrivateAttr(default=None)
    _camera = PrivateAttr()
    _image_shape = PrivateAttr()
    _sio = PrivateAttr()
    _sio_app = PrivateAttr()
    _app = PrivateAttr()

    def run(self):
        self.model = self.model.build()

        self._camera = cv2.VideoCapture(self.camera_device)
        re, image = self._camera.read()
        if re:
            self._image_shape = {"width": int(image.shape[1]), "height": int(image.shape[0])}
        else:
            self._camera.release()
            raise RuntimeError("Could not read an image from the camera.")

        self._sio = socketio.AsyncServer(async_mode="asgi")
        self._sio_app = socketio.ASGIApp(self._sio)
        self._app = Starlette(
            debug=True, 
            routes=self.routes(), 
            on_startup=[self._start_inference],
            on_shutdown=[self._stop_inference]
        )
        uvicorn.run(self._app, host=self.host, port=self.port)


    def get_image_shape(self, request):
        return JSONResponse(self._image_shape)

    def routes(self):
        routes = [
            Route('/', self.index),
            Mount("/ws", self._sio_app),
            Route("/image_shape", self.get_image_shape)
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
        camera = self._camera
        re, image = camera.read()
        
        while self._running and re:
            image_jpeg = bytes(
                cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            output = self.model(image)
            if not self.exclude_image:
                loop.run_until_complete(self._sio.emit("image", image_jpeg))
            loop.run_until_complete(self._sio.emit("output", output.json()))

            re, image = camera.read()

    def index(self, request):
        raise NotImplementedError


def register_args(parser):
    parser.add_argument('model', type=str)
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--camera_device', type=int, default=0)
    parser.add_argument('--exclude_image', action="store_true")


def run_args(args):
    from .classification import ClassificationDemo
    from .detection import DetectionDemo
    from .text_detection import TextDetectionDemo
    from .pose import PoseDemo

    model = import_object(args.model)
    
    if issubclass(model.__class__, ClassificationModel):
        demo = ClassificationDemo(model=model, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model.__class__, DetectionModel):
        demo = DetectionDemo(model=model, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model.__class__, TextDetectionModel):
        demo = TextDetectionDemo(model=model, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)
    if issubclass(model.__class__, PoseModel):
        demo = PoseDemo(model=model, host=args.host, port=args.port, camera_device=args.camera_device, exclude_image=args.exclude_image)

    demo.run()


def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)

if __name__ == "__main__":
    main()