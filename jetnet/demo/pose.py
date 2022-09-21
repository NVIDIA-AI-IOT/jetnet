import os
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates

from .demo import Demo


dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class PoseDemo(Demo):

    def routes(self):
        routes = [
            Route("/keypoints/", self.keypoints), 
            Route("/skeleton/", self.skeleton)
        ]
        return routes + super().routes()

    def index(self, request):
        return templates.TemplateResponse('pose.html.jinja', {'request': request, 'port': self.port})

    def keypoints(self, request):
        return JSONResponse(self.model.get_keypoints())

    def skeleton(self, request):
        return JSONResponse(self.model.get_skeleton())
