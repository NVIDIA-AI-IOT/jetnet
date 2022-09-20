import os
from starlette.templating import Jinja2Templates

from .demo import Demo, DemoConfig


dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class DetectionDemo(Demo):

    def index(self, request):
        return templates.TemplateResponse('detection.html.jinja', {'request': request, 'port': self.port})


class DetectionDemoConfig(DemoConfig):

    def build(self) -> DetectionDemo:
        return DetectionDemo(self)
