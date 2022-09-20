import os
from starlette.templating import Jinja2Templates

from .demo import Demo, DemoConfig


dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class TextDetectionDemo(Demo):

    def index(self, request):
        return templates.TemplateResponse('text_detection.html.jinja', {'request': request, 'port': self.port})


class TextDetectionDemoConfig(DemoConfig):

    def build(self) -> TextDetectionDemo:
        return TextDetectionDemo(self)

