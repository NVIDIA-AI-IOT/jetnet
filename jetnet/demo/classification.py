import os
import argparse
from starlette.templating import Jinja2Templates

from .demo import Demo


dir_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=os.path.join(dir_path, "templates"))


class ClassificationDemo(Demo):

    def index(self, request):
        return templates.TemplateResponse('classification.html.jinja', {'request': request, 'port': self.port})
