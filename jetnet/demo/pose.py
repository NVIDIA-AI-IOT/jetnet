# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
