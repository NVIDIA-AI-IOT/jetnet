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

from abc import ABC, abstractmethod

from typing import TypeVar, Generic

from pydantic.generics import GenericModel


X = TypeVar("X")
Y = TypeVar("Y")


__all__ = ["Model"]


class Model(GenericModel, Generic[X, Y]):
    
    def init(self):
        pass

    def build(self):
        clone = self.copy(deep=True)
        clone.init()
        return clone

    @abstractmethod
    def __call__(self, x: X) -> Y:
        raise NotImplementedError
