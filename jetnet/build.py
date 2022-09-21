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


from argparse import ArgumentParser
from jetnet.utils import import_object


def register_args(parser):
    parser.add_argument("model_config", type=str)


def run_args(args):
    config = import_object(args.model_config).copy(deep=True)
    config.build()



def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_args()
    run_args(args)


if __name__ == "__main__":
    main()