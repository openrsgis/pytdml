# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang
# Created: 2022-05-04
# Email: sgby@whu.edu.cn
#
# ------------------------------------------------------------------------------
#
# Copyright (c) 2022 OGC Training Data Markup Language for AI Standard Working Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
import argparse
import os
import yaml
from pytdml.io import write_to_json
from pytdml.type import EOTrainingDataset, TrainingDataset


def yaml_to_eo_tdml(yaml_path):
    """
    Transform yaml to tdml
    """
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    eo_training_dataset = EOTrainingDataset.from_dict(yaml_dict)
    return eo_training_dataset


def yaml_to_tdml(yaml_path):
    """
    Transform yaml to tdml
    """
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    eo_training_dataset = TrainingDataset.from_dict(yaml_dict)
    return eo_training_dataset


def traverse_folder(file_path, file_format):
    """
    Traverse target folder
    """
    try:
        file_dir = []
        for root, dirs, files in os.walk(file_path):
            for f in sorted(files):
                if file_format == os.path.splitext(f)[-1]:
                    file_dir.append(os.path.join(root, f))
        return file_dir
    except IOError:
        return IOError("Failed to load dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Encode a training dataset to TrainingDML-AI JSON format based on "
        "YAML configuration file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML configuration file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output TrainingDML-AI JSON file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["EO-TDML", "TDML"],
        help="Specify the output format",
    )

    args = parser.parse_args()
    yaml_path = args.config
    json_path = args.output
    if args.format == "EO-TDML":
        training_datasets = yaml_to_eo_tdml(yaml_path)
    else:
        training_datasets = yaml_to_tdml(yaml_path)
    if training_datasets:
        write_to_json(training_datasets, json_path)


if __name__ == "__main__":
    main()
