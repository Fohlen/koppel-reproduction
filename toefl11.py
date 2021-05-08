# coding=utf-8
# MIT License

# Copyright (c) 2021 Lennard Berger

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import List
import os
import csv
import datasets

_CITATION = """\
@article{blanchard2013toefl11,
  title={TOEFL11: A corpus of non-native English},
  author={Blanchard, Daniel and Tetreault, Joel and Higgins, Derrick and Cahill, Aoife and Chodorow, Martin},
  journal={ETS Research Report Series},
  volume={2013},
  number={2},
  pages={i--15},
  year={2013},
  publisher={Wiley Online Library}
}
"""

_DESCRIPTION = """\
ETS Corpus of Non-Native Written English was developed by Educational Testing Service and is comprised of 12,100 English essays written by speakers of 11 non-English native languages as part of an international test of academic English proficiency, TOEFL (Test of English as a Foreign Language). \
The test includes reading, writing, listening, and speaking sections and is delivered by computer in a secure test center. \
This release contains 1,100 essays for each of the 11 native languages sampled from eight topics with information about the score level (low/medium/high) for each essay.

The corpus was developed with the specific task of native language identification in mind, but is likely to support tasks and studies in the educational domain, including grammatical error detection and correction and automatic essay scoring, in addition to a broad range of research studies in the fields of natural language processing and corpus linguistics. \
For the task of native language identification, the following division is recommended: 82% as training data, 9% as development data and 9% as test data, split according to the file IDs accompanying the data set.
"""

@dataclass
class TOEFL11Config(datasets.BuilderConfig):
    """BuilderConfig for TOEFL11"""
    tokenized: bool = False

class TOEFL11(datasets.GeneratorBasedBuilder):
    """TOEFL11: ETS Corpus of Non-Native Written English. Version 1.0."""

    BUILDER_CONFIG_CLASS = TOEFL11Config
    BUILDER_CONFIGS = [
        TOEFL11Config(
            name="plain text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text"
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        feature_dict = {
            "Filename": datasets.Value("string"),
            "Prompt": datasets.Value("string"),
            "Language": datasets.Value("string"),
            "Text": datasets.Value("string"),
            "Score Level": datasets.Value("string")
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(feature_dict),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://catalog.ldc.upenn.edu/LDC2014T06",
            citation=_CITATION
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"labels": "index-dev.csv"}),
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"labels": "index-training.csv"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"labels": "index-test.csv"})
        ]

    def _generate_examples(self, labels):
        labels_file = os.path.join(self.config.data_dir, labels)
        responses_dir = "responses/tokenized" if self.config.tokenized else "responses/original"
        
        with open(labels_file) as csvfile:
            toefl11reader = csv.DictReader(csvfile, fieldnames=["Filename", "Prompt", "Language", "Score Level"])
            for row in toefl11reader:
                with open(os.path.join(self.config.data_dir, responses_dir, row["Filename"])) as f:
                    text = f.read()

                    yield row["Filename"], {
                        "Filename": row["Filename"],
                        "Prompt": row["Prompt"],
                        "Language": row["Language"],
                        "Text": text,
                        "Score Level": row["Score Level"]
                    }
