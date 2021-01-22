# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb
import sys

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, hamming_loss, precision_score, recall_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, category_label_id, sentiment_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.category_label_id = category_label_id
        self.sentiment_label_ids = sentiment_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class JCSCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type, year):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/"+string+"_"+year+"_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_"+year+"_train.tsv")), "train")

    def get_dev_examples(self, data_dir,domain_type, year):
        """See base class."""
        string = domain_type
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_"+year+"_test.tsv")), "test")
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "data/"+"rest_2015_test_notarget.tsv")), "test")

    def get_labels(self, domain_type):
        """See base class."""
        entity = None
        attribute = None
        sentiment = None
        if domain_type == 'restaurant':
            entity = ['FOOD', 'DRINKS', 'SERVICE', 'AMBIENCE', 'LOCATION', 'RESTAURANT']
            attribute = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
        elif domain_type == 'laptop':
            entity = ['LAPTOP','DISPLAY','CPU','MOTHERBOARD','HARD_DISC','MEMORY','BATTERY',
                    'POWER_SUPPLY','KEYBOARD','MOUSE','FANS_COOLING','OPTICAL_DRIVES','PORTS',
                    'GRAPHICS','MULTIMEDIA_DEVICES','HARDWARE','OS','SOFTWARE','WARRANTY','SHIPPING',
                    'SUPPORT','COMPANY']
            attribute = ['GENERAL','PRICE','QUALITY','OPERATION_PERFORMANCE','USABILITY','DESIGN_FEATURES',
            'PORTABILITY','CONNECTIVITY','MISCELLANEOUS']
        sentiment = ['-1', '0', '1', '2']
        label_list = []
        l = []
        for e in entity:
            for att in attribute:
                string = e+'#'+att
                l.append(string)

        label_list.append(l)
        label_list.append(sentiment)
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = line[1:]
            text_b = None
            # if set_type == "train":
            #     text_b = "categories are :"
            #     for index, lab in enumerate(labels):
            #         cate = ' '.join(ele for ele in lab.split('#')[:-1])
            #         if len(labels) == 1:
            #             text_b = "category is : " + cate + ' .'
            #         else:
            #             if index < len(labels)-2:
            #                 text_b = text_b + " " + cate + " ,"
            #             elif index < len(labels)-1:
            #                 text_b = text_b + " " + cate + " and"
            #             else:
            #                 text_b = text_b + " " + cate + " ."
            #     text_b = text_b.replace('_', '')
            # else:
            #     text_b = "which categories are mentioned ? what are their corresponding polarities ?"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels))
        return examples


class JCSC14Processor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/"+string+"_2014_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2014_train.tsv")), "train")

    def get_dev_examples(self, data_dir,domain_type):
        """See base class."""
        string = domain_type
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2014_test.tsv")), "test")

    def get_labels(self, domain_type):
        """See base class."""
        category = ['food', 'service', 'price', 'ambience', 'anecdotes miscellaneous']
        sentiment = ['-1', '0', '1', '2']
        label_list = []

        label_list.append(category)
        label_list.append(sentiment)
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = line[1:]
            text_b = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels))
        return examples


class ESCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/rest_2015_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        entity = ['FOOD', 'DRINKS', 'SERVICE', 'AMBIENCE', 'LOCATION', 'RESTAURANT']

        return entity

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = [ele.split('#')[0] for ele in line[1:]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


class ASCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/rest_2015_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        attribute = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']

        return attribute

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = [ele.split('#')[1] for ele in line[1:]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


class SSCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/rest_2015_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/rest_2015_test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        sentiment = ['-1', '0', '1']

        return sentiment

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = [ele.split('#')[-1] for ele in line[1:]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


class EASCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")), "train")

    def get_dev_examples(self, data_dir,domain_type):
        """See base class."""
        string = domain_type
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_test.tsv")), "test")

    def get_labels(self, domain_type):
        """See base class."""
        entity = None
        attribute = None
        sentiment = None
        if domain_type == 'restaurant':
            entity = ['FOOD', 'DRINKS', 'SERVICE', 'AMBIENCE', 'LOCATION', 'RESTAURANT']
            attribute = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
        elif domain_type == 'laptop':
            entity = ['LAPTOP','DISPLAY','CPU','MOTHERBOARD','HARD_DISC','MEMORY','BATTERY',
                    'POWER_SUPPLY','KEYBOARD','MOUSE','FANS_COOLING','OPTICAL_DRIVES','PORTS',
                    'GRAPHICS','MULTIMEDIA_DEVICES','HARDWARE','OS','SOFTWARE','WARRANTY','SHIPPING',
                    'SUPPORT','COMPANY']
            attribute = ['GENERAL','PRICE','QUALITY','OPERATION_PERFORMANCE','USABILITY','DESIGN_FEATURES',
            'PORTABILITY','CONNECTIVITY','MISCELLANEOUS']
        label_list = []
        for e in entity:
            for att in attribute:
                    string = e+'#'+att
                    label_list.append(string)

        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            labels = [ele.split('#')[0]+'#'+ele.split('#')[1] for ele in line[1:]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


class ESSCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")), "train")

    def get_dev_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_test.tsv")), "test")

    def get_labels(self, domain_type):
        """See base class."""
        if domain_type == 'restaurant':
            entity = ['FOOD', 'DRINKS', 'SERVICE', 'AMBIENCE', 'LOCATION', 'RESTAURANT']
        else:
            entity = ['LAPTOP','DISPLAY','CPU','MOTHERBOARD','HARD_DISC','MEMORY','BATTERY',
                    'POWER_SUPPLY','KEYBOARD','MOUSE','FANS_COOLING','OPTICAL_DRIVES','PORTS',
                    'GRAPHICS','MULTIMEDIA_DEVICES','HARDWARE','OS','SOFTWARE','WARRANTY','SHIPPING',
                    'SUPPORT','COMPANY']
        sentiment = ['-1', '0', '1', '2']
        label_list = []
        label_list.append(entity)
        label_list.append(sentiment)

        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            entity_dict = {}
            labels = [ele.split('#')[0]+'#'+ele.split('#')[2] for ele in line[1:]]
            flag = True
            for ele in labels:
                if ele.split('#')[0] not in entity_dict:
                    entity_dict[ele.split('#')[0]] = ele.split('#')[1]
                else:
                    if entity_dict[ele.split('#')[0]] != ele.split('#')[1]:
                        flag = False
            if flag:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


class ASSCProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_train.tsv")), "train")

    def get_dev_examples(self, data_dir, domain_type):
        """See base class."""
        string = domain_type
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "data/"+string+"_2016_test.tsv")), "test")

    def get_labels(self, domain_type):
        """See base class."""
        attribute = None
        if domain_type == 'restaurant':
            attribute = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE_OPTIONS', 'MISCELLANEOUS']
        else:
            attribute = ['GENERAL','PRICE','QUALITY','OPERATION_PERFORMANCE','USABILITY','DESIGN_FEATURES',
            'PORTABILITY','CONNECTIVITY','MISCELLANEOUS']
        sentiment = ['-1', '0', '1', '2']
        label_list = []
        label_list.append(attribute)
        label_list.append(sentiment)

        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            attribute_dict = {}
            labels = [ele.split('#')[1]+'#'+ele.split('#')[2] for ele in line[1:]]
            flag = True
            for ele in labels:
                if ele.split('#')[0] not in attribute_dict:
                    attribute_dict[ele.split('#')[0]] = ele.split('#')[1]
                else:
                    if attribute_dict[ele.split('#')[0]] != ele.split('#')[1]:
                        flag = False
            if flag:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, task_name):
    """Loads a data file into a list of `InputBatch`s."""

    label_map_category = {label : i for i, label in enumerate(label_list[0])}
    label_map_sentiment = {label : i for i, label in enumerate(label_list[1])}

    features = []
    senti_stat = np.zeros((3, 3), dtype=np.int32)
    overall_map = np.zeros((len(label_list[1])*len(label_list[0]), len(label_list[1])*len(label_list[0])), dtype=np.int32)
    senti_map = np.zeros((len(label_list[1]), len(label_list[0]), len(label_list[0])), dtype=np.int32)
    senti_cate_map = np.zeros((len(label_list[1]), len(label_list[0]), len(label_list[0])), dtype=np.int32)
    category_map = np.zeros((len(label_list[0]), len(label_list[0])), dtype=np.int32)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        category_label_id = [0] * len(label_list[0])
        sentiment_label_id = [3] * len(label_list[0])
        label_map = {0 : [1, 0, 0], 1 : [0, 1, 0], 2 : [0, 0, 1], 3 : [0, 0, 0]}

        # sentiment_label_id = [3] * len(label_list[0])
        # label_map = {0 : [1, 0, 0, 0], 1 : [0, 1, 0, 0], 2 : [0, 0, 1, 0], 3 : [0, 0, 0, 1]}

        category_relation_list = []
        senti_cate_list = []
        if output_mode == "classification":
            for ele in example.label:
                t = ele.split('#')
                senti = t[-1]
                type_list = ['jcsc14', 'essc', 'assc']
                if task_name in type_list :
                    cat = t[0]
                else:
                    cat = t[0]+'#'+t[1]
                category_label_id[label_map_category[cat]] = 1
                category_relation_list.append(label_map_category[cat])
                senti_cate_list.append([label_map_category[cat], label_map_sentiment[senti]])
                sentiment_label_id[label_map_category[cat]] = label_map_sentiment[senti]
                sentiment_label_ids = [label_map[sentiment_label_id[i]] for i in range(len(sentiment_label_id))]

            for c_i in range(len(category_relation_list)):
                category_map[category_relation_list[c_i]][category_relation_list[c_i]] += 1
            for c_i in range(len(category_relation_list)):
                for c_j in range(c_i+1, len(category_relation_list)):
                    category_map[category_relation_list[c_i]][category_relation_list[c_j]] += 1
                    category_map[category_relation_list[c_j]][category_relation_list[c_i]] += 1

            for c_i in range(len(senti_cate_list)):
                senti_cate_map[senti_cate_list[c_i][1]][senti_cate_list[c_i][0]][senti_cate_list[c_i][0]] += 1
                senti_map[senti_cate_list[c_i][1]][senti_cate_list[c_i][0]][senti_cate_list[c_i][0]] += 1
                overall_index = senti_cate_list[c_i][1]*len(label_list[0])+senti_cate_list[c_i][0]
                overall_map[overall_index][overall_index] += 1
            for c_i in range(len(senti_cate_list)):
                for c_j in range(c_i+1, len(senti_cate_list)):
                    #i->j
                    senti_cate_map[senti_cate_list[c_j][1]][senti_cate_list[c_i][0]][senti_cate_list[c_j][0]] += 1
                    #j->i
                    senti_cate_map[senti_cate_list[c_i][1]][senti_cate_list[c_j][0]][senti_cate_list[c_i][0]] += 1

                    if senti_cate_list[c_i][1] == senti_cate_list[c_j][1]:
                        #i->j
                        senti_map[senti_cate_list[c_i][1]][senti_cate_list[c_i][0]][senti_cate_list[c_j][0]] += 1
                        #j->i
                        senti_map[senti_cate_list[c_i][1]][senti_cate_list[c_j][0]][senti_cate_list[c_i][0]] += 1

                    overall_x = senti_cate_list[c_i][1]*len(label_list[0])+senti_cate_list[c_i][0]
                    overall_y = senti_cate_list[c_j][1]*len(label_list[0])+senti_cate_list[c_j][0]
                    overall_map[overall_x][overall_y] += 1
                    overall_map[overall_y][overall_x] += 1
                    senti_stat[senti_cate_list[c_i][1]][senti_cate_list[c_j][1]] += 1
                    senti_stat[senti_cate_list[c_j][1]][senti_cate_list[c_i][1]] += 1

        else:
            raise KeyError(output_mode)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s\n%s)" % (example.label, category_label_id, sentiment_label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              category_label_id=category_label_id,
                              sentiment_label_ids=sentiment_label_ids))
    # sns.heatmap(overall_map, annot=False, fmt="d")
    # plt.show()
    # sns.heatmap(senti_stat, annot=True, fmt="d")
    # plt.show()
    # pdb.set_trace()
    diag_category_map = np.diag(category_map)
    for i in range(len(senti_cate_map)):
        np.fill_diagonal(senti_cate_map[i], diag_category_map)
    # pdb.set_trace()
    return [category_map, senti_cate_map, senti_map], features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    hamming = hamming_loss(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "micro-f1": f1,
        "hamming_loss":hamming,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

processors = {
    "jcsc": JCSCProcessor,
    "esc": ESCProcessor,
    "asc": ASCProcessor,
    "ssc": SSCProcessor,
    "easc": EASCProcessor,
    "essc": ESSCProcessor,
    "assc": ASSCProcessor,
    "jcsc14": JCSC14Processor,
}

output_modes = {
    "jcsc": "classification",
    "esc": "classification",
    "asc": "classification",
    "ssc": "classification",
    "easc": "classification",
    "essc": "classification",
    "assc": "classification",
    "jcsc14": "classification",
}

def fix_14_label(pred, gold):
    fix_dict = {'category': 6, 'sentiment': 3}
    for ins_index, one_instance in enumerate(pred):
        for index, label in enumerate(one_instance):
            if label == 1:
                ls = index % 3
                lc = (index // fix_dict['sentiment']) % fix_dict['category']
                val_indexs = []
                for j in range(fix_dict['sentiment']):
                    ind = 3 * lc + j
                    if gold[ins_index][ind] == 1:
                        val_indexs.append(ind)
                if len(val_indexs) > 0:
                    for ele in val_indexs:
                        one_instance[ele] = 1
                    if index not in val_indexs:
                        one_instance[index] = 0
    return pred, gold

def fix_label(domain, fix_type, pred, gold):
    def get_val(l, num_list):
        return l[0] + num_list[0]*l[1] + num_list[1]*l[2]
    num_list = None
    if domain == 'restaurant':
        fix_dict = {'entity': 6, 'attribute': 5, 'sentiment': 3}
        num_list = [3, 15]
    else:
        fix_dict = {'entity': 22, 'attribute': 9, 'sentiment': 3}
        num_list = [3, 27]
    for ins_index, one_instance in enumerate(pred):
        for index, label in enumerate(one_instance):
            if label == 1:
                ls = index % fix_dict['sentiment']
                la = (index // fix_dict['sentiment']) % fix_dict['attribute']
                le = (index // num_list[1]) % fix_dict['entity']
                val_indexs = []
                for j in range(fix_dict[fix_type]):
                    if fix_type == 'entity':
                        ind = get_val([ls, la, j], num_list)
                    elif fix_type == 'attribute':
                        ind = get_val([ls, j, le], num_list)
                    elif fix_type == 'sentiment':
                        ind = get_val([j, la, le], num_list)
                    if gold[ins_index][ind] == 1:
                        val_indexs.append(ind)
                if len(val_indexs) > 0:
                    for ele in val_indexs:
                        one_instance[ele] = 1
                    if index not in val_indexs:
                        one_instance[index] = 0
    return pred, gold