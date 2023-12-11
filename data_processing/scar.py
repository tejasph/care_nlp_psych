# from torchtext.data.datasets_utils import _wrap_split_argument
# from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.vocab import build_vocab_from_iterator
import io
import os.path
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchdata.datapipes as dp
import torchtext.transforms as T
import torch
import random


class SCAR:
    DATASET_NAME = "SCAR"
    NUM_CLASSES = 1 #?

    def __init__(self, batch_size, data_root, target, eval_only, undersample=False, device=torch.device('cuda:0'),
                 min_freq=10):
        self.batch_size = batch_size
        self.device = device

        if eval_only:
            self.data_dir = os.path.join(data_root, target)

            # Load previously compiled vocab path
            self.vocab = torch.load(os.path.join(self.data_dir, 'scar_neural_vocab.pth'))

            self.path_test = os.path.join(self.data_dir, 'test.tsv')

            # A dict of pathpaths for the different data partitions
            self.path_dict = {'train': None,
                           'dev': None,
                           'test': self.path_test}

            # Count number of lines in all of the data splits
            with open(self.path_test) as f:
                self.n_test = len(f.readlines())
            f.close()

            # Create a dict with the lengths
            self.n_lines = {'train': 0,
                            'dev': 0,
                            'test': self.n_test}

            # Make iters of the input test text paths
            self.test_iter = self.create_iter(split='test')
        else:

            if undersample:
                self.data_dir = os.path.join(data_root, target + "_undersampled")
            else:
                self.data_dir = os.path.join(data_root, target)

            self.path_train = os.path.join(self.data_dir, 'train.tsv')
            self.path_dev = os.path.join(self.data_dir, 'dev.tsv')
            self.path_test = os.path.join(self.data_dir, 'test.tsv')
            self.path_dict = {'train': self.path_train,
                           'dev': self.path_dev,
                           'test': self.path_test}

            # Count number of lines in all of the data splits
            with open(self.path_train) as f:
                self.n_train = len(f.readlines())
            f.close()
            with open(self.path_dev) as f:
                self.n_dev = len(f.readlines())
            f.close()
            with open(self.path_test) as f:
                self.n_test = len(f.readlines())
            f.close()

            # Create a dict with the lengths
            self.n_lines = {'train': self.n_train,
                            'dev': self.n_dev,
                            'test': self.n_test}

            ## if undersample:
            ##    self.n_lines['train'] = 1815
            ##    raise NotImplementedError

            # Make iters of the input text files
            # self.train_iter, self.dev_iter, self.test_iter = self.create_iter(split=('train', 'dev', 'test'))
            self.train_dp = self.read_text('train')
            self.dev_dp = self.read_text('dev')
            self.test_dp = self.read_text('test')

            # Build vocab using training split (to avoid data leakage)
            self.vocab = build_vocab_from_iterator(
                self.getTokens(self.train_dp),
                min_freq=10,
                specials=['<unk>', '<BOS>', '<EOS>', '<PAD>'],
                special_first=True
            )

            # If token not recognized as part of the vocab, then set as unknown
            self.vocab.set_default_index(self.vocab['<unk>'])

            # Transforms strings by tokenizing, and then converting to indices (based on self.vocab)
            self.train_dp = self.train_dp.map(self.applyTransform)
            self.dev_dp = self.dev_dp.map(self.applyTransform)
            self.test_dp = self.test_dp.map(self.applyTransform)

            # Bucket batch data to be of similar sizes (for efficiency purposes, and to minimize padding
            self.train_dp = self.train_dp.bucketbatch(
                batch_size=32, batch_num=100, bucket_num=1,
                use_in_batch_shuffle=False, sort_key=self.sortBucket
            )
            self.dev_dp = self.dev_dp.bucketbatch(
                batch_size=32, batch_num=100, bucket_num=1,
                use_in_batch_shuffle=False, sort_key=self.sortBucket
            )
            self.test_dp = self.test_dp.bucketbatch(
                batch_size=32, batch_num=100, bucket_num=1,
                use_in_batch_shuffle=False, sort_key=self.sortBucket
            )

            ## Separate labels from targets
            self.train_dp = self.train_dp.map(self.separateSourceTarget)
            self.dev_dp = self.dev_dp.map(self.separateSourceTarget)
            self.test_dp = self.test_dp.map(self.separateSourceTarget)

            # Add padding based on each batch size
            self.train_dp = self.train_dp.map(self.applyPadding)
            self.dev_dp = self.dev_dp.map(self.applyPadding)
            self.test_dp = self.test_dp.map(self.applyPadding)
            # Convert training set to iter of tokens
            # self.train_token_iter = self.get_scar_tokens(self.create_iter(split='train'))

            # Build vocabulary from tokens
            # self.vocab = self.build_scar_vocab(min_freq)

    def get_vocab_size(self):
        return len(self.vocab)

    # def build_scar_vocab(self, min_freq=10):
    #     vocab = build_vocab_from_iterator(self.train_token_iter, min_freq=min_freq,
    #                                       specials=('<BOS>', '<EOS>', '<PAD>'))
    #
    #     # Add unknown token at default index position
    #     unknown_token = '<unk>'
    #     vocab.insert_token(unknown_token, 0)
    #     vocab.set_default_index(vocab[unknown_token])
    #
    #     # Export so can be used in evaluation-only mode
    #     torch.save(vocab, os.path.join(self.data_dir, 'scar_neural_vocab.pth'))
    #
    #     return vocab

    # @staticmethod
    # def tokenizer(t):
    #     our_tokenizer = get_tokenizer('basic_english')
    #     return our_tokenizer(t)

    def getTokens(self, data_iter):
        """
        Function to yield tokens from an iterator.
        """

        tokenizer = get_tokenizer('basic_english')

        for _, text in data_iter:
            yield tokenizer(text)

    def getTransform(self, vocab):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """
        text_tranform = T.Sequential(
            ## converts the sentences to indices based on given vocabulary
            T.VocabTransform(vocab=vocab),
            ## Add <bos> at beginning of each sentence. 1 because the index for <bos> in vocabulary is
            # 1 as seen in previous section
            T.AddToken(1, begin=True),
            ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
            # 2 as seen in previous section
            T.AddToken(2, begin=False)
        )
        return text_tranform

    def applyTransform(self, data_iter):
        """
        Apply transforms to each example
        """
        tokenizer = get_tokenizer('basic_english')

        return self.label_transform(data_iter[0]), self.getTransform(self.vocab)(tokenizer(data_iter[1]))

    def sortBucket(self,bucket):
        """
        Function to sort a given bucket. Here, we want to sort based on the length of
        each doc.
        """
        return sorted(bucket, key=lambda x: len(x[1]))

    def separateSourceTarget(self,label_text_pairs):
        """
        input of form: `[(y_1,X_1), (y_2,X_2), (y_3,X_3), (y_4,X_4)]`
        output of form: `((y_1,y_2,y_3,y_4), (X_1,X_2,X_3,X_4))`
        """
        labels, text = zip(*label_text_pairs)
        return labels, text

    def applyPadding(self, data_iter):
        """
        Convert sequences to tensors and apply padding
        """
        return (torch.tensor(data_iter[0]), T.ToTensor(3)(list(data_iter[1])))

    ## `T.ToTensor(0)` returns a transform that converts the sequence to `torch.tensor` and also applies
    # padding. Here, `0` is passed to the constructor to specify the index of the `<pad>` token in the
    # vocabulary.

    # @staticmethod
    # def target_parse(target_text):
    #     """
    #     DELETE THIS, USE TARGET TRANSFORMER BELOW INSTEAD
    #
    #     Compability with Hedwig requires that a binary target be two digits, to support multi-class.
    #     For now, keep it like this but convert it to single digit (0 or 1) at this point
    #
    #     :param target_text:
    #     :return:
    #     """
    #
    #     if len(target_text) == 2:
    #         print("Target is two digits, converting)")
    #         if target_text == "10":
    #             return 0
    #         elif target_text == "01":
    #             return 1
    #     elif len(target_text) == 1:
    #         print("Target is already only one digit!")
    #         return int(target_text)

    # @_wrap_split_argument(('train', 'dev', 'test'))
    # def create_iter(root, split):
    #     def generate_scar_data(key, files):
    #         f_name = files[key]
    #         f = io.open(f_name, "r")
    #         for line in f:
    #             values = line.split("\t")
    #             assert len(values) == 2, \
    #                 'Error: excepted SCAR datafile to be tsv format, but splitting by tab did not yield 2 parts'
    #             label = values[0]  # root.target_parse(values[0])
    #             text = values[1]
    #             yield label, text
    #
    #     iterator = generate_scar_data(split, root.path_dict)
    #     return _RawTextIterableDataset(root.DATASET_NAME, root.n_lines[split], iterator)

    def read_text(self, split):

        split_path = os.path.join(self.data_dir, split + '.tsv')
        print(split_path)
        # Creates iterable of filenames
        data_pipe = dp.iter.IterableWrapper([split_path])

        # Iterable passed to a file opener
        data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')

        # Parses file, returns iterable of tupeles representing each row of tsv file
        data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)

        return data_pipe

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.label_transform(_label))
            processed_text = torch.tensor(self.text_transform(_text), device=self.device)
            text_list.append(processed_text)
        return pad_sequence(text_list, padding_value=3.0), torch.tensor(label_list, device=self.device)

    def batch_sampler(self, split):
        split_list = list(self.create_iter(split=split))
        indices = [(i, len(self.tokenizer(s[1]))) for i, s in enumerate(split_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), self.batch_size):
            yield pooled_indices[i:i + self.batch_size]

    def get_bucket_dataloader(self, split):
        split_list = list(self.create_iter(split=split))
        return DataLoader(split_list, batch_sampler=self.batch_sampler(split=split),
                          collate_fn=self.collate_batch)

    # def train_dataloader(self):
    #     return self.get_bucket_dataloader('train')

    # def dev_dataloader(self):
    #     return self.get_bucket_dataloader('dev')
    #
    # def test_dataloader(self):
    #     return self.get_bucket_dataloader('test')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dp)

    def dev_dataloader(self):
        return DataLoader(dataset=self.dev_dp)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dp)



    @classmethod
    def get_scar_tokens(cls, train_iter):
        """
        :param train_iter: SCAR training data iterator returning label and text for each example
        :return: Generator with each text line tokenized
        """
        for label, text in train_iter:
            yield cls.tokenizer(text)

    @staticmethod
    def label_transform(label):
        """
        Transforms labels, which may be in the hedwig format of 10 = 0, 01 = 1, to a binary float

        Will need to add more support to make multi-label, if we wanna go there.

        :param label: the string representation of the label, maybe '0'/'1' or '10'/'01'
        :return: float representation, 0 or 1. If need multi-label, will need to change to return a list etc.
        """

        # print(f'Here is a label: {label} with type {type(label)}')

        if len(label) == 1:
            return float(label)
        elif len(label) == 2 and label in ['10', '01']:
            if label == '10':
                return 0.0
            elif label == '01':
                return 1.0
        else:
            raise ValueError("Invalid target provided, current supports '0'/'1' or '10'/'01'")

    def text_transform(self, text):
        """
        Text transformer. Currently we do the hedwig preprocessing before any of this which is:
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)

        So then, all we need to do is convert a patient's text to tokens
        :param text: text of a patient
        :return: text after transformation to vocab index position
        """
        return [self.vocab['<BOS>']] + [self.vocab[token] for token in self.tokenizer(text)] + [self.vocab['<EOS>']]

    def get_class_balance(self):
        targets_equal_one = 0
        targets_total = 0

        for targets, _ in self.train_dataloader():
            targets = targets.view(-1, 1).float()
            targets_total += len(targets.cpu().detach().numpy())
            targets_equal_one += targets.cpu().detach().numpy().sum()

        return targets_equal_one / targets_total