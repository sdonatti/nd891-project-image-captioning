import json
import os

import nltk
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm.auto import tqdm

from vocabulary import Vocabulary


def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file=os.path.join('.', 'vocab.pkl'),
               start_word='<start>',
               end_word='<end>',
               unk_word='<unk>',
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc=os.path.join('/', 'opt')):
    """
    Returns the data loader.

    Args:
      transform: Image transform.
      mode: One of 'train', 'val', or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary.
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    assert mode in ['train', 'val', 'test'], "Mode must be one of 'train' or 'test'."

    if not vocab_from_file:
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # based on mode (train, val, test), obtain img_folder and annotations_file
    if mode == 'train':

        if vocab_from_file:
            assert os.path.exists(vocab_file), "Change vocab_from_file to False to create vocab_file."

        img_folder = os.path.join(cocoapi_loc, 'cocoapi', 'images', 'train2014')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi', 'annotations', 'captions_train2014.json')

    else:
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."

        if mode == 'val':
            img_folder = os.path.join(cocoapi_loc, 'cocoapi', 'images', 'val2014')
            annotations_file = os.path.join(cocoapi_loc, 'cocoapi', 'annotations', 'captions_val2014.json')

        else:
            img_folder = os.path.join(cocoapi_loc, 'cocoapi', 'images', 'test2014')
            annotations_file = os.path.join(cocoapi_loc, 'cocoapi', 'annotations', 'image_info_test2014.json')

    # create a COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        # randomly sample a caption length, and sample indices with that length
        indices = dataset.get_train_indices()
        # create and assign a batch sampler to retrieve a batch with the sampled indices
        initial_sampler = SubsetRandomSampler(indices=indices)
        # create a data loader for COCO dataset
        batch_sampler = BatchSampler(sampler=initial_sampler, batch_size=dataset.batch_size, drop_last=False)
        data_loader = DataLoader(dataset=dataset, num_workers=num_workers, batch_sampler=batch_sampler)

    else:
        data_loader = DataLoader(dataset=dataset, batch_size=dataset.batch_size, shuffle=True, num_workers=num_workers)

    return data_loader


class CoCoDataset(Dataset):

    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word,
                 end_word, unk_word, annotations_file, vocab_from_file, img_folder):

        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        if self.mode == 'train' or self.mode == 'val':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())

            if self.mode == 'train':
                print('Obtaining caption lengths...')
                all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower())
                              for index in tqdm(np.arange(len(self.ids)))]
                self.caption_lengths = [len(token) for token in all_tokens]

        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):

        # obtain image and caption if in training mode
        if self.mode == 'train' or self.mode == 'val':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # convert caption to tensor of word ids
            if self.mode == 'train':

                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = [self.vocab(self.vocab.start_word)]
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(self.vocab(self.vocab.end_word))
                caption = torch.Tensor(caption).long()

            # get all captions for the image id
            else:
                ann_ids = self.coco.getAnnIds(img_id)
                ann_list = self.coco.loadAnns(ann_ids)
                caption = [ann['caption'] for ann in ann_list]

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # convert image to tensor and pre-process using transform
            pil_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(pil_image)
            image = self.transform(pil_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):

        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))

        return indices

    def clean_sentence(self, output):
        """
        Generates a human readable sentence from a list of word indices that represent an image caption

        Parameters
        ----------
        output : list
            The word indices that represent an image caption

        Returns
        -------
        str:
            The human readable sentence generated from the word indices
        """
        # get the words from the indices and fix end of sentence
        words = [self.vocab.idx2word[idx] for idx in output]
        words = words[1:-1] if words[-2] == '.' else words[1:-1] + ['.']

        # humanize the sentence
        sentence = words[0]
        if len(words) > 1:
            punctuation = ['.', ',', ':', ';']
            for idx in torch.arange(len(words[1:])):
                sentence += words[int(idx+1)] if words[int(idx+1)] in punctuation else ' {}'.format(words[int(idx+1)])

        # return the sentence
        return sentence.capitalize()

    def __len__(self):

        if self.mode == 'train' or self.mode == 'val':
            return len(self.ids)

        else:
            return len(self.paths)
