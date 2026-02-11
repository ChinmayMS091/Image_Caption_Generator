import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import string

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


# Loading a text file into memory
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text


# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption_text = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [caption_text]
        else:
            descriptions[img[:-2]].append(caption_text)
    return descriptions


# Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            desc = img_caption.split()

            cleaned_desc = []
            for word in desc:
                # If word is '<start>' or '<end>', keep it exactly
                if word == '<start>' or word == '<end>':
                    cleaned_desc.append(word)
                else:
                    w = word.lower()
                    w = w.translate(table)
                    # remove hanging 's and a
                    if len(w) > 1 and w.isalpha():
                        cleaned_desc.append(w)
            img_caption = ' '.join(cleaned_desc)
            captions[img][i] = img_caption
    return captions

# All descriptions in one file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(data)


def extract_features(directory, model, single_image_name=None):
    features = {}
    
    if single_image_name:
        image_list = [single_image_name]
    else:
        image_list = tqdm(os.listdir(directory))

    for img_name in image_list:
        try:
            filename = os.path.join(directory, img_name)
            # --- NEW: Check if file is a valid image before processing ---
            if not os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png']:
                continue
            image = load_img(filename, target_size=(299, 299))
            image = img_to_array(image)
            # for images that have 4 channels, we convert them into 3 channels
            if image.shape[2] == 4:
                image = image[..., :3]
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img_name] = feature
        except (IOError, ValueError) as e:
            print(f"ERROR: Could not process image {img_name}. Reason: {e}")
    return features


# converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# creating tokenizer class
def create_tokenizer(descriptions):
    # Flatten descriptions to a list
    desc_list = dict_to_list(descriptions)

    # Ensure each description is wrapped with explicit start/end tokens so
    # the tokenizer will always contain them. This makes the tokenizer
    # robust even if calling code did not add the tokens earlier.
    wrapped = []
    for d in desc_list:
        d = d.strip()
        if not d:
            continue
        if not d.startswith('<start>') and not d.startswith('start') and not d.startswith('startseq'):
            d = '<start> ' + d
        if not d.endswith('<end>') and not d.endswith('end') and not d.endswith('endseq'):
            d = d + ' <end>'
        wrapped.append(d)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(wrapped)
    return tokenizer


# --- GloVe embedding loader ---
def load_glove_embeddings(glove_path, tokenizer, embedding_dim=100):
    """
    Loads GloVe vectors and creates an embedding matrix for the tokenizer vocabulary.
    glove_path: path to glove.6B.100d.txt
    tokenizer: fitted Keras Tokenizer
    embedding_dim: dimension of GloVe vectors (default 100)
    Returns: embedding_matrix (vocab_size x embedding_dim)
    """
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
