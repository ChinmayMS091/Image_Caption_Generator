import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from PIL import Image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import extract_features

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
ap.add_argument('-m', '--model', default='models/model_final.h5', help="Path to the trained model")
ap.add_argument('-t', '--tokenizer', default='tokenizer.p', help="Path to the tokenizer")
args = ap.parse_args()
img_path = args.image
model_arg = args.model

def word_for_id(integer, tokenizer):
    # Prefer tokenizer.index_word if available (faster lookup)
    if hasattr(tokenizer, 'index_word') and tokenizer.index_word:
        return tokenizer.index_word.get(integer)
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length, beam_width=3):
    """
    Generate caption using beam search (more accurate than greedy).
    beam_width=1 is equivalent to greedy; higher values explore more candidates.
    """
    start_candidates = ['<start>', 'start', 'startseq', '<startseq>', 'sos']
    end_candidates = ['<end>', 'end', 'endseq', '<endseq>', 'eos']

    start_token_idx = None
    end_token_idx = None
    for t in start_candidates:
        if t in tokenizer.word_index:
            start_token_idx = tokenizer.word_index[t]
            break
    for t in end_candidates:
        if t in tokenizer.word_index:
            end_token_idx = tokenizer.word_index[t]
            break

    if start_token_idx is None or end_token_idx is None:
        sample = list(tokenizer.word_index.items())[:30]
        raise KeyError(
            "Tokenizer missing recognized start/end tokens.\n"
            f"Searched for {start_candidates} and {end_candidates}.\n"
            f"Sample tokenizer entries: {sample}\n"
            "Regenerate `tokenizer.p` using the project's preprocessing so it contains a start/end token."
        )

    # Beam search with bounded candidate list
    sequences = [([start_token_idx], 0.0)]  # (token_seq, total_score)

    for step in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            # If sequence ends with end token, mark as complete
            if seq[-1] == end_token_idx:
                all_candidates.append((seq, score))
                continue

            # Pad and predict next token
            padded_seq = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model.predict([photo, padded_seq], verbose=0)[0]

            # Get top beam_width predictions (log probabilities for numerical stability)
            top_indices = np.argsort(preds)[-beam_width:][::-1]  # descending order

            for idx in top_indices:
                prob = float(preds[idx])
                if prob > 0:
                    log_prob = np.log(prob)
                else:
                    log_prob = -1000.0  # very low score for zero probability
                new_seq = seq + [idx]
                new_score = score + log_prob
                all_candidates.append((new_seq, new_score))

        # Keep only top beam_width candidates (sorted by score, descending)
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # If all candidates ended, break early
        if all(seq[-1] == end_token_idx for seq, _ in sequences):
            break

    # Pick the best sequence
    best_seq = sequences[0][0]
    words = [word_for_id(i, tokenizer) for i in best_seq]
    return ' '.join(w for w in words if w is not None)

# --- NEW: Make paths absolute from script location ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = model_arg
tokenizer_path = args.tokenizer

# Make paths absolute when given relative paths
if not os.path.isabs(model_path):
    model_path = os.path.join(script_dir, model_path)

if not os.path.isabs(tokenizer_path):
    tokenizer_path = os.path.join(script_dir, tokenizer_path)

# --- MODIFIED: Point to the correct nested image directory ---
# Assume images are in the 'Flickr8k_Dataset/Flicker8k_Dataset' directory
# relative to the script, which is standard for this project.
if not os.path.isabs(img_path) and not img_path.startswith('Flickr8k_Dataset'):
    img_path = os.path.join(script_dir, 'Flickr8k_Dataset', 'Flicker8k_Dataset', img_path)
elif not os.path.isabs(img_path):
    # If the user provides the full relative path, use it directly
    img_path = os.path.join(script_dir, img_path)

def load_descriptions_from_file(filename):
    descriptions = {}
    if not os.path.exists(filename):
        return descriptions
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' not in line:
                continue
            img, desc = line.split('\t', 1)
            descriptions.setdefault(img, []).append(desc)
    return descriptions

descriptions_file = os.path.join(script_dir, 'descriptions.txt')
file_descriptions = load_descriptions_from_file(descriptions_file)
if file_descriptions:
    try:
        from data_utils import max_length as calc_max_length
        max_length = calc_max_length(file_descriptions)
    except Exception:
        max_length = 35
else:
    max_length = 35

import time
print(f"[test.py] Loading tokenizer from: {tokenizer_path}")
start_t = time.time()
tokenizer = load(open(tokenizer_path, "rb"))
print(f"[test.py] Tokenizer loaded ({time.time()-start_t:.2f}s). Vocabulary size: {len(tokenizer.word_index)}")

print(f"[test.py] Loading model from: {model_path}")
start_t = time.time()
model = load_model(model_path)
print(f"[test.py] Model loaded ({time.time()-start_t:.2f}s).")

# If the model defines a fixed input length for the sequence input, use that
# to avoid shape mismatches (model may expect a different max_length than
# the one inferred from descriptions.txt).
try:
    # model.inputs is typically a list: [image_input, seq_input]
    if hasattr(model, 'inputs') and len(model.inputs) >= 2:
        seq_shape = model.inputs[1].shape
        # seq_shape may be a TensorShape; try to read the second dim
        try:
            seq_len = int(seq_shape.as_list()[1])
        except Exception:
            # fallback for different shape representations
            seq_len = int(seq_shape[1]) if seq_shape[1] is not None else None
        if seq_len is not None:
            max_length = seq_len
except Exception:
    # if anything goes wrong, keep previously computed max_length
    pass

# Prefer using precomputed features if available to avoid loading Xception/weights
features_file = os.path.join(script_dir, 'features.p')
photo = None
if os.path.exists(features_file):
    try:
        print(f"[test.py] Loading precomputed features from {features_file}")
        all_features = load(open(features_file, 'rb'))
        img_name = os.path.basename(img_path)
        if img_name in all_features:
            photo = all_features[img_name]
            print("[test.py] Found precomputed features for image")
        else:
            print("[test.py] No precomputed features for image; will extract via Xception")
    except Exception as e:
        print(f"[test.py] Failed to load features.p: {e}; will extract via Xception")

if photo is None:
    print("[test.py] Creating Xception model for feature extraction (this may be slow)")
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(os.path.dirname(img_path), xception_model, single_image_name=os.path.basename(img_path))

img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)

# clean up output caption: remove any start/end tokens in common forms
remove_tokens = {'<start>', '<end>', 'start', 'end', 'startseq', 'endseq'}
words = [w for w in description.split() if w not in remove_tokens]
cleaned = ' '.join(words).strip()
print("\nGenerated caption:\n", cleaned)

plt.imshow(img)
plt.axis('off')
plt.show()


# python test.py --image Flicker8k_Dataset/1859941832_7faf6e5fa9.jpg
# correct