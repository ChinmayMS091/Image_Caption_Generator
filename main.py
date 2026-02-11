import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import time
from pickle import dump, load
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_file, to_categorical

# small library for seeing the progress of loops.
from data_utils import (all_img_captions, cleaning_text, create_tokenizer,
                        extract_features, load_doc, max_length as calc_max_length,
                        save_descriptions, load_glove_embeddings)
from model_utils import define_model

parser = argparse.ArgumentParser(description='Image Caption Generator Training')
parser.add_argument('--text-path', type=str, default='Flickr8k_text', help='Path to the Flickr8k text dataset directory')
parser.add_argument('--images-path', type=str, default='Flickr8k_Dataset/Flicker8k_Dataset', help='Path to the Flickr8k image dataset directory')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training (reduced for better gradients)')
parser.add_argument('--learning-rate', type=float, default=0.0001, help='Initial learning rate')
args = parser.parse_args()

# --- NEW: Make paths absolute from script location ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# If the provided paths are not absolute, assume they are relative to the script's directory
if not os.path.isabs(args.text_path):
    args.text_path = os.path.join(script_dir, args.text_path)

if not os.path.isabs(args.images_path):
    args.images_path = os.path.join(script_dir, args.images_path)

#we prepare our text data
filename = os.path.join(args.text_path, "Flickr8k.token.txt")
#loading the file that contains all data
#mapping them into descriptions dictionary img to 5 captions
descriptions = all_img_captions(filename)
print("Length of descriptions =" ,len(descriptions))

#cleaning the descriptions
clean_descriptions = cleaning_text(descriptions)

#building vocabulary 
# vocabulary size is derived from the tokenizer later; no separate vocabulary variable required

#saving each description to file 
save_descriptions(clean_descriptions, "descriptions.txt")

def download_with_retry(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Download attempt {attempt + 1} failed. Retrying in 5 seconds...")
            time.sleep(5)


weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, 'xception_weights.h5')
model = Xception(include_top=False, pooling='avg', weights=weights_path)
 
# 2048 feature vector
print("Extracting features...")
features = extract_features(args.images_path, model)
dump(features, open("features.p","wb"))

#load the data 
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(args.images_path, photo))]
    return photos_present


def load_clean_descriptions(filename, photos): 
    #loading clean_descriptions
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):

        words = line.split()
        if len(words)<1 :
            continue

        image, image_caption = words[0], words[1:]

        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)

    return descriptions


def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))
    #selecting only needed features
    features = {k:all_features[k] for k in photos}
    return features


filename = os.path.join(args.text_path, "Flickr_8k.trainImages.txt")

train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

# --- Validation set (dev) ---
dev_filename = os.path.join(args.text_path, "Flickr_8k.devImages.txt")
val_imgs = load_photos(dev_filename)
val_descriptions = load_clean_descriptions("descriptions.txt", val_imgs)
val_features = load_features(val_imgs)

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

max_length = calc_max_length(train_descriptions)
print(max_length)

# --- Load GloVe embeddings ---
glove_path = os.path.join(script_dir, 'glove.6B.100d.txt')
embedding_dim = 100
if os.path.exists(glove_path):
    print('Loading GloVe embeddings...')
    embedding_matrix = load_glove_embeddings(glove_path, tokenizer, embedding_dim)
else:
    print('GloVe file not found, using random embeddings.')
    embedding_matrix = None

#create input-output sequence pairs from the image description.

#data generator, used by model.fit()
def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]
    
    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    return dataset.batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def validation_data_generator(descriptions, features, tokenizer, max_length):
    """Finite generator for validation data (yields once over the dataset)."""
    def generator():
        for key, description_list in descriptions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            for i in range(len(input_image)):
                yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]

    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset.batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Note: training dataset will be created below as `train_dataset` and used for
# both training and a quick shape check.

# train our model
print('Dataset: ', len(train_imgs))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)
print('Epochs:', args.epochs)
print('Batch Size:', args.batch_size)

def get_steps_per_epoch(train_descriptions):
    total_sequences = 0
    for img_captions in train_descriptions.values():
        for caption in img_captions:
            words = caption.split()
            total_sequences += len(words) - 1
    # Use ceiling division so partial batches count towards a step
    steps = (total_sequences + args.batch_size - 1) // args.batch_size
    return max(1, steps)

# Update training loop
steps = get_steps_per_epoch(train_descriptions)

model = define_model(vocab_size, max_length, embedding_matrix=embedding_matrix, embedding_dim=embedding_dim)

# Use Adam optimizer with controlled learning rate and gradient clipping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# IMPROVED: Lower learning rate + gradient clipping to prevent exploding gradients
optimizer = Adam(learning_rate=args.learning_rate, clipnorm=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print(model.summary())

# making a directory models to save our models
os.makedirs("models", exist_ok=True)

# IMPROVED Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced from 10 - stop earlier if not improving
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # More aggressive reduction (was 0.5)
    patience=3,  # Reduce faster (was 5)
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='models/best_val.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Create training and validation datasets
train_dataset = data_generator(train_descriptions, train_features, tokenizer, max_length)
val_dataset = validation_data_generator(val_descriptions, val_features, tokenizer, max_length)

# Compute validation steps
val_steps = get_steps_per_epoch(val_descriptions)

# Quick sanity check: print one training batch shape
for (a, b) in train_dataset.take(1):
    print('Train batch shapes:', a['input_1'].shape, a['input_2'].shape, b.shape)
    break

model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[reduce_lr, early_stopping, checkpoint],
    verbose=1
)

# Save final model
model.save(f"models/model_final.h5")
print("Training complete. Final model saved.")