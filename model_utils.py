from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Bidirectional, Attention, Concatenate, RepeatVector

# define the captioning model with improvements
def define_model(vocab_size, max_length, embedding_matrix=None, embedding_dim=100):
    # Image feature branch: more layers for better representation
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.3)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = Dropout(0.3)(fe2)
    fe4 = Dense(256, activation='relu')(fe3)
    # Expand image features to sequence length for attention
    fe4_seq = RepeatVector(max_length)(fe4)
    # Project image feature sequence to match the Bidirectional LSTM output dimension (512)
    fe4_seq_proj = Dense(512, activation='relu')(fe4_seq)

    # Sequence/text branch: bidirectional LSTM with higher capacity
    inputs2 = Input(shape=(max_length,), name='input_2')
    if embedding_matrix is not None:
        se1 = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], mask_zero=True, trainable=False)(inputs2)
    else:
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.3)(se1)
    se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)
    se4 = Dropout(0.3)(se3)

    # Attention layer: attends over image features for each word
    attn = Attention()([se4, fe4_seq_proj])
    attn_combined = Concatenate()([se4, attn])
    se5 = LSTM(256)(attn_combined)

    # Merging both branches
    decoder1 = add([fe4, se5])
    decoder2 = Dense(256, activation='relu')(decoder1)
    decoder3 = Dropout(0.3)(decoder2)
    decoder4 = Dense(256, activation='relu')(decoder3)
    outputs = Dense(vocab_size, activation='softmax')(decoder4)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
