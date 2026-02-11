from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Bidirectional, Attention, Concatenate, RepeatVector, LayerNormalization

# define the captioning model with improvements
def define_model(vocab_size, max_length, embedding_matrix=None, embedding_dim=100):
    """
    Improved image captioning model with:
    1. Consistent embedding dimensions (always 256)
    2. Trainable embeddings for fine-tuning
    3. Reduced dropout (0.2 instead of 0.3)
    4. Layer normalization for stable training
    """
    
    # ===== IMAGE ENCODER =====
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.2)(inputs1)  # Reduced dropout
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = Dropout(0.2)(fe2)
    fe4 = Dense(256, activation='relu')(fe3)
    
    # Expand image features to sequence length for attention
    fe4_seq = RepeatVector(max_length)(fe4)
    fe4_seq_proj = Dense(512, activation='relu')(fe4_seq)

    # ===== TEXT DECODER =====
    inputs2 = Input(shape=(max_length,), name='input_2')
    
    # FIXED: Always use 256 dimensions for consistency
    if embedding_matrix is not None:
        # Use GloVe but make it trainable and project to 256
        actual_dim = embedding_matrix.shape[1]
        se1 = Embedding(vocab_size, actual_dim, weights=[embedding_matrix], 
                       mask_zero=True, trainable=True)(inputs2)  # Changed to trainable
        # Project to consistent 256 dimensions
        if actual_dim != 256:
            se1 = Dense(256)(se1)
    else:
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    
    se2 = Dropout(0.2)(se1)  # Reduced dropout
    se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)
    se3 = LayerNormalization()(se3)  # Added layer normalization
    se4 = Dropout(0.2)(se3)

    # Attention layer: attends over image features for each word
    attn = Attention()([se4, fe4_seq_proj])
    attn_combined = Concatenate()([se4, attn])
    se5 = LSTM(256)(attn_combined)
    se5 = LayerNormalization()(se5)  # Added layer normalization

    # Merging both branches
    decoder1 = add([fe4, se5])
    decoder2 = Dense(256, activation='relu')(decoder1)
    decoder2 = LayerNormalization()(decoder2)  # Added layer normalization
    decoder3 = Dropout(0.2)(decoder2)  # Reduced dropout
    outputs = Dense(vocab_size, activation='softmax')(decoder3)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
