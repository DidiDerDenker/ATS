# Imports
import numpy as np
import json
import random
import pickle
import nltk
import tensorflow.compat.v1 as tf


# Global Variables
nltk.download("punkt")
summary_max_len = 30 # TODO: Set maximum summary length dynamically


# Methods
def load_embeddings(file_path):
    vocab2emb = {}

    with open(file_path, encoding="utf8") as f:
        for line in f.readlines():
            row = line.strip().split(" ")
            word = row[0].lower()

            if word not in vocab2emb:
                vocab2emb[word] = np.asarray(row[1:], np.float32)

    return vocab2emb


def setup_vocabulary(vocab2emb, vocab2idx):
    vocab = []
    embeddings = []
    special_tags = ["<UNK>", "<PAD>", "<EOS>"]

    for word in vocab2idx:
        if word in vocab2emb:
            vocab.append(word)
            embeddings.append(vocab2emb[word])

    for special_tag in special_tags:
        vocab.append(special_tag)
        embeddings.append(np.random.rand(len(embeddings[0]), ))

    embeddings = np.asarray(embeddings, np.float32)
    vocab2idx = {word: idx for idx, word in enumerate(vocab)}

    return embeddings, vocab2idx


def vectorize_and_shuffle_data(corpus, vocab2idx):
    vec_texts = []
    vec_summaries = []

    texts = [pair[0] for pair in corpus]
    summaries = [pair[1] for pair in corpus]

    for text, summary in zip(texts, summaries):
        vec_texts.append([vocab2idx.get(word, vocab2idx["<UNK>"]) for word in text])
        vec_summaries.append([vocab2idx.get(word, vocab2idx["<UNK>"]) for word in summary])

    random.seed(101)

    texts_idx = [idx for idx in range(len(vec_texts))]
    random.shuffle(texts_idx)

    vec_texts = [vec_texts[idx] for idx in texts_idx]
    vec_summaries = [vec_summaries[idx] for idx in texts_idx]

    return vec_texts, vec_summaries


def prepare_batches(vec_texts, vec_summaries, embeddings, vocab2idx, output_path):
    X_test = vec_texts[0:10000]
    y_test = vec_summaries[0:10000]

    X_val = vec_texts[10000:20000]
    y_val = vec_summaries[10000:20000]

    X_train = vec_texts[20000:]
    y_train = vec_summaries[20000:]

    train_batches_text, train_batches_summary, \
    train_batches_true_text_len, train_batches_true_summary_len \
        = bucket_and_batch(X_train, y_train, vocab2idx)

    val_batches_text, val_batches_summary, \
    val_batches_true_text_len, val_batches_true_summary_len \
        = bucket_and_batch(X_val, y_val, vocab2idx)

    test_batches_text, test_batches_summary, \
    test_batches_true_text_len, test_batches_true_summary_len \
        = bucket_and_batch(X_test, y_test, vocab2idx)

    d = {}

    d["vocab"] = vocab2idx
    d["embd"] = embeddings.tolist()

    d["train_batches_text"] = train_batches_text
    d["val_batches_text"] = val_batches_text
    d["test_batches_text"] = test_batches_text

    d["train_batches_summary"] = train_batches_summary
    d["val_batches_summary"] = val_batches_summary
    d["test_batches_summary"] = test_batches_summary

    d["train_batches_true_text_len"] = train_batches_true_text_len
    d["val_batches_true_text_len"] = val_batches_true_text_len
    d["test_batches_true_text_len"] = test_batches_true_text_len

    d["train_batches_true_summary_len"] = train_batches_true_summary_len
    d["val_batches_true_summary_len"] = val_batches_true_summary_len
    d["test_batches_true_summary_len"] = test_batches_true_summary_len

    with open(output_path, "w") as f:
        json.dump(d, f)


def bucket_and_batch(texts, summaries, vocab2idx, batch_size=32):
    global summary_max_len

    text_lens = [len(text) for text in texts]
    sortedidx = np.flip(np.argsort(text_lens), axis=0)
    texts = [texts[idx] for idx in sortedidx]
    summaries = [summaries[idx] for idx in sortedidx]

    batches_text = []
    batches_summary = []
    batches_true_text_len = []
    batches_true_summary_len = []

    i = 0

    while i < (len(texts) - batch_size):
        max_len = len(texts[i])

        batch_text = []
        batch_summary = []
        batch_true_text_len = []
        batch_true_summary_len = []

        for j in range(batch_size):
            padded_text = texts[i + j]
            padded_summary = summaries[i + j]

            batch_true_text_len.append(len(texts[i + j]))
            batch_true_summary_len.append(len(summaries[i + j]) + 1)

            while len(padded_text) < max_len:
                padded_text.append(vocab2idx["<PAD>"])

            padded_summary.append(vocab2idx["<EOS>"])

            while len(padded_summary) < summary_max_len + 1:
                padded_summary.append(vocab2idx["<PAD>"])

            batch_text.append(padded_text)
            batch_summary.append(padded_summary)

        batches_text.append(batch_text)
        batches_summary.append(batch_summary)
        batches_true_text_len.append(batch_true_text_len)
        batches_true_summary_len.append(batch_true_summary_len)

        i += batch_size

    return batches_text, batches_summary, batches_true_text_len, batches_true_summary_len


def model_notebook(file_path):
    global summary_max_len

    # Hyperparameters
    hidden_size = 300
    learning_rate = 0.001
    epochs = 5
    local_attention_window_size = 5
    window_len = 2 * local_attention_window_size + 1
    l2 = 1e-6

    # Load Data
    with open(file_path) as f:
        for json_data in f:
            saved_data = json.loads(json_data)

            vocab2idx = saved_data["vocab"]
            embd = saved_data["embd"]

            train_batches_text = saved_data["train_batches_text"]
            val_batches_text = saved_data["val_batches_text"]
            test_batches_text = saved_data["test_batches_text"] # TODO: Use

            train_batches_summary = saved_data["train_batches_summary"]
            val_batches_summary = saved_data["val_batches_summary"]
            test_batches_summary = saved_data["test_batches_summary"] # TODO: Use

            train_batches_true_text_len = saved_data["train_batches_true_text_len"]
            val_batches_true_text_len = saved_data["val_batches_true_text_len"]
            test_batches_true_text_len = saved_data["test_batches_true_text_len"]  # TODO: Use

            train_batches_true_summary_len = saved_data["train_batches_true_summary_len"]
            val_batches_true_summary_len = saved_data["val_batches_true_summary_len"]
            test_batches_true_summary_len = saved_data["test_batches_true_summary_len"]  # TODO: Use

            break

    idx2vocab = {v: k for k, v in vocab2idx.items()}

    # Tensorflow Placeholders
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

    embd_dim = len(embd[0])

    tf_text = tf.placeholder(tf.int32, [None, None])
    tf_embd = tf.placeholder(tf.float32, [len(vocab2idx), embd_dim])
    tf_true_summary_len = tf.placeholder(tf.int32, [None])
    tf_summary = tf.placeholder(tf.int32, [None, None])
    tf_train = tf.placeholder(tf.bool)

    # Embed Vectorized Text
    embd_text = tf.nn.embedding_lookup(tf_embd, tf_text)
    embd_text = dropout(embd_text, rate=0.3, training=tf_train)

    # Forward Encoding in Bi-Directional LSTM-Encoder
    S = tf.shape(embd_text)[1] # text sequence length
    N = tf.shape(embd_text)[0] # batch_size

    i = 0
    hidden = tf.zeros([N, hidden_size], dtype=tf.float32)
    cell = tf.zeros([N, hidden_size], dtype=tf.float32)
    hidden_forward = tf.TensorArray(size=S, dtype=tf.float32)

    embd_text_t = tf.transpose(embd_text, [1, 0, 2]) # shape: [N, S, embd_dim]

    def cond(i, hidden, cell, hidden_forward):
        return i < S

    def body(i, hidden, cell, hidden_forward):
        x = embd_text_t[i]

        hidden, cell = LSTM(x, hidden, cell, embd_dim, hidden_size, scope="forward_encoder")
        hidden_forward = hidden_forward.write(i, hidden)

        return i + 1, hidden, cell, hidden_forward

    _, _, _, hidden_forward = tf.while_loop(cond, body, [i, hidden, cell, hidden_forward])

    # Backward Encoding in Bi-Directional LSTM-Encoder
    i = S - 1
    hidden = tf.zeros([N, hidden_size], dtype=tf.float32)
    cell = tf.zeros([N, hidden_size], dtype=tf.float32)
    hidden_backward = tf.TensorArray(size=S, dtype=tf.float32)

    def cond(i, hidden, cell, hidden_backward):
        return i >= 0

    def body(i, hidden, cell, hidden_backward):
        x = embd_text_t[i]
        hidden, cell = LSTM(x, hidden, cell, embd_dim, hidden_size, scope="backward_encoder")
        hidden_backward = hidden_backward.write(i, hidden)

        return i - 1, hidden, cell, hidden_backward

    _, _, _, hidden_backward = tf.while_loop(cond, body, [i, hidden, cell, hidden_backward])

    # Merge Hidden States of Forward and Backward Encoder
    hidden_forward = hidden_forward.stack()
    hidden_backward = hidden_backward.stack()

    encoder_states = tf.concat([hidden_forward, hidden_backward], axis=-1)
    encoder_states = tf.transpose(encoder_states, [1, 0, 2])

    encoder_states = dropout(encoder_states, rate=0.3, training=tf_train)
    final_encoded_state = dropout(tf.concat([hidden_forward[-1], hidden_backward[-1]], axis=-1), rate=0.3, training=tf_train)

    # LSTM-Decoder with Local Attention
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        SOS = tf.get_variable("sos", shape=[1, embd_dim],
                              dtype=tf.float32,
                              trainable=True,
                              initializer=tf.glorot_uniform_initializer())

        Wc = tf.get_variable("Wc", shape=[4 * hidden_size, embd_dim],
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.glorot_uniform_initializer())

    SOS = tf.tile(SOS, [N, 1]) # shape: [N, embd_dim]
    inp = SOS
    hidden = final_encoded_state
    cell = tf.zeros([N, 2 * hidden_size], dtype=tf.float32)
    decoder_outputs = tf.TensorArray(size=summary_max_len, dtype=tf.float32)
    outputs = tf.TensorArray(size=summary_max_len, dtype=tf.int32)

    attention_scores = align(encoder_states, hidden, hidden_size, N, S, local_attention_window_size, window_len)
    encoder_context_vector = tf.reduce_sum(encoder_states * attention_scores, axis=1)

    for i in range(summary_max_len):
        inp = dropout(inp, rate=0.3, training=tf_train)
        inp = tf.concat([inp, encoder_context_vector], axis=-1)

        hidden, cell = LSTM(inp, hidden, cell, embd_dim + 2 * hidden_size, 2 * hidden_size, scope="decoder")
        hidden = dropout(hidden, rate=0.3, training=tf_train)

        attention_scores = align(encoder_states, hidden, hidden_size, N, S, local_attention_window_size, window_len)
        encoder_context_vector = tf.reduce_sum(encoder_states * attention_scores, axis=1)
        concated = tf.concat([hidden, encoder_context_vector], axis=-1)

        linear_out = tf.nn.tanh(tf.matmul(concated, Wc))
        decoder_output = tf.matmul(linear_out, tf.transpose(tf_embd, [1, 0]))
        decoder_outputs = decoder_outputs.write(i, decoder_output)

        next_word_vec = tf.cast(tf.argmax(decoder_output, 1), tf.int32)
        next_word_vec = tf.reshape(next_word_vec, [N])
        outputs = outputs.write(i, next_word_vec)
        next_word = tf.nn.embedding_lookup(tf_embd, next_word_vec)
        inp = tf.reshape(next_word, [N, embd_dim])

    decoder_outputs = decoder_outputs.stack()
    outputs = outputs.stack()

    decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])
    outputs = tf.transpose(outputs, [1, 0])

    # Cross-Entropy Cost-Function and L2-Regularization
    filtered_trainables = [var for var in tf.trainable_variables() if
                           not ("Bias" in var.name or "bias" in var.name or "noreg" in var.name)]

    regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var
                                    in filtered_trainables])

    with tf.variable_scope("loss"):
        epsilon = tf.constant(1e-9, tf.float32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_summary, logits=decoder_outputs)
        pad_mask = tf.sequence_mask(tf_true_summary_len, maxlen=summary_max_len, dtype=tf.float32)
        masked_cross_entropy = cross_entropy * pad_mask
        cost = tf.reduce_mean(masked_cross_entropy) + l2 * regularization
        cross_entropy = tf.reduce_mean(masked_cross_entropy)

    # Accuracy
    comparison = tf.cast(tf.equal(outputs, tf_summary), tf.float32)
    pad_mask = tf.sequence_mask(tf_true_summary_len, maxlen=summary_max_len, dtype=tf.bool)
    masked_comparison = tf.boolean_mask(comparison, pad_mask)
    accuracy = tf.reduce_mean(masked_comparison)

    # Optimizer
    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(cost, all_vars)
    capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # Training and Validation
    with tf.Session() as sess:
        display_step = 100
        patience = 5

        load = input("\nLoad checkpoint? y/n: ") # TODO: Remove unnecessary print-commands
        print("")

        saver = tf.train.Saver()

        if load.lower() == "y":
            print("Loading pre-trained weights for the model...")

            saver.restore(sess, "models/model.ckpt")
            sess.run(tf.global_variables())
            sess.run(tf.tables_initializer())

            with open("models/summarization.pkl", "rb") as fp:
                train_data = pickle.load(fp)

            covered_epochs = train_data["covered_epochs"]
            best_loss = train_data["best_loss"]
            impatience = 0

        else:
            best_loss = 2 ** 30
            impatience = 0
            covered_epochs = 0

            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(tf.tables_initializer())

        epoch = 0

        while (epoch + covered_epochs) < epochs:
            print("Starting training...")

            batches_indices = [i for i in range(0, len(train_batches_text))]
            random.shuffle(batches_indices)

            total_train_acc = 0
            total_train_loss = 0

            for i in range(0, len(train_batches_text)):
                j = int(batches_indices[i])

                cost, prediction, acc, _ \
                    = sess.run([cross_entropy, outputs, accuracy, train_op], # values depend on the dict-items
                               feed_dict={tf_text: train_batches_text[j],
                                          tf_embd: embd,
                                          tf_summary: train_batches_summary[j],
                                          tf_true_summary_len: train_batches_true_summary_len[j],
                                          tf_train: True})

                total_train_acc += acc
                total_train_loss += cost

                if i % display_step == 0:
                    print("Iter: " + str(i) + ", Cost = {:.3f}".format(cost) + ", Acc = {:.2f}%".format(acc * 100))

                if i % 500 == 0:
                    idx = random.randint(0, len(train_batches_text[j]) - 1)

                    text = " ".join([idx2vocab.get(vec, "<UNK>") for vec in train_batches_text[j][idx]])
                    predicted_summary = [idx2vocab.get(vec, "<UNK>") for vec in prediction[idx]]
                    actual_summary = [idx2vocab.get(vec, "<UNK>") for vec in train_batches_summary[j][idx]]

                    print("\nSample Text\n")
                    print(text)
                    print("\nSample Predicted Summary\n")

                    for word in predicted_summary:
                        if word == "<EOS>":
                            break

                        else:
                            print(word, end=" ")

                    print("\n\nSample Actual Summary\n")

                    for word in actual_summary:
                        if word == "<EOS>":
                            break

                        else:
                            print(word, end=" ")

                    print("\n\n")

            print("Starting validation...")

            total_val_loss = 0
            total_val_acc = 0

            for i in range(0, len(val_batches_text)):
                cost, prediction, acc = sess.run([cross_entropy, outputs, accuracy],
                                                 feed_dict={tf_text: val_batches_text[i],
                                                            tf_embd: embd,
                                                            tf_summary: val_batches_summary[i],
                                                            tf_true_summary_len: val_batches_true_summary_len[i],
                                                            tf_train: False})

                total_val_loss += cost
                total_val_acc += acc

            avg_val_loss = total_val_loss / len(val_batches_text)

            print("\n\nEpoch: {}\n\n".format(epoch + covered_epochs))
            print("Average Training Loss: {:.3f}".format(total_train_loss / len(train_batches_text)))
            print("Average Training Accuracy: {:.2f}".format(100 * total_train_acc / len(train_batches_text)))
            print("Average Validation Loss: {:.3f}".format(avg_val_loss))
            print("Average Validation Accuracy: {:.2f}".format(100 * total_val_acc / len(val_batches_text)))

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_data = {"best_loss": best_loss, "covered_epochs": covered_epochs + epoch + 1}
                impatience = 0

                with open("models/summarization.pkl", "wb") as fp:
                    pickle.dump(save_data, fp)

                saver.save(sess, "models/summarization.ckpt")
                print("\nModel saved\n")

            else:
                impatience += 1

            if impatience > patience:
                break

            epoch += 1


def dropout(x, rate, training):
    return tf.cond(training, lambda: tf.nn.dropout(x, rate=rate), lambda: x)


def LSTM(x, hidden_state, cell, input_dim, hidden_size, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", shape=[4, input_dim, hidden_size],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())

        u = tf.get_variable("u", shape=[4, hidden_size, hidden_size],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())

        b = tf.get_variable("bias", shape=[4, 1, hidden_size],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.zeros_initializer())

    input_gate = tf.nn.sigmoid(tf.matmul(x, w[0]) + tf.matmul(hidden_state, u[0]) + b[0])
    forget_gate = tf.nn.sigmoid(tf.matmul(x, w[1]) + tf.matmul(hidden_state, u[1]) + b[1])
    output_gate = tf.nn.sigmoid(tf.matmul(x, w[2]) + tf.matmul(hidden_state, u[2]) + b[2])

    cell_ = tf.nn.tanh(tf.matmul(x, w[3]) + tf.matmul(hidden_state, u[3]) + b[3])
    cell = forget_gate * cell + input_gate * cell_

    hidden_state = output_gate * tf.tanh(cell)

    return hidden_state, cell


def attention_score(encoder_states, decoder_hidden_state, hidden_size, N, S, scope="attention_score"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Wa = tf.get_variable("Wa", shape=[2 * hidden_size, 2 * hidden_size],
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.glorot_uniform_initializer())

    encoder_states = tf.reshape(encoder_states, [N * S, 2 * hidden_size])

    encoder_states = tf.reshape(tf.matmul(encoder_states, Wa), [N, S, 2 * hidden_size])
    decoder_hidden_state = tf.reshape(decoder_hidden_state, [N, 2 * hidden_size, 1])

    return tf.reshape(tf.matmul(encoder_states, decoder_hidden_state), [N, S])


def align(encoder_states, decoder_hidden_state, hidden_size, N, S, local_attention_window_size, window_len, scope="attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        Wp = tf.get_variable("Wp", shape=[2 * hidden_size, 128],
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.glorot_uniform_initializer())

        Vp = tf.get_variable("Vp", shape=[128, 1],
                             dtype=tf.float32,
                             trainable=True,
                             initializer=tf.glorot_uniform_initializer())

    positions = tf.cast(S - window_len, dtype=tf.float32)
    ps = positions * tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(decoder_hidden_state, Wp)), Vp))
    pt = ps + local_attention_window_size
    pt = tf.reshape(pt, [N])

    i = 0
    gaussian_position_based_scores = tf.TensorArray(size=S, dtype=tf.float32)
    sigma = tf.constant(local_attention_window_size / 2, dtype=tf.float32)

    def cond(i, gaussian_position_based_scores):
        return i < S

    def body(i, gaussian_position_based_scores):
        score = tf.exp(-((tf.square(tf.cast(i, tf.float32) - pt)) / (2 * tf.square(sigma))))
        gaussian_position_based_scores = gaussian_position_based_scores.write(i, score)

        return i + 1, gaussian_position_based_scores

    i, gaussian_position_based_scores = tf.while_loop(cond, body, [i, gaussian_position_based_scores])

    gaussian_position_based_scores = gaussian_position_based_scores.stack()
    gaussian_position_based_scores = tf.transpose(gaussian_position_based_scores, [1, 0])
    gaussian_position_based_scores = tf.reshape(gaussian_position_based_scores, [N, S])

    scores = attention_score(encoder_states, decoder_hidden_state, hidden_size, N, S) * gaussian_position_based_scores
    scores = tf.nn.softmax(scores, axis=-1)

    return tf.reshape(scores, [N, S, 1])
