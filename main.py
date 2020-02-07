import sys

from framework_NN.Tensor import Tensor
from framework_NN.optimizer import SGD
from framework_NN.loss.CrossEntropyLoss import CrossEntropyLoss
from framework_NN.layers.RNN import LSTMCell
from framework_NN.layers.Embedding import Embedding
import numpy as np


def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        #         output.data *= 25
        #         temp_dist = output.softmax()
        #         temp_dist /= temp_dist.sum()

        #         m = (temp_dist > np.random.rand()).argmax()
        m = output.data.argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s


def train(iterations=400):
    global t, min_loss
    for iter in range(iterations):
        total_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        batches_to_train = len(input_batches)
        #     batches_to_train = 32
        for batch_i in range(batches_to_train):

            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))

            losses = list()
            for bp in range(bptt):
                input = Tensor(input_batches[batch_i][bp], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target = Tensor(target_batches[batch_i][bp], autograd=True)
                batch_loss = criterion.forward(output, target)

                if bp == 0:
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])

            loss = losses[-1]

            loss.backward()
            optim.step()
            total_loss += loss.data / bptt

            epoch_loss = np.exp(total_loss / (batch_i + 1))
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                print()

            log = "\r Iter:" + str(iter)
            log += " - Alpha:" + str(optim.alpha)[0:5]
            log += " - Batch " + str(batch_i + 1) + "/" + str(len(input_batches))
            log += " - Min Loss:" + str(min_loss)[0:5]
            log += " - Loss:" + str(epoch_loss)
            if batch_i == 0:
                log += " - " + generate_sample(n=70, init_char='T').replace("\n", " ")
            if batch_i % 1 == 0:
                sys.stdout.write(log)
        optim.alpha *= 0.99


if __name__ == "__main__":
    np.random.seed(0)

    f = open('shakespear.txt', 'r')
    raw = f.read()
    f.close()

    vocab = list(set(raw))
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    indices = np.array(list(map(lambda x: word2index[x], raw)))

    embed = Embedding(vocab_size=len(vocab), dim=512)
    model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
    model.w_ho.weight.data *= 0

    criterion = CrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

    batch_size = 16
    bptt = 25
    n_batches = int((indices.shape[0] / batch_size))

    trimmed_indices = indices[:n_batches * batch_size]
    batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

    input_batched_indices = batched_indices[0:-1]
    target_batched_indices = batched_indices[1:]
    n_bptt = int(((n_batches - 1) / bptt))
    input_batches = input_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
    target_batches = target_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
    min_loss = 1000
    train()
