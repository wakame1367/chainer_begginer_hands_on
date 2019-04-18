import chainer
from chainer.datasets import split_dataset_random
from chainer.datasets.fashion_mnist import get_fashion_mnist, \
    get_fashion_mnist_labels
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import optimizers, training
from chainer.training import extensions


class MLP2(Chain):

    # Initialization of layers
    def __init__(self):
        super(MLP2, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784,
                               200)  # From 784-dimensional input to hidden unit with 200 nodes
            self.l2 = L.Linear(200,
                               10)  # From hidden unit with 200 nodes to output unit with 10 nodes  (10 classes)

    # Forward computation
    def forward(self, x):
        h1 = F.tanh(self.l1(
            x))  # Forward from x to h1 through activation with tanh function
        y = self.l2(h1)  # Forward from h1to y
        return y


def train_and_validate(
        model, optimizer, train, validation, n_epoch, batchsize, device):
    # 1. deviceがgpuであれば、gpuにモデルのデータを転送する
    if device >= 0:
        model.to_gpu(device)

    # 2. Optimizerを設定する
    optimizer.setup(model)

    # 3. DatasetからIteratorを作成する
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    validation_iter = chainer.iterators.SerialIterator(
        validation, batchsize, repeat=False, shuffle=False)

    # 4. Updater・Trainerを作成する
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='out')

    # 5. Trainerの機能を拡張する
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(validation_iter, model, device=device),
                   name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
        file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    # 6. 訓練を開始する
    trainer.run()


if __name__ == '__main__':
    train, test = get_fashion_mnist(withlabel=True, ndim=1)
    # https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.TupleDataset.html#chainer.datasets.TupleDataset
    print("train's size: {}".format(len(train)))
    print("test's size: {}".format(len(test)))

    labels = get_fashion_mnist_labels()
    print(labels)

    # sample 1
    img, label = test[0]
    print("image's shape: {}".format(img.shape))
    print("label_number: {}".format(label))
    print("label_name: {}".format(labels[label]))

    seed_n = 42
    train, validation = split_dataset_random(train, 50000, seed=seed_n)
    print("train's size: {}".format(len(train)))
    print("validation's size: {}".format(len(validation)))

    device = -1  # specify gpu id. if device == -1, use cpu
    n_epoch = 5  # Only 5 epochs
    batchsize = 256

    model = MLP2()  # MLP2 model
    classifier_model = L.Classifier(model)
    optimizer = optimizers.SGD()

    train_and_validate(
        classifier_model, optimizer, train, validation, n_epoch, batchsize,
        device)
