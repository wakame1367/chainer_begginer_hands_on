from chainer.datasets.fashion_mnist import get_fashion_mnist, \
    get_fashion_mnist_labels

if __name__ == '__main__':
    train, test = get_fashion_mnist(withlabel=True, ndim=1)
