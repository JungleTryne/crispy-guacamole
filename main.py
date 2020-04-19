import os
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from lib import load_images, ConvAutoEncoder, opts, train

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

writer = None
test_data = None
model = None


def main():
    global writer
    global test_data
    global model
    data_folder = './data'
    dataset = load_images(data_folder)
    print('data loaded')

    model = ConvAutoEncoder()

    train_data, test_data = train_test_split(dataset, test_size=0.33, random_state=1)

    print('data split')

    optimizations = opts()
    writer = SummaryWriter()
    print('training started')
    train(train_data, *optimizations, num_epochs=5)


if __name__ == '__main__':
    main()
