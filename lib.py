import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from models import *
from main import model, test_data, writer


def load_images(img_folder):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset = ImageFolder('./data', trans..=...)
    dataset = MNIST(img_folder, download=True, transform=img_transform)

    return dataset


def opts():
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.MSELoss()
    return optimizer, loss_func


def create_grid_from(data, size, n_vis):
    grid = []
    for j in range(n_vis):
        el = data[j].reshape(*size)
        grid.append(el)

    return torchvision.utils.make_grid(grid)


def plot_add(data, name, epoch):
    temp = create_grid_from(data, (1, 28, 28), 8)
    writer.add_image(name, temp, epoch)


def apply_model(dataset):
    new_dataset = []

    for el in dataset:
        new_el = model(el)
        new_dataset.append(new_el)

    return new_dataset


def visualize(mean_test_loss, inp, output, vis_info):
    ep = vis_info[0]
    plot_add(inp, "Input", ep)
    plot_add(output, "Output", ep)
    # tested = apply_model(model, test_dataset)
    # plot_add_8(tested)
    writer.add_scalar("Loss/Test", mean_test_loss, ep)


def test(test_data_iter, loss_func, vis_info):
    total_test_loss = 0
    with torch.no_grad():
        for test_images, _ in test_data_iter:
            _input = test_images
            output = model(_input)

            loss = loss_func(_input, output)
            total_test_loss += loss.item()

        mean_test_loss = total_test_loss / len(test_data)
    print('test complete')
    print('visualisation started')
    visualize(mean_test_loss, _input, output, vis_info)
    print('Loss:', mean_test_loss)
    print('test', vis_info[0], 'visualized')


def train(data, optimizer, loss_func, num_epochs=100):
    for epoch in range(num_epochs):
        data_iter = DataLoader(data, batch_size=128)
        model.improve(data_iter, optimizer, loss_func)
        print('model improved')
        if test_data:
            print('test started')
            test_data_iter = DataLoader(test_data, batch_size=128)
            vis_info = [epoch]
            test(test_data_iter, loss_func, vis_info)