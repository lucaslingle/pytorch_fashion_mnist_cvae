import torch as tc
from torch.nn import functional as F
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np


training_data = tv.datasets.FashionMNIST(
    root='data', train=True, download=True, transform=tv.transforms.ToTensor())

test_data = tv.datasets.FashionMNIST(
    root='data', train=False, download=True, transform=tv.transforms.ToTensor())

# Create data loaders.
batch_size = 64
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


class MLPEncoder(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, num_classes, hidden_dim, z_dim, c_dim):
        super(MLPEncoder, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.flatten = tc.nn.Flatten()
        self.embed = tc.nn.Embedding(num_classes, c_dim)
        self.fc1 = tc.nn.Linear(img_height*img_width*img_channels+c_dim, hidden_dim)
        self.fc2 = tc.nn.Linear(hidden_dim, 2*z_dim)

    def forward(self, x, y):
        flat_x = self.flatten(x)
        class_emb = self.embed(y)
        input_vec = tc.cat([flat_x, class_emb], dim=1)
        a1 = tc.nn.ReLU()(self.fc1(input_vec))
        a2 = self.fc2(a1)
        mu_z, logvar_z = tc.chunk(a2, 2, dim=-1)
        return mu_z, logvar_z


class MLPDecoder(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, discrete_output,
                 num_classes, hidden_dim, z_dim, c_dim):
        super(MLPDecoder, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.discrete_output = discrete_output
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.discrete_output = discrete_output
        self.chan_mult = 1 if discrete_output else 2
        self.embed = tc.nn.Embedding(num_classes, c_dim)
        self.fc1 = tc.nn.Linear(z_dim + c_dim, hidden_dim)
        self.fc2 = tc.nn.Linear(hidden_dim, self.chan_mult * img_height * img_width * img_channels)

    def forward(self, z, y):
        class_emb = self.embed(y)
        input_vec = tc.cat([z, class_emb], dim=1)
        fc1 = self.fc1(input_vec)
        a1 = tc.nn.ReLU()(fc1)
        fc2 = self.fc2(a1)
        x_recon_params = tc.reshape(fc2, (-1, self.img_channels, self.img_height, self.img_width))
        if self.discrete_output:
            return {"x_probs": tc.nn.Sigmoid()(x_recon_params)}
        mu_x, logvar_x = tc.chunk(x_recon_params, 2, dim=1)
        return {
            "mu_x": mu_x,
            "logvar_x": logvar_x
        }


class CVAE(tc.nn.Module):
    def __init__(self, img_height, img_width, img_channels, discrete_output,
                 num_classes, hidden_dim, z_dim, c_dim):
        super(CVAE, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.discrete_output = discrete_output
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.recognition_model = MLPEncoder(img_height, img_width, img_channels,
                                            num_classes, hidden_dim, z_dim, c_dim)
        self.generator = MLPDecoder(img_height, img_width, img_channels, discrete_output,
                                    num_classes, hidden_dim, z_dim, c_dim)

    def forward(self, x, y):
        mu_z, logvar_z = self.recognition_model(x, y) # q(z|x,c)
        z = mu_z + tc.exp(0.5 * logvar_z) * tc.randn_like(mu_z) # z ~ q(z|x,c)
        px_given_zy = self.generator(z, y) # p(x|z,y)
        return px_given_zy, mu_z, logvar_z

    def decode(self, z, y):
        px_given_zy = self.generator(z, y)  # p(x|z,y)
        return px_given_zy

    def sample(self, num_samples):
        z = tc.randn(size=(num_samples, self.z_dim))
        y = tc.randint(low=0, high=10, size=(num_samples,))
        px_given_zy = self.decode(z, y)
        if self.discrete_output:
            return px_given_zy['x_probs'] # dont sample black and white it looks bad
        else:
            return px_given_zy['mu_x'] + tc.exp(0.5 * px_given_zy['logvar_x']) * tc.randn(px_given_zy['mu_x'])

    def loss_fn(self, x, px_given_z, mu_z, logvar_z, discrete_output=True):
        batch_size = len(x)
        kl_div = 0.5 * (logvar_z.exp() + mu_z.pow(2) - 1.0 - logvar_z).sum() / batch_size
        if discrete_output:
            log_px_given_zy = -F.binary_cross_entropy(
                input=px_given_z['x_probs'], target=x, reduction='sum') / batch_size
            elbo = log_px_given_zy - kl_div
            return -elbo
        else:
            log_px_given_zy_unred = -tc.log(2 * np.pi) - 0.5 * px_given_z['logvar_x'] + \
                          -0.5 * tc.square((px_given_z['mu_x'] - x) / tc.exp(0.5 * px_given_z['logvar_x']))
            log_px_given_zy = log_px_given_zy_unred.sum() / batch_size
            elbo = log_px_given_zy - kl_div
            return -elbo

device = "cuda" if tc.cuda.is_available() else "cpu"
model = CVAE(img_height=28, img_width=28, img_channels=1, discrete_output=True,
             num_classes=10, hidden_dim=512, z_dim=100, c_dim=10).to(device)
print(model)

optimizer = tc.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    num_training_examples = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        x_recon, mu_z, logvar_z = model(X, y)
        loss = loss_fn(X, x_recon, mu_z, logvar_z)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            current_idx = batch_size * (batch-1) + len(X)
            print(f"loss: {loss.item():>7f}  [{current_idx:>5d}/{num_training_examples:>5d}]")


def test(dataloader, model, loss_fn):
    num_test_examples = len(dataloader.dataset)
    model.eval()
    test_loss = 0.0
    with tc.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            x_recon, mu_z, logvar_z = model(X, y)
            test_loss += len(X) * loss_fn(X, x_recon, mu_z, logvar_z).item()
    test_loss /= num_test_examples
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, model.loss_fn, optimizer)
    test(test_dataloader, model, model.loss_fn)

print("Done!")

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    input_example = X[0]
    input_label = y[0]

    x_recon, _, _ = model.forward(
        tc.unsqueeze(input_example, dim=0),
        tc.unsqueeze(input_label, dim=0))

    output_example = x_recon["x_probs"].detach()[0]

    img = np.transpose(np.concatenate([input_example, output_example], axis=-1), axes=[1,2,0])
    img_3channel = np.concatenate([img for _ in range(0,3)], axis=-1)
    plt.imshow(img_3channel)
    plt.show()
    break


output_examples = model.sample(num_samples=8).detach()
output_examples = np.transpose(output_examples, axes=[0, 2, 3, 1])
img = np.concatenate([output_examples[i] for i in range(0,8)], axis=1)
img_3channel = np.concatenate([img for _ in range(0, 3)], axis=-1)
plt.imshow(img_3channel)
plt.show()




