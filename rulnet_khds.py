import torch
import torchvision
import torch.nn.functional as F

from sklearn.datasets import fetch_covtype
from khds import parser, Experiment
from khds import UniversalDataset
from khds import Algorithm
from khds import LinearNet
import pandas as pd

class TabularDataset(UniversalDataset):

    '''
    Name                    Abbr # Train # Validation # Test # Num # Cat Task type Batch size
    California Housing      CA      13209       3303        4128 8 0 Regression 256
    Adult                   AD      26048       6513        16281 6 8 Binclass 256
    Helena                  HE      41724       10432       13040 27 0 Multiclass 512
    Jannis                  JA      53588       13398       16747 54 0 Multiclass 512
    Higgs Small             HI      62752       15688       19610 28 0 Binclass 512
    ALOI                    AL      69120       17280       21600 128 0 Multiclass 512
    Epsilon                 EP      320000      80000       100000 2000 0 Binclass 1024
    Year                    YE      370972      92743       51630 90 0 Regression 1024
    Covtype                 CO      371847      92962       116203 54 0 Multiclass 1024
    Yahoo                   YA      473134      71083       165660 699 0 Regression 1024
    Microsoft               MI      723412      235259      241521 136 0 Regression 1024
    '''

    def __init__(self, name, train=True, seed=3463):
        super().__init__()

        if name == 'ca':
            data = fetch_covtype()
            labels = data['target']
            data = pd.DataFrame(data['data'])

        elif name == 'ja':
            data = pd.read_csv('jannis.csv')
            labels = data['class']
            data = data.drop(columns=['class'])

        else:
            raise NotImplementedError



        self.data = torchvision.datasets.MNIST(root=path, train=train, transform=torchvision.transforms.ToTensor())

    def __getitem__(self, index):
        return {'x': self.data.data[index].float() / 255, 'y': self.data.targets[index]}

    def __len__(self):
        return len(self.data)


class MNISTAlgorithm(Algorithm):

    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizers['net'].dense, gamma=0.99)

    def postprocess_epoch(self, sample, aux, results, epoch, train=True):

        x, y = sample['x'], sample['y']

        results['images']['sample'] = x[:16].view(16, 1, 28, 28).data.cpu()

        if train:
            self.scheduler.step()
            results['scalar'][f'lr'] = self.optimizers['net'].dense.param_groups[0]['lr']

        aux = {}
        return aux, results

    def iteration(self, sample, aux, results, train=True):

        x, y = sample['x'], sample['y']

        x = x.view(len(x), -1)
        net = self.networks['net']
        opt = self.optimizers['net']

        y_hat = net(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')

        if train:

            opt.zero_grad()
            loss.backward()
            opt.step()

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((y_hat.argmax(1) == y).float().mean()))
        aux = {}

        return aux, results


def run_mnist(rank, world_size, experiment):

    dataloader = {}
    for k, v in {'train': True, 'test': False}.items():

        dataloader[k] = MNISTDataset(experiment.path_to_data, train=v).dataloader(batch_size=experiment.batch_size,
                                                                       num_workers=experiment.cpu_workers,
                                                                       pin_memory=True)

    # choose your network
    net = LinearNet(784, 256, 10, 4)

    # we recommend using the algorithm argument to determine the type of algorithm to be used
    Alg = globals()[experiment.algorithm]
    alg = Alg(net, dataloader, experiment)

    # simulate input to the network
    x = next(alg.data_generator(train=False))[1]['x']
    x = x.view(len(x), -1)

    experiment.writer_control(enable=not(bool(rank)), networks=alg.get_networks(), inputs={'net': x})

    for results in iter(alg):
        experiment.save_model_results(results, alg,
                                      print_results=True, visualize_results='yes',
                                      store_results='logscale', store_networks='logscale',
                                      visualize_weights=True,
                                                    argv={'images': {'sample': {'dataformats': 'NCHW'}}})


if __name__ == '__main__':

    # here you put all actions which are performed only once before initializing the workers
    # for example, setting running arguments and experiment:

    args = parser.parse_args()

    # we can set here arguments that are considered as constant for this file (mnist_example.py)
    args.project_name = 'mnist'
    args.root_dir = '/home/shared/data/results'
    args.algorithm = 'MNISTAlgorithm'
    args.path_to_data = '/home/elad/projects/mnist'

    experiment = Experiment(args)

    # here we initialize the workers (can be single or multiple workers, depending on the configuration)
    experiment.run(run_mnist)
