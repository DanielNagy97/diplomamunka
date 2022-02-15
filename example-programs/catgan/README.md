# DCGAN példa program

A `catgan` csomag segítségével betaníthatunk egy GAN hálózatot, amely 64x64-es képek generálására lesz képes.
A csomag tartalmaz egy tanítóhalmaz-beolvasó modult is, amely előkészíti a képeket a tanításhoz.

Példa a csomag használatára:

Az adatbeolvasó és GAN model importálása
```python
from catgan.DatasetLoader import DatasetLoader
from catgan.GanModel import GanModel
```

Paraméterek megadása, adatok beolvasása és GANmodel példányosítása a megfelelő paraméterekkel
```python
epochs = 100
number_of_examples = 16
batch_size = 32
latent_dim = 100
img_height = 64
img_width = 64
data_dir = r'../input/animal-faces/afhq/train'

dataset = DatasetLoader.build_dataset(
    data_dir, (img_height, img_width), batch_size
)
# DatasetLoader.normalize_dataset(dataset)
dataset = dataset.map(lambda x: (x - 127.5) / 127.5)

gan_model = GanModel(latent_dim, batch_size, number_of_examples, 1e-4, 0.5, 0.9)
```

Tanítás
```python
gan_model.train(dataset, epochs)
```
