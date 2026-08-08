import torch
import _pickle as cPickle
import numpy as np

def output_label(yy):

    data = list(yy)
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)), data] = 1
    return yy1

with open("Copy_RML2016.10a_dict.pkl", 'rb') as f:
  Xd = cPickle.load(f, encoding="latin1")
# Preprocessing the data
# Separate the SNR and Modulation into snrs and mods lists from the key of Xd
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []

for mod in mods:
    # mod is the label. mod = modulation scheme
    for snr in snrs:
       if snr>=-10 and snr<=10:
         X.append(Xd[(mod, snr)])
#         #snr = signal to noise ratio
         for i in range(Xd[(mod, snr)].shape[0]):
           lbl.append((mod, snr))

X = np.vstack(X)
X = (X - np.average(X))/np.std(X)
X = np.expand_dims(X, axis=1)

# Partition the data into training and test sets
# Taking 75% of the samples for the train set & 25% for the test set
np.random.seed(2016)

n_examples = X.shape[0]
n_train = int(n_examples * 0.75)

train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples))-set(train_idx))

X_train = X[train_idx]
X_test =  X[test_idx]

Y_train = output_label(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = np.argmax(Y_train, axis=1)

Y_test = output_label(map(lambda x: mods.index(lbl[x][0]), test_idx))
Y_test = np.argmax(Y_test, axis=1)

X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)

Y_train_tensor = torch.Tensor(Y_train)
Y_test_tensor = torch.Tensor(Y_test)

torch.save(X_train_tensor,"X_train.pt")
torch.save(X_test_tensor,"X_test.pt")

torch.save(Y_train_tensor,"Y_train.pt")
torch.save(Y_test_tensor,"Y_test.pt")

in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
print(classes)
print(snrs)

for snr in snrs:
    if snr>=-10 and snr<=10:
      # extract classes @ SNR
      #changed map to list as part of upgrade from python2
      test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
      test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
      test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

      X_test_tensor_i = torch.Tensor(test_X_i)
      Y_test_tensor_i = torch.Tensor(test_Y_i)

      torch.save(X_test_tensor_i,"X_test_%s.pt" % snr)
      torch.save(Y_test_tensor_i,"Y_test_%s.pt" % snr)
