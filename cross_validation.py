import random
def cv(x, y, k=10):
    randomize = random.sample(range(len(x)),int(len(x)))
    x = x.loc[randomize]
    y = y.loc[randomize]
    fold_size = len(x)/k
    storage = []
    for i in range(k):
        a = i*fold_size
        b = (i+1)*fold_size
        training_sample = list(x[a:b].index)
        testing_sample = []
        index = 0
        
        x_train = x.loc[training_sample]
        y_train = y.loc[training_sample]
        
        training_sample.sort()
        for i in range(len(x)):
            while training_sample[index] < i and index < len(training_sample):
                index += 1
                if index >= len(training_sample):
                    break
            else:
                continue
            break
            if training_sample[index] != i:
                testing_sample.append(i)

        
        x_test = x.loc[testing_sample]
        y_test = y.loc[testing_sample]
        
        
        storage.append([training_sample, testing_sample])
    return storage