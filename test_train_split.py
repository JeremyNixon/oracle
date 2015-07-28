import random
def test_train_split(x, y, test_size=.20):
    training_sample = random.sample(range(len(x)),int(len(x)*(1-test_size))) # Takes the floor
    
    training_sample.sort()
    testing_sample = []
    index = 0
    
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
    
    x_train = x.loc[training_sample]
    y_train = y.loc[training_sample]
    x_test = x.loc[testing_sample]
    y_test = y.loc[testing_sample]
    
    return x_train, y_train, x_test, y_test