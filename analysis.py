import matplotlib.pyplot as plt

def view_learning_curve():
    plt.rc('font', size=14)
    train_acc = [0.9436, 0.9725, 0.9738, 0.974, 0.976, 0.976, 0.9767, 0.9786, 0.9783, 0.9796]
    test_acc = [0.9714, 0.9657, 0.9742, 0.9763, 0.9746, 0.9729, 0.9759, 0.977, 0.9794, 0.9767]
    epochs = range(1,len(train_acc)+1)
    
    plt.plot(epochs, train_acc, 'bs', epochs, test_acc, 'g^')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs of Training')
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.show()

view_learning_curve()
