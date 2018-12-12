import matplotlib.pyplot as plt

def view_learning_curve():
    plt.rc('font', size=14)
    train_acc = [0.9542, 0.9824, 0.9851, 0.9863, 0.9871, 0.9882, 0.9883, 0.9897, 0.9897, 0.9904]
    test_acc = [0.9737, 0.9865, 0.9867, 0.987, 0.9883, 0.9887, 0.9884, 0.9844, 0.9883, 0.9891]
    epochs = range(1,len(train_acc)+1)
    
    plt.plot(epochs, train_acc, 'bs', epochs, test_acc, 'g^')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs of Training')
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.show()
