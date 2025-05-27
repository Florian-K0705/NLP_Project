import numpy as np
import classic.mlp.helpfunctions as help
import classic.mlp.mlp as m

def runMPL(X_train, X_test, y_train, y_test):

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(np.unique(y_train))

    mlp = m.MLP(input_dim, hidden_dim, output_dim, lr=0.08)
    mlp.train(X_train, y_train, epochs=20)

    # Test
    y_pred = mlp.predict(X_test)
    test_acc = help.accuracy(y_test, y_pred)
    print("Test Accuracy:", test_acc)