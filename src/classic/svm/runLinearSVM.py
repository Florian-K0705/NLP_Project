import classic.svm.linearsvm as linearSVM


def runLinearSVM (X_train, X_test, y_train, y_test) :

    #print(X_train, y_train)
    #svm = linearSVM.LinearSVM(0.001, 0.01, 100)
    #svm.fit(X_train, y_train)
    #preds = svm.predict(X_test)
    #print("Vorhersagen:", preds)
    #print("Wahre Labels:", y_test)

    num_classes = 28  # Die ersten 10 Emotionen
    for cls in range(num_classes):
        emotion_name = Emotion(cls).name
        print(f"\n=== Emotion {cls} ({emotion_name}) ===")


        #print("y_Train:", y_train)
        # Erzeuge binäre Labels: 1 = diese Emotion, -1 = andere
        y_train_binary = [1 if cls == label else -1 for label in y_train]
        y_test_binary = [1 if cls == label else -1 for label in y_test]
        #print("y_Train:",  y_train_binary )

        # Initialisiere und trainiere das SVM-Modell
        svm = linearSVM.LinearSVM(0.001, 0.01, 100)
        svm.fit(X_train, y_train_binary)

        # Vorhersage
        preds = svm.predict(X_test)
        #print("Vorhersagen für ", cls, ":\n", preds)

        # Berechne TP, FP, FN
        TP = sum(1 for p, y in zip(preds, y_test_binary) if p == 1 and y == cls+1)
        FP = sum(1 for p, y in zip(preds, y_test_binary) if p == 1 and y != cls+1)
        FN = sum(1 for p, y in zip(preds, y_test_binary) if p != 1 and y == cls+1)

        # Metriken berechnen
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Ausgabe
        print(f"TP={TP}, FP={FP}, FN={FN}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1-Score:  {f1:.2f}")


    #print("Wahre Labels:", y_test)


    















from enum import Enum

class Emotion(Enum):
    ADMIRATION = 0
    AMUSEMENT = 1
    ANGER = 2
    ANNOYANCE = 3
    APPROVAL = 4
    CARING = 5
    CONFUSION = 6
    CURIOSITY = 7
    DESIRE = 8
    DISAPPOINTMENT = 9
    DISAPPROVAL = 10
    DISGUST = 11
    EMBARRASSMENT = 12
    EXCITEMENT = 13
    FEAR = 14
    GRATITUDE = 15
    GRIEF = 16
    JOY = 17
    LOVE = 18
    NERVOUSNESS = 19
    OPTIMISM = 20
    PRIDE = 21
    REALIZATION = 22
    RELIEF = 23
    REMORSE = 24
    SADNESS = 25
    SURPRISE = 26
    NEUTRAL = 27
