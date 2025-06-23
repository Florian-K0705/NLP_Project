from enum import Enum

class GoEmotion(Enum):
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

label_mapping = {
    0: 0,    # ADMIRATION
    1: 1,    # AMUSEMENT
    2: 2,    # ANGER
    3: 3,    # ANNOYANCE
    4: 4,    # APPROVAL
    5: 20,   # CARING → OPIMISM
    6: 7,    # CONFUSION → CURIOSITY
    7: 7,    # CURIOSITY
    8: 20,   # DESIRE → OPTIMISM
    9: 9,    # DISAPPOINTMENT
    10: 10,  # DISAPPROVAL
    11: 2,   # DISGUST → ANGER
    12: 9,   # EMBARRASSMENT → DISAPPOINTMENT
    13: 17,  # EXCITEMENT → JOY
    14: 25,  # FEAR → SADNESS
    15: 15,  # GRATITUDE
    16: 25,  # GRIEF → SADNESS
    17: 17,  # JOY
    18: 18,  # LOVE
    19: 25,  # NERVOUSNESS → SADNESS
    20: 20,  # OPTIMISM
    21: 0,   # PRIDE → ADMIRATION
    22: 22,  # REALIZATION
    23: 15,  # RELIEF → GRATITUDE
    24: 25,  # REMORSE → SADNESS
    25: 25,  # SADNESS
    26: 7,   # SURPRISE → CURIOSITY
    27: 27   # NEUTRAL
}
