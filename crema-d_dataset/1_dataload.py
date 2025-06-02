import os
import pandas as pd

audio_directory = "cremad_data/AudioWAV"

def extract_cremad_labels(data_directory):
    crema_directory_list = os.listdir(data_directory)
    file_emotion = []
    file_gender = []
    file_path = []

    female_ids = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021,
                  1024, 1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047, 1049, 1052, 1053, 1054,
                  1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079,
                  1082, 1084, 1089, 1091]

    emotion_mapping = {
        'SAD': 'SAD',
        'ANG': 'ANGRY',
        'DIS': 'DISGUST',
        'FEA': 'FEAR',
        'HAP': 'HAPPY',
        'NEU': 'NEUTRAL'
    }

    for file in crema_directory_list:
        file_path.append(os.path.join(data_directory, file))
        part = file.split('_')
        emotion_abbreviation = part[2]
        emotion = emotion_mapping.get(emotion_abbreviation, 'UNKNOWN')

        actor_id = int(part[0])
        gender = 'female' if actor_id in female_ids else 'male'

        file_emotion.append(emotion)
        file_gender.append(gender)

    emotion_df = pd.DataFrame({'path': file_path, 'emotion': file_emotion, 'gender': file_gender})
    return emotion_df

labels_df = extract_cremad_labels(audio_directory)

gender_counts = labels_df['gender'].value_counts()
print(gender_counts)

emotion_counts = labels_df['emotion'].value_counts()
print(emotion_counts)

gender_emotion_counts = pd.crosstab(labels_df['gender'], labels_df['emotion'])
print(gender_emotion_counts)

# emotion_df = pd.read_csv("crema_data.csv")
labels_df.to_csv("crema_data.csv", index=False)
