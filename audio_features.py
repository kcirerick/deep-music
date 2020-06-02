import numpy as np
import pandas as pd
import librosa
import os



#https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc

 
genres = {
    name : i for i, name in enumerate(
    ['blues', 'metal', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'pop', 'reggae', 'rock' ]
)}


def extract_audio_features(audio_path, n_mfcc=40):
    audio, sample_rate = librosa.load(audio_path)


    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed







def create_feature_df(audio_dir, df_path='./'):
    features = []

    for subdir, genre, files in os.walk(audio_dir):
        for filename in files:
            audio_path = subdir + os.sep + filename

            x = extract_audio_features(audio_path)
            label_arr = np.zeros((len(genres)), 1)
            label_arr[genres[genre], 0] =  1.0
            features.append((filename, x, y))


    features_df = pd.DataFrame(features, columns=['feature', 'label'])

    feature_df.to_pickle(os.path.join(df_path, 'features_df.pkl'))



def get_features(df_path='./'):
    full_path = os.path.join(df_path, 'features_df.pkl')
    if not os.exists(full_path):
        
        create_feature_df('./genres/')

    df = pad.read_pickle()

    X = np.array(df.feature.tolist())

    Y = np.array(df.label.tolist())

    return X, Y




