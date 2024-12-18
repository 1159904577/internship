import joblib
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import skew, kurtosis
import xgboost

class bin_predict:
    def __init__(self, input_domain):
        self.input_domain = input_domain
        self.binary_model = joblib.load("artifacts/binary/binary_classification_model.pkl")
        self.unigrams = joblib.load("artifacts/binary/unigram_vectorizer.pkl")
        self.bigrams = joblib.load("artifacts/binary/bigram_vectorizer.pkl")
        self.trigrams = joblib.load("artifacts/binary/trigram_vectorizer.pkl")
        self.scaler = joblib.load("artifacts/binary/scaler.pkl")

    def count_features(self, domain):
        L = len(domain)
        consonant_count = sum(1 for char in domain if char in 'bcdfghjklmnpqrstvwxyz')
        Rc = consonant_count / len(domain) if len(domain) > 0 else 0

        letter_count = sum(1 for char in domain if char.isalpha())
        Rl = letter_count / len(domain) if len(domain) > 0 else 0

        number_count = sum(1 for char in domain if char.isdigit())
        Rn = number_count / len(domain) if len(domain) > 0 else 0

        vowel_count = sum(1 for char in domain if char in 'aeiou')
        Rv = vowel_count / len(domain) if len(domain) > 0 else 0

        symbolic_count = sum(1 for char in domain if not char.isalnum())
        Rs = symbolic_count / len(domain) if len(domain) > 0 else 0

        return L, Rc, Rv, Rn, Rl, Rs

    def calc_custom_features(self):

        features = []
        parts = self.input_domain.split('.')
        subdomain = '.'.join(parts[:-2]) if len(parts) >= 3 else ''
        sld = parts[-2]
        tld = parts[-1]

        N = 3 if subdomain else 2

        consonants = re.findall(r'[^aeiou\d\s\W]+', self.input_domain)
        LCc = max(len(consonant) for consonant in consonants) if consonants else 0
        numbers = re.findall(r'\d+', self.input_domain)
        LCn = max(len(number) for number in numbers) if numbers else 0
        vowels = re.findall(r'[aeiou]+', self.input_domain)
        LCv = max(len(vowel) for vowel in vowels) if vowels else 0

        L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld = self.count_features(tld)
        L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld = self.count_features(sld)
        L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub = self.count_features(subdomain) if subdomain else (0, 0, 0, 0, 0, 0)

        features.append([N, LCc, LCv, LCn, 
                        L_tld, Rc_tld, Rv_tld, Rn_tld, Rl_tld, Rs_tld,
                        L_sld, Rc_sld, Rv_sld, Rn_sld, Rl_sld, Rs_sld,
                        L_sub, Rc_sub, Rv_sub, Rn_sub, Rl_sub, Rs_sub])
        
        feature_columns = ['N', 'LCc', 'LCv', 'LCn',
                    'L_tld', 'Rc_tld', 'Rv_tld', 'Rn_tld', 'Rl_tld','Rs_tld',
                    'L_sld', 'Rc_sld', 'Rv_sld', 'Rn_sld', 'Rl_sld','Rs_sld',
                    'L_sub', 'Rc_sub', 'Rv_sub', 'Rn_sub', 'Rl_sub', 'Rs_sub']
        
        return pd.DataFrame(features, columns=feature_columns)
    
    def ngrams_features_per_sample(self, matrix, prefix):
        # Convert sparse matrix to dense
        ngram_frequencies = matrix.toarray()

        # Initialize a list to store feature dictionaries
        features_list = []

        # Loop through each sample
        for sample_frequencies in ngram_frequencies:
            features = {}

            # Count distinct n-grams (non-zero frequencies)
            if np.count_nonzero(sample_frequencies) > 0:  # Avoid NaN errors if no n-grams are present

                features[f'{prefix}-MEAN'] = np.mean(sample_frequencies)
                features[f'{prefix}-VAR'] = np.var(sample_frequencies)
                features[f'{prefix}-PVAR'] = np.var(sample_frequencies, ddof=0)  # Population variance
                features[f'{prefix}-STD'] = np.std(sample_frequencies)
                features[f'{prefix}-PSTD'] = np.std(sample_frequencies, ddof=0)  # Population std deviation
                features[f'{prefix}-SKE'] = skew(sample_frequencies)
                features[f'{prefix}-KUR'] = kurtosis(sample_frequencies)
            else:
                features[f'{prefix}-MEAN'] = 0
                features[f'{prefix}-VAR'] = 0
                features[f'{prefix}-PVAR'] = 0
                features[f'{prefix}-STD'] = 0
                features[f'{prefix}-PSTD'] = 0
                features[f'{prefix}-SKE'] = 0  # Skewness is 0 for no data
                features[f'{prefix}-KUR'] = 0  # Kurtosis is 0 for no data

            # Append features for this sample to the list
            features_list.append(features)

        # Convert the list of feature dictionaries into a DataFrame
        return pd.DataFrame(features_list)

    def calc_ngrams(self):
        unigrams_matrix = self.unigrams.transform([self.input_domain])
        bigrams_matrix = self.bigrams.transform([self.input_domain])
        trigrams_matrix = self.trigrams.transform([self.input_domain])

        # Extract features
        unigrams_features_df = self.ngrams_features_per_sample(unigrams_matrix, prefix='UNI')
        bigrams_features_df = self.ngrams_features_per_sample(bigrams_matrix, prefix='BI')
        trigrams_features_df = self.ngrams_features_per_sample(trigrams_matrix, prefix='TRI')

        # Concatenate bigrams and trigrams features
        df_ngrams_features = pd.concat([unigrams_features_df, bigrams_features_df, trigrams_features_df], axis=1)

        return df_ngrams_features
    
    def scaling(self):
        X = pd.concat([self.calc_custom_features(), self.calc_ngrams()], axis=1)

        return self.scaler.transform(X)
    
    def predict(self):
        label = self.binary_model.predict(self.scaling())
        return label[0]

class multi_predict:
    def __init__(self, input_domain):
        self.input_domain = input_domain
        self.max_sequence_length = 50
        self.multiclass_model = joblib.load("artifacts/multi/multiclass_classification_model.pkl")
        self.tokenizer = joblib.load("artifacts/multi/tokenizer.pkl")
        self.encoder = joblib.load("artifacts/multi/encoder.pkl")

    def preprocess(self):
        sequence = self.tokenizer.texts_to_sequences([self.input_domain])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
        return padded_sequence

    def predict(self):
        processed_input = self.preprocess()

        prediction = self.multiclass_model.predict(processed_input)

        top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Indices of top 3 predictions (descending)
        top_3_probs = prediction[0][top_3_indices]  # Get the probabilities of the top 3
        top_3_classes = self.encoder.inverse_transform(top_3_indices)  # Get the class labels for top 3

        # Combine the class labels and their probabilities
        top_3_predictions = [{"class": top_3_classes[i], "probability": float(top_3_probs[i])} for i in range(3)]

        return top_3_predictions  # Return the top 3 predictions as a list of dictionaries