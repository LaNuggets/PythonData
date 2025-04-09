import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, validation_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ajouter le répertoire 'src' au chemin d'importation pour pouvoir utiliser 'data_preprocessing'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importer le module de prétraitement des données
from data_preprocessing import preprocess_data

# Fonction de conversion des booléens en entiers
def convert_bool_to_int(X):
    bool_columns = X.select_dtypes(include=[bool]).columns
    X[bool_columns] = X[bool_columns].astype(int)
    return X

def load_data(file_path):
    """Charge les données depuis un fichier CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_and_split_data(df):
    """
    Prétraiter les données et séparer en ensembles d'entraînement et de test.
    
    Retourne les ensembles X_train, X_test, y_train, y_test.
    """
    # Prétraitement des données
    X, y = preprocess_data(df)
    
    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def create_pipeline():
    """
    Crée un pipeline de traitement et d'entraînement avec prétraitement et modèle.
    
    Retourne un pipeline complet pour l'entraînement du modèle.
    """
    numeric_features = ['age', 'sibsp', 'parch', 'fare', 'pclass']
    # Mettre à jour la liste des caractéristiques catégorielles
    categorical_features = ['sex_male', 'embarked_S', 'embarked_Q']
    
    # Transformation des caractéristiques numériques
    numeric_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())  # Normalisation
    ])
    
    # Transformation des caractéristiques catégorielles
    categorical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder(drop='first'))  
    ])
    
    # Préprocesseur pour appliquer les transformations
    preprocessor = ColumnTransformer(
        transformers=[ 
            ('num', numeric_transformer, numeric_features), 
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Pipeline final avec traitement des booléens et entraînement du modèle
    model = Pipeline(steps=[ 
        ('convert_bool', FunctionTransformer(func=convert_bool_to_int, validate=False)),
        ('preprocessor', preprocessor), 
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    return model

def train_model(X_train, y_train):
    """
    Entraîne le modèle avec les données d'entraînement.
    
    Retourne le modèle entraîné.
    """
    # Afficher les colonnes de X_train pour vérifier l'alignement
    print("Colonnes de X_train :")
    print(X_train.columns)
    
    model = create_pipeline()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle en termes de performance (accuracy, f1, classification report).
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    
    # Sauvegarder la matrice de confusion en fichier PNG
    plt.savefig("confusion_matrix.png")
    print("Matrice de confusion sauvegardée sous 'confusion_matrix.png'")
    
    return accuracy, f1

def save_model_and_scaler(model, model_filename='model.pkl', scaler_filename='scaler.pkl'):
    """
    Sauvegarde le modèle et le scaler dans des fichiers.
    """
    # Sauvegarder le modèle complet (y compris le préprocesseur et le classificateur)
    joblib.dump(model, model_filename)
    
    # Sauvegarder uniquement le préprocesseur (ColumnTransformer) si nécessaire
    joblib.dump(model.named_steps['preprocessor'], scaler_filename)

def load_model_and_scaler(model_filename='model.pkl', scaler_filename='scaler.pkl'):
    """
    Charge le modèle et le scaler à partir des fichiers.
    """
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    return model, scaler

def plot_validation_curves(X_train, y_train, model):
    """
    Affiche une courbe de validation pour évaluer la performance du modèle avec différents hyperparamètres.
    """
    param_range = range(1, 11)
    train_scores, test_scores = validation_curve(
        model.named_steps['classifier'], X_train, y_train, param_name="C", param_range=param_range, cv=5, scoring='accuracy')
    
    plt.plot(param_range, train_scores.mean(axis=1), label="Train Score", color='blue')
    plt.plot(param_range, test_scores.mean(axis=1), label="Test Score", color='red')
    plt.title("Courbe de Validation pour différents paramètres de C")
    plt.xlabel("Valeur de C")
    plt.ylabel("Score")
    plt.legend()
    
    # Sauvegarder la courbe de validation en fichier PNG
    plt.savefig("validation_curve.png")
    print("Courbe de validation sauvegardée sous 'validation_curve.png'")

def predict_new_data(model, X_new):
    """
    Prédire la classe pour de nouvelles données.
    
    Paramètres :
    -----------
    model : pipeline
        Le modèle entraîné.
    
    X_new : pd.DataFrame
        Le DataFrame contenant les nouvelles données à prédire.
    
    Retourne :
    ---------
    predictions : array
        Les prédictions pour les nouvelles données.
    """
    predictions = model.predict(X_new)
    return predictions


