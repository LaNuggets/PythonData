import sys
import os
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend 'Agg' pour l'environnement non interactif
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

# Ajouter 'src' au chemin d'importation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Maintenant, vous pouvez importer depuis src.data_preprocessing
from data_preprocessing import preprocess_data
from modeling import load_data, train_model, evaluate_model, save_model_and_scaler, load_model_and_scaler, plot_validation_curves, predict_new_data

def main():
    # Définir le chemin d'accès aux données
    data_path = 'data/train.csv'

    # Vérifier si le fichier existe avant de le charger
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier '{data_path}' n'a pas été trouvé.")
        return

    # Charger les données
    df = load_data(data_path)
    
    # Prétraiter les données et séparer X (features) et y (labels)
    X, y = preprocess_data(df)
    
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = train_model(X_train, y_train)
    
    # Évaluer le modèle
    evaluate_model(model, X_test, y_test)
    
    # Sauvegarder le modèle et le scaler
    save_model_and_scaler(model, model_filename='model.pkl', scaler_filename='scaler.pkl')
    print("Modèle et scaler enregistrés avec succès.")
    
    # Charger à nouveau le modèle et le scaler
    loaded_model, loaded_scaler = load_model_and_scaler(model_filename='model.pkl', scaler_filename='scaler.pkl')
    print("Modèle et scaler chargés avec succès.")
    
    # Utiliser loaded_model et loaded_scaler pour des prédictions, des évaluations, etc.
    # Evaluer le modèle chargé avec les données de test :
    evaluate_model(loaded_model, X_test, y_test)

    # Afficher la courbe de validation (optionnel)
    plot_validation_curves(X_train, y_train, loaded_model)

    # Prédiction avec de nouvelles données (en utilisant les 5 premières lignes de X_test)
    X_new = X_test[:5]
    predictions = predict_new_data(loaded_model, X_new)
    print("Prédictions pour les nouvelles données:", predictions)

    # Sauvegarder et afficher les matrices de confusion et courbes (optionnel)
    save_and_display_images()

def save_and_display_images():
    """Sauvegarder et afficher les images (matrices de confusion et courbes)"""
    # Afficher la matrice de confusion
    img = mpimg.imread('confusion_matrix.png')
    plt.imshow(img)
    plt.axis('off')  # Masquer les axes pour une meilleure visualisation
    plt.savefig("confusion_matrix.png")  # Sauvegarder l'image si nécessaire

    # Afficher la courbe de validation
    img = mpimg.imread('validation_curve.png')
    plt.imshow(img)
    plt.axis('off')  # Masquer les axes pour une meilleure visualisation
    plt.savefig("validation_curve.png")  # Sauvegarder l'image si nécessaire

    # Ces lignes ne seront pas exécutées dans un terminal, mais elles permettent de sauvegarder les graphiques
    print("Matrice de confusion et courbe de validation sauvegardées avec succès.")

if __name__ == '__main__':
    main()

