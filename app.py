from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Chargement des paramètres
W = np.load("W.npy")
b = np.load("b.npy")
mu = np.load("mu.npy")
sigma = np.load("sigma.npy")

with open('categories.pkl', 'rb') as f:
    cat_cols = pickle.load(f)

# Dictionnaire des options pour les champs catégoriels
category_options = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 
                 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 
                 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 
                      'Separated', 'Widowed', 'Married-spouse-absent', 
                      'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                  'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                  'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                  'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                  'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                    'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Black', 'Other'],
    'sex': ['Male', 'Female'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                      'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 
                      'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 
                      'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                      'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 
                      'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 
                      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 
                      'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 
                      'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
}

# Ordre des colonnes
features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

def preprocess_input(form_data):
    x = []
    for feat in features:
        val = form_data.get(feat, '')
        
        # Traitement spécial pour les champs catégoriels
        if feat in category_options:
            # Trouver l'index de la valeur sélectionnée
            options = category_options[feat]
            try:
                val = options.index(val) if val in options else 0
            except:
                val = 0
        else:
            # Pour les champs numériques
            try:
                val = float(val) if val else 0.0
            except:
                val = 0.0
        
        x.append(val)
    
    x = np.array(x)
    x_norm = (x - mu) / sigma
    return x_norm.reshape(1, -1)

def predire_flask(X):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return (A >= 0.5).astype(int).flatten()[0]

@app.route("/", methods=["GET", "POST"])
def index():
    resultat = None
    if request.method == "POST":
        try:
            X_input = preprocess_input(request.form)
            pred = predire_flask(X_input)
            resultat = "Le revenu est supérieur à 50K" if pred == 1 else "Le revenu est inférieur ou égal à 50K"
        except Exception as e:
            resultat = f"Erreur: {str(e)}"
    return render_template("index.html", 
                         resultat=resultat,
                         category_options=category_options,
                         features=features)
import os

port = int(os.environ.get("PORT", 5000))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)