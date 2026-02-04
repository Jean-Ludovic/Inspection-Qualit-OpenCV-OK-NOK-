# 🏭 Visual Quality Inspection — OpenCV + Streamlit

Mini-application d’inspection visuelle industrielle simulant un **contrôle qualité sur ligne de production**
(chocolats, biscuits, bonbons, produits alimentaires).

L’application analyse des images de produits, détecte des défauts visuels
et fournit une **décision OK / NOK explicable**, comme dans un contexte industriel réel.

---

## 🎯 Objectif du projet

Ce projet a pour but de démontrer :
- la **détection et segmentation d’objets** avec OpenCV
- l’**extraction de features interprétables**
- la **prise de décision industrielle (OK / NOK)**
- la **robustesse face au bruit et à l’éclairage**
- une **approche explicable**, compréhensible par un opérateur qualité

Il est volontairement **simple, lisible et déployable rapidement**, comme attendu dans un contexte d’entretien technique.

---

## ⚙️ Fonctionnalités principales

### 🔍 Inspection visuelle
- Upload d’une **image unique ou d’un lot d’images**
- Segmentation du produit vs fond :
  - conversion HSV
  - seuillage
  - opérations morphologiques
- Détection du **produit principal** (plus grand contour)

### 📐 Extraction de features
- **Surface** du produit
- **Circularité** (forme régulière / irrégulière)
- **Couleur moyenne (Lab)** → robustesse à l’éclairage
- **Texture** :
  - variance des niveaux de gris
  - variance du Laplacian (netteté / défauts)

### ✅ Décision qualité
- Résultat **OK / NOK**
- **Score de confiance**
- Explication lisible :
  - *trop sombre*
  - *forme irrégulière*
  - *surface trop petite*
  - *texture anormale*, etc.

### 🧪 Mode calibration (bonus)
- Upload d’images **OK (référence)**
- Calcul automatique des seuils :
  - moyenne ± k × écart-type
- Simulation réaliste d’un **réglage de ligne de production**

### 📊 Analyse & export
- Affichage des :
  - masques
  - contours
  - ROI (zones d’intérêt)
- Export des résultats en **CSV**

---

## 🛠️ Stack technique

- **Python**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Streamlit**

---

## 🚀 Installation & lancement

### 1. Installer les dépendances
```bash
pip install streamlit opencv-python numpy pandas

merciiii