# Efficient Resolution Cascade

Recherche menée pendant un stage Mitacs à l'ÉTS Montréal : réduire le coût de calcul d'un classifieur d'images sans perdre en précision.

**Principe :** une image est d'abord classée en basse résolution par un petit réseau. Si la confiance est sous un seuil, elle passe au réseau suivant, plus profond et en plus haute résolution : ResNet18 (32 px) → ResNet34 (96 px) → ResNet50 (224 px). Les seuils sont optimisés par recherche sur grille.

**Résultats sur Imagenette :**
- même précision que ResNet34 avec **3 fois moins de FLOPs** ;
- précision équivalente à ResNet50 avec **2,3 fois moins de FLOPs**.

Modélisation du coût, état de l'art et résultats détaillés : [Study.pdf](Study.pdf).

## Utilisation

```bash
# <imagenette> contient train/ et val/ ; un modèle est entraîné par résolution
python main.py --training --datadir <imagenette> --model resnet18 --resolution 32 --epochs <n> --batch <n> --lr <lr> --savepath models/
python main.py --cascade --datadir <imagenette> --modelsfolder models/ --ccresults results/
```
**Stack :** Python · PyTorch · torchvision · NumPy
