# Figures de pipeline

Deux figures conçues pour un article en anglais, inspirées du style fourni :
fond blanc, aplats pastel, contours fins, flèches orthogonales et équations.

| Figure | PDF vectoriel pour l'article | SVG modifiable | Aperçu PNG |
| --- | --- | --- | --- |
| Rectified flow matching | [PDF](hand_flow_base/rectified_flow_matching.pdf) | [SVG](hand_flow_base/rectified_flow_matching.svg) | [PNG](hand_flow_base/rectified_flow_matching.png) |
| VAE + latent flow matching | [PDF](latent_hand_flow/latent_flow_matching.pdf) | [SVG](latent_hand_flow/latent_flow_matching.svg) | [PNG](latent_hand_flow/latent_flow_matching.png) |

Les deux figures mesurent **180 × 151 mm**, pour une insertion sur la largeur
d'un article à deux colonnes (`figure*`). Le texte est principalement en 8 pt
à cette taille. Les PDF et SVG sont entièrement vectoriels ; les PDF incorporent
les polices TrueType. Les PNG sont exportés à **600 dpi**. Préférer les PDF pour
LaTeX et éviter de réduire ces figures à une seule colonne.

## Contenu et hypothèses

**Rectified flow matching.** Le panneau (a) montre l'apprentissage initial avec
des couples indépendants bruit / mouvement, l'interpolation linéaire et la perte
sur la vitesse. Le panneau (b) montre une étape de reflow : un enseignant figé
intègre l'ODE et fournit un mouvement synthétique, conservé avec **le bruit qui
l'a engendré**. Le modèle rectifié est entraîné sur l'interpolation de ce couple.
Le panneau (c) intègre le champ rectifié pour générer un mouvement.

**Latent flow matching.** Le VAE est d'abord entraîné avec une reconstruction
quadratique et une régularisation KL pondérée par un coefficient symbolique
`beta`. L'encodeur et le décodeur sont ensuite figés. Le modèle de flow apprend à
transporter un bruit gaussien vers les latents échantillonnés dans le posterior
du VAE. À la génération, seul le modèle de vitesse et le décodeur sont utilisés.
La structure et la dimension du latent restent volontairement non spécifiées.

Notation commune : `y` est le texte, `c = T5(y)` son encodage figé, `t` le temps
du flow (distinct des trames temporelles du mouvement), `T` le nombre de trames,
et `198` la dimension des paramètres MANO bimanuels (translations + rotations
6D). Les figures utilisent la convention mathématique `[T, 198]` ; le U-Net du
code reçoit les tenseurs transposés `[B, 198, T]`. Les espérances des pertes de
flow portent sur les couples d'extrémités, leurs conditions et le temps uniforme.
Les flèches pointillées représentent le conditionnement ; les flèches pleines
représentent les données et les calculs des pertes.

La mention **« AdaLN or cross-attention » désigne deux variantes possibles**
du U-Net proposé. Pour AdaLN, le temps encodé et le texte agrégé modulent les
normalisations ; pour la cross-attention textuelle, les caractéristiques du
mouvement sont les requêtes et les tokens du texte fournissent clés et valeurs,
avec un conditionnement temporel séparé. La figure ne prescrit pas leur emploi
simultané. L'illustration miniature du U-Net indique l'encodeur, le goulot,
le décodeur et une connexion de saut, sans fixer le nombre de résolutions.

## Correspondance avec le code inspecté

- `utils/reflow_trainer.py` implémente l'interpolation
  `x_t = (1-t) x_0 + t x_1`, la cible `x_1 - x_0` et la MSE masquée.
- `utils/reflow_sampler.py` intègre le champ par Euler, du bruit vers les données,
  avec classifier-free guidance configurable.
- `models/temporal_unet_1d/model.py` et `modules.py` implémentent actuellement
  **FiLM avec GroupNorm**, et non AdaLN ou cross-attention textuelle.
- `models/mdit/blocks.py` comporte AdaLN pour le temps et cross-attention pour
  le texte, mais appartient au backbone MotionDiT, distinct du U-Net demandé.
- L'étape de reflow et le VAE / flow latent ne sont pas implémentés dans les
  fichiers inspectés. Ces parties des figures présentent donc le pipeline
  **proposé**, sans constituer une description d'expériences déjà exécutées.

Les masques de padding, la normalisation, le dropout de conditionnement, la CFG
et les détails de l'optimiseur sont omis pour conserver un schéma lisible.
Aucune perte physique supplémentaire n'est supposée.

## Régénération et insertion

Depuis la racine du dépôt, avec Python 3 et Matplotlib :

```bash
python3 materials/figures/generate_pipelines.py
```

Le script reconstruit les six exports et vérifie que les libellés des boîtes ne
débordent pas. Les positions sont exprimées en millimètres, les tailles de
police en points ; couleurs, textes et équations sont modifiables dans le script.
Des aperçus de travail à 180 dpi sont également écrits dans `/tmp`.

[captions.tex](captions.tex) fournit deux environnements `figure*` et leurs
légendes en anglais. Depuis un document compilé à la racine du dépôt :

```latex
% Dans le préambule :
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
% Dans le document :
\input{materials/figures/captions.tex}
```

Adapter les chemins si le manuscrit est compilé depuis un autre répertoire.
