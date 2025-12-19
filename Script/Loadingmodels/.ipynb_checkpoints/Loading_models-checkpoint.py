import torch
import torch.nn as nn
from torchvision import datasets, models
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "PyramidNetPyTorch"))
from PyramidNet import PyramidNet

def initialize_model(model_name, num_classes, architecture, target_params=600000, pretrained=True, feature_extract=False, activation_fct=nn.ReLU(inplace=False), p_dropout=0.2):
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model_ft = models.mobilenet_v3_large(weights=weights)
        num_ftrs = model_ft.classifier[0].in_features

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model_ft = models.mobilenet_v2(weights=weights)
        num_ftrs = model_ft.classifier[1].in_features 
        
    elif model_name == "resnet18":
        weights = models.ResNet18_Weigths.IMAGENET1K_V2 if pretrained else None
        model_ft = models.resnet18(weights=weights)
        num_ftrs = model_ft.fc.in_features

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model_ft = models.resnet50(weights=weights)
        num_ftrs = model_ft.fc.in_features

    elif model_name == "efficientnet_v2":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model_ft = models.efficientnet_v2_s(weights=weights)
        num_ftrs = model_ft.classifier[1].in_features

    elif model_name == "pyramidnet":
        model_ft = PyramidNet(dataset='imagenet', depth=101, alpha=360, num_classes=num_classes)
        
        # Charger les poids pré-entraînés
        checkpoint_path = os.path.join(os.getcwd(), "PyramidNetPyTorch", "pyrmaidnet101_360.pth")
        weights = torch.load(checkpoint_path, map_location="cpu")

        # --- Supprimer la tête fc du checkpoint pour éviter les conflits
        if 'fc.weight' in weights:
            del weights['fc.weight']
            del weights['fc.bias']
        
        # Charger dans le modèle
        model_ft.load_state_dict(weights, strict=False)
        
        num_ftrs= model_ft.fc.in_features

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # --- Optionnel : Freeze des features (fine-tuning léger)
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False

    # --- Remplacer la tête de classification
    if architecture == "one_layer":
        ftrs_classif = int((target_params - num_classes) / (num_ftrs + num_classes + 1))
        classifier = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=ftrs_classif, bias=True),
            activation_fct,
            nn.Dropout(p=p_dropout, inplace=True),
            nn.Linear(in_features=ftrs_classif, out_features=num_classes, bias=True)
        )
    elif architecture == "two_layers":
        ratio = 2  # pour éviter que f2 soit trop petit

        #résolution equation du second ordre
        a = 1/ratio
        b = num_ftrs + num_classes/ratio
        c = -target_params
        # formule du discriminant
        disc = b**2 - 4*a*c
        if disc < 0:
            raise ValueError("Discriminant négatif, ajuste la cible ou le ratio")
        
        ftrs_classif1 = int((-b + math.sqrt(disc)) / (2*a))
        ftrs_classif2 = int(ftrs_classif1 / ratio)
        classifier = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=ftrs_classif1, bias=True),
            activation_fct,
            nn.Linear(in_features=ftrs_classif1, out_features=ftrs_classif2, bias=True),
            activation_fct,
            nn.Dropout(p=p_dropout, inplace=True),
            nn.Linear(in_features=ftrs_classif2, out_features=num_classes, bias=True)
        )

    # --- Adapter selon le modèle
    if "mobilenet" in model_name:
        model_ft.classifier = classifier
    elif "resnet" in model_name or "pyramidnet" in model_name:
        model_ft.fc = classifier
    elif "efficientnet" in model_name:
        model_ft.classifier = classifier
    else:
        raise ValueError(f"Architecture {model_name} non prise en charge")

    return model_ft