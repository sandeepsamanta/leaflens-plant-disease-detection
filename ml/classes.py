"""The 38 PlantVillage classes, display names, and the PlantDoc -> PlantVillage label map."""

CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

_PLANT_NAMES = {
    "Cherry_(including_sour)": "Cherry", "Corn_(maize)": "Corn (maize)", "Pepper,_bell": "Bell pepper",
}
_DISEASE_NAMES = {
    "Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot",
    "Common_rust_": "Common rust",
    "Esca_(Black_Measles)": "Esca (black measles)",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight (Isariopsis leaf spot)",
    "Haunglongbing_(Citrus_greening)": "Huanglongbing (citrus greening)",
    "Spider_mites Two-spotted_spider_mite": "Spider mites (two-spotted)",
    "Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus",
    "Tomato_mosaic_virus": "Mosaic virus",
}


def split_label(label: str) -> tuple[str, str]:
    plant, disease = label.split("___")
    plant = _PLANT_NAMES.get(plant, plant.replace("_", " "))
    disease = _DISEASE_NAMES.get(disease, disease.replace("_", " "))
    return plant, disease[0].upper() + disease[1:]


def display_name(label: str) -> str:
    plant, disease = split_label(label)
    return f"{plant} - {disease}"


def is_healthy(label: str) -> bool:
    return label.endswith("___healthy")


# PlantDoc (real field photos, 27 categories) -> the PlantVillage class it corresponds to.
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy", "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy", "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy", "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus", "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold", "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot",
}
assert all(v in CLASS_INDEX for v in PLANTDOC_TO_PV.values())
