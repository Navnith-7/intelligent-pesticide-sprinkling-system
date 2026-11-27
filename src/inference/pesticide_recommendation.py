"""
38-Class Pesticide Prescription Engine & OpenCV Lesion Severity Estimator.
Computes:
  1. Disease-specific pesticide, active ingredient, pathogen type, and dosage/L.
  2. Foliar lesion severity (%) via HSV chlorophyll-vs-necrosis segmentation.
  3. Adaptive variable-rate spray pulse duration (1.0s Mild / 1.5s Moderate / 2.0s Severe).

NOTES / LIMITATIONS (read before relying on any value in this module):
  * PESTICIDE_DB is a SIMULATED lookup table for prototype demonstration. It is NOT
    validated agronomic advice and cites no external sources.
    [TODO: add agronomic sources for dosages before any real-world use]
  * Lesion severity is an UNVALIDATED HSV colour-threshold heuristic (HSV only - no LAB;
    not validated against expert labels; not meaningful for viral, mite or
    citrus-greening classes).
  * flow_rate_ml_per_sec = 12.0 is an ASSUMED, uncalibrated constant (never measured).
  * The hardware prototype dispenses water, not chemical pesticides.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np

PESTICIDE_DB: Dict[str, Dict[str, str]] = {
    # Apple
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "group": "Fungicide",
        "pathogen": "Fungal (Venturia inaequalis)",
        "pesticide": "Captan 50% WP / Myclobutanil 10% WP",
        "active_ingredient": "Captan + Myclobutanil",
        "dosage": "2.0 g / L water",
        "notes": "Apply protectant fungicide before rain events; rotate FRAC groups.",
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot",
        "group": "Fungicide",
        "pathogen": "Fungal (Botryosphaeria obtusa)",
        "pesticide": "Thiophanate-Methyl 70% WP / Mancozeb 75% WP",
        "active_ingredient": "Thiophanate-Methyl",
        "dosage": "1.5 g / L water",
        "notes": "Pair foliar spray with pruning of mummified fruit and dead cankers.",
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "group": "Fungicide",
        "pathogen": "Fungal (Gymnosporangium juniperi-virginianae)",
        "pesticide": "Myclobutanil 10% WP (Systhane) / Propiconazole",
        "active_ingredient": "Myclobutanil (Triazole)",
        "dosage": "1.0 g / L water",
        "notes": "Target early orange-yellow leaf lesions during spring spore release.",
    },
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Blueberry
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "group": "Fungicide",
        "pathogen": "Fungal (Podosphaera clandestina)",
        "pesticide": "Wettable Sulfur 80% WP / Tebuconazole 25.9% EC",
        "active_ingredient": "Elemental Sulfur / Tebuconazole",
        "dosage": "2.5 g / L water",
        "notes": "Avoid sulfur application when ambient temperature exceeds 32°C.",
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Corn (Maize)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Gray Leaf Spot (Cercospora)",
        "group": "Fungicide",
        "pathogen": "Fungal (Cercospora zeae-maydis)",
        "pesticide": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
        "active_ingredient": "Azoxystrobin + Difenoconazole",
        "dosage": "1.0 mL / L water",
        "notes": "Dual-mode strobilurin + triazole targets rectangular necrotic lesions.",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "group": "Fungicide",
        "pathogen": "Fungal (Puccinia sorghi)",
        "pesticide": "Mancozeb 75% WP / Propiconazole 25% EC",
        "active_ingredient": "Mancozeb / Propiconazole",
        "dosage": "2.0 g / L water",
        "notes": "Spray upper and lower leaf surfaces where rust pustules form.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "group": "Fungicide",
        "pathogen": "Fungal (Exserohilum turcicum)",
        "pesticide": "Zineb 75% WP / Pyraclostrobin 20% WG",
        "active_ingredient": "Zineb / Pyraclostrobin",
        "dosage": "2.0 g / L water",
        "notes": "Critical to treat cigar-shaped lesions before tasseling stage.",
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Grape
    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "group": "Fungicide",
        "pathogen": "Fungal (Guignardia bidwellii)",
        "pesticide": "Mancozeb 75% WP / Myclobutanil 10% WP",
        "active_ingredient": "Mancozeb + Myclobutanil",
        "dosage": "2.0 g / L water",
        "notes": "Prevent secondary spread from leaf spots to developing berries.",
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "group": "Fungicide",
        "pathogen": "Fungal Complex (Phaeomoniella chlamydospora)",
        "pesticide": "Carbendazim 50% WP + Copper Oxychloride 50% WP",
        "active_ingredient": "Carbendazim + Copper Oxychloride",
        "dosage": "1.5 g / L water",
        "notes": "Protect pruning wounds and apply targeted systemic fungicide.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Isariopsis Leaf Blight",
        "group": "Fungicide",
        "pathogen": "Fungal (Pseudocercospora vitis)",
        "pesticide": "Copper Hydroxide 77% WP / Bordeaux Mixture",
        "active_ingredient": "Copper Hydroxide",
        "dosage": "2.0 g / L water",
        "notes": "Effective against dark irregular fungal blight patches.",
    },
    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "group": "Insecticide + Bactericide",
        "pathogen": "Bacterial (Candidatus Liberibacter) via Citrus Psyllid",
        "pesticide": "Imidacloprid 17.8% SL + Streptocycline",
        "active_ingredient": "Imidacloprid + Streptomycin Sulfate",
        "dosage": "0.5 mL + 0.1 g / L water",
        "notes": "Suppresses Asian Citrus Psyllid vector to halt orchard transmission.",
    },
    # Peach
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "group": "Bactericide",
        "pathogen": "Bacterial (Xanthomonas campestris pv. pruni)",
        "pesticide": "Copper Oxychloride 50% WP + Streptocycline",
        "active_ingredient": "Copper Oxychloride + Streptomycin",
        "dosage": "2.5 g + 0.1 g / L water",
        "notes": "Use low-rate copper bactericide to prevent leaf shot-holing.",
    },
    "Peach___healthy": {
        "crop": "Peach",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Pepper (Bell)
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "group": "Bactericide",
        "pathogen": "Bacterial (Xanthomonas euvesicatoria)",
        "pesticide": "Copper Hydroxide 77% WP + Kasugamycin",
        "active_ingredient": "Copper Hydroxide + Kasugamycin",
        "dosage": "2.0 g / L water",
        "notes": "Halts water-soaked bacterial leaf spots during warm humid weather.",
    },
    "Pepper,_bell___healthy": {
        "crop": "Pepper (Bell)",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Potato
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "group": "Fungicide",
        "pathogen": "Fungal (Alternaria solani)",
        "pesticide": "Chlorothalonil 75% WP / Azoxystrobin 23% SC",
        "active_ingredient": "Chlorothalonil / Azoxystrobin",
        "dosage": "2.0 g / L water",
        "notes": "Targets concentric 'bullseye' target-ring foliar lesions.",
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "group": "Systemic Fungicide",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "pesticide": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold)",
        "active_ingredient": "Metalaxyl-M + Mancozeb",
        "dosage": "2.5 g / L water",
        "notes": "High-priority rapid pathogen; immediate targeted treatment advised.",
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Raspberry & Soybean
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    "Soybean___healthy": {
        "crop": "Soybean",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Squash
    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "group": "Fungicide",
        "pathogen": "Fungal (Podosphaera xanthii)",
        "pesticide": "Dinocap 48% EC (Karathane) / Hexaconazole 5% EC",
        "active_ingredient": "Dinocap / Hexaconazole",
        "dosage": "1.0 mL / L water",
        "notes": "Targets white powdery fungal colonies across leaf surface.",
    },
    # Strawberry
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "group": "Fungicide",
        "pathogen": "Fungal (Diplocarpon earliana)",
        "pesticide": "Captan 50% WP / Azoxystrobin 23% SC (Amistar)",
        "active_ingredient": "Captan / Azoxystrobin",
        "dosage": "2.0 g / L water",
        "notes": "Controls dark purple-brown scorch blotches and protects runners.",
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
    # Tomato
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "group": "Bactericide",
        "pathogen": "Bacterial (Xanthomonas perforans)",
        "pesticide": "Copper Oxychloride 50% WP + Streptocycline",
        "active_ingredient": "Copper Oxychloride + Streptomycin",
        "dosage": "2.5 g + 0.1 g / L water",
        "notes": "Limits spread of dark bacterial lesions during overhead moisture.",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "group": "Fungicide",
        "pathogen": "Fungal (Alternaria tomatophila)",
        "pesticide": "Mancozeb 75% WP / Difenoconazole 25% EC",
        "active_ingredient": "Mancozeb / Difenoconazole",
        "dosage": "2.0 g / L water",
        "notes": "Focus spray coverage on lower older foliage where blight begins.",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "group": "Systemic Fungicide",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "pesticide": "Cymoxanil 8% + Mancozeb 64% WP (Curzate)",
        "active_ingredient": "Cymoxanil + Mancozeb",
        "dosage": "2.5 g / L water",
        "notes": "Fast-spreading oomycete; treat immediately upon confirmation.",
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "group": "Fungicide",
        "pathogen": "Fungal (Passalora fulva)",
        "pesticide": "Chlorothalonil 75% WP / Copper Hydroxide",
        "active_ingredient": "Chlorothalonil",
        "dosage": "2.0 g / L water",
        "notes": "Improve greenhouse ventilation alongside foliar fungicide.",
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "group": "Fungicide",
        "pathogen": "Fungal (Septoria lycopersici)",
        "pesticide": "Zineb 75% WP / Azoxystrobin 23% SC",
        "active_ingredient": "Zineb / Azoxystrobin",
        "dosage": "2.0 g / L water",
        "notes": "Targets numerous small circular spots with dark margins.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mite",
        "group": "Acaricide / Miticide",
        "pathogen": "Arachnid Pest (Tetranychus urticae)",
        "pesticide": "Abamectin 1.9% EC / Spiromesifen 22.9% SC",
        "active_ingredient": "Abamectin / Spiromesifen",
        "dosage": "0.75 mL / L water",
        "notes": "Ensure thorough coverage on leaf undersides where mites colonize.",
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "group": "Fungicide",
        "pathogen": "Fungal (Corynespora cassiicola)",
        "pesticide": "Boscalid + Pyraclostrobin / Chlorothalonil 75% WP",
        "active_ingredient": "Pyraclostrobin + Boscalid",
        "dosage": "1.5 g / L water",
        "notes": "Controls expanding brown necrotic target spots.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus (TYLCV)",
        "group": "Vector Insecticide",
        "pathogen": "Viral (TYLCV) via Whitefly (Bemisia tabaci)",
        "pesticide": "Thiamethoxam 25% WG + Neem Oil 10000 ppm",
        "active_ingredient": "Thiamethoxam + Azadirachtin",
        "dosage": "0.5 g + 2.0 mL / L water",
        "notes": "Controls whitefly vectors to stop viral spread to healthy plants.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "group": "Vector Shield + Sanitizer",
        "pathogen": "Viral (ToMV — Mechanical & Aphid Vector)",
        "pesticide": "Virkon-S Disinfectant + Imidacloprid 17.8% SL",
        "active_ingredient": "Potassium Peroxymonosulfate + Imidacloprid",
        "dosage": "1.0 g / L water",
        "notes": "Sanitize pruning tools and suppress sap-sucking insect vectors.",
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "group": "None",
        "pathogen": "None (Healthy Foliage)",
        "pesticide": "No Pesticide Required (Pump Locked OFF)",
        "active_ingredient": "N/A",
        "dosage": "0.0 mL (Zero Discharge)",
        "notes": "Healthy leaf verified — skip spraying to conserve chemical.",
    },
}


def get_pesticide_recommendation(raw_class_name: str) -> Dict[str, Any]:
    """Return full agronomy prescription dictionary for any PlantVillage class."""
    info = PESTICIDE_DB.get(
        raw_class_name,
        {
            "crop": "Crop Foliage",
            "disease": raw_class_name.replace("___", " - ").replace("_", " "),
            "group": "Broad-Spectrum Fungicide",
            "pathogen": "Fungal / Bacterial Foliar Pathogen",
            "pesticide": "Mancozeb 75% WP + Copper Oxychloride 50% WP",
            "active_ingredient": "Mancozeb + Copper Oxychloride",
            "dosage": "2.0 g / L water",
            "notes": "General broad-spectrum foliar protectant.",
        },
    )
    return {"raw_class": raw_class_name, **info}


def estimate_leaf_severity_and_duration(
    roi_bgr: np.ndarray,
    is_diseased: bool,
    flow_rate_ml_per_sec: float = 12.0,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Estimate foliar lesion severity (%) on a 224x224 leaf ROI using HSV tissue ratios
    and compute an adaptive variable-rate spray pulse duration (1.0s / 1.5s / 2.0s).

    Returns:
      (metrics_dict, lesion_overlay_bgr)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return {
            "severity_pct": 0.0,
            "severity_grade": "None",
            "recommended_pulse_sec": 0.0,
            "est_volume_ml": 0.0,
        }, blank

    overlay = roi_bgr.copy()
    if not is_diseased:
        return {
            "severity_pct": 0.0,
            "severity_grade": "Healthy (0%)",
            "recommended_pulse_sec": 0.0,
            "est_volume_ml": 0.0,
        }, overlay

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Identify leaf tissue (green + yellow/brown foliage, excluding neutral gray/white background)
    leaf_mask = ((h_ch >= 8) & (h_ch <= 98) & (s_ch >= 28) & (v_ch >= 25)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    total_leaf_px = int(np.count_nonzero(leaf_mask))
    if total_leaf_px < 200:
        # Fallback when leaf fills nearly the full frame under extreme lighting
        total_leaf_px = roi_bgr.shape[0] * roi_bgr.shape[1]
        leaf_mask = np.full(roi_bgr.shape[:2], 255, dtype=np.uint8)

    # Discolored / chlorotic / necrotic lesions inside leaf tissue:
    # Yellow/orange/rust/brown hues (H: 5..30) OR dark necrotic spots (V < 78) inside leaf
    chlorotic_brown = (h_ch >= 5) & (h_ch <= 30) & (s_ch >= 35) & (leaf_mask > 0)
    dark_necrotic = (v_ch < 78) & (s_ch >= 25) & (leaf_mask > 0)
    lesion_mask = (chlorotic_brown | dark_necrotic).astype(np.uint8) * 255
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    lesion_px = int(np.count_nonzero(lesion_mask))
    severity_pct = float(np.clip((lesion_px / max(total_leaf_px, 1)) * 100.0, 2.0, 95.0))

    # Variable-Rate Precision Pulse Mapping (bounded safely between 1.0s and 2.0s)
    if severity_pct < 12.0:
        grade = "Mild Infection"
        pulse_sec = 1.0
    elif severity_pct <= 30.0:
        grade = "Moderate Infection"
        pulse_sec = 1.5
    else:
        grade = "Severe Infection"
        pulse_sec = 2.0

    # Render translucent red-amber lesion overlay for visual inspection
    red_layer = np.zeros_like(roi_bgr)
    red_layer[:, :] = (0, 60, 255)  # BGR orange-red
    mask_bool = lesion_mask > 0
    overlay[mask_bool] = cv2.addWeighted(roi_bgr, 0.55, red_layer, 0.45, 0)[mask_bool]

    return {
        "severity_pct": round(severity_pct, 1),
        "severity_grade": grade,
        "recommended_pulse_sec": pulse_sec,
        "est_volume_ml": round(pulse_sec * flow_rate_ml_per_sec, 1),
    }, overlay
