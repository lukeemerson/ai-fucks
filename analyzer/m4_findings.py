from dataclasses import dataclass

@dataclass
class Finding:
    name: str
    tier: int
    tier_label: str
    description: str
    m4_action: str

FINDINGS = {
    "cardiomegaly": Finding(
        name="Cardiomegaly",
        tier=2,
        tier_label="Should recognize",
        description="Cardiothoracic ratio > 0.50 on PA film",
        m4_action="Assess for CHF, valvular disease; order Echo, BNP",
    ),
    "pneumothorax": Finding(
        name="Possible Pneumothorax",
        tier=1,
        tier_label="MUST recognize",
        description="Peripheral lucency with absent lung markings",
        m4_action="Check tracheal deviation, breath sounds; if tension → needle decompression 2nd ICS MCL; order stat CXR/CT",
    ),
    "pleural_effusion": Finding(
        name="Pleural Effusion",
        tier=2,
        tier_label="Should recognize",
        description="Basal opacification with meniscus / blunted costophrenic angle",
        m4_action="Grade size; consider thoracentesis if large/symptomatic; work up exudate vs transudate (Light's criteria)",
    ),
    "pulmonary_edema": Finding(
        name="Pulmonary Edema",
        tier=1,
        tier_label="MUST recognize",
        description="Bilateral diffuse haziness, perihilar bat-wing pattern",
        m4_action="Assess volume status, BNP, Echo; start diuresis (furosemide); monitor O₂ sat",
    ),
    "consolidation": Finding(
        name="Consolidation / Pneumonia",
        tier=2,
        tier_label="Should recognize",
        description="Focal airspace opacity with or without air bronchograms",
        m4_action="Determine CAP vs HAP; start empiric antibiotics per guidelines; follow up film in 6 wks",
    ),
    "atelectasis": Finding(
        name="Atelectasis",
        tier=2,
        tier_label="Should recognize",
        description="Linear or lobar opacity with volume loss / tracheal shift",
        m4_action="Encourage deep breathing/incentive spirometry; rule out obstructing lesion if lobar; chest PT",
    ),
    "emphysema": Finding(
        name="Emphysema / Hyperinflation",
        tier=3,
        tier_label="Should know about",
        description="Flat diaphragm, barrel chest, increased AP diameter",
        m4_action="Confirm COPD history; PFTs; bronchodilators; smoking cessation counseling",
    ),
    "focal_opacity": Finding(
        name="Focal Opacity / Nodule",
        tier=3,
        tier_label="Should know about",
        description="Focal high-density region — nodule, mass, or infiltrate",
        m4_action="Apply Fleischner Society guidelines for nodule follow-up; CT chest; consider PET if >8mm",
    ),
}

TIER_COLORS = {1: "\033[91m", 2: "\033[93m", 3: "\033[96m"}
RESET = "\033[0m"

def tier_badge(tier: int) -> str:
    color = TIER_COLORS.get(tier, "")
    label = {1: "Tier 1 ▲ MUST RECOGNIZE", 2: "Tier 2 ● Should Recognize", 3: "Tier 3 ○ Should Know"}.get(tier, "")
    return f"{color}[{label}]{RESET}"
