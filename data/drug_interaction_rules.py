"""
Small, documented sample dataset of drug–drug interaction rules.

IMPORTANT – READ BEFORE USING
- This is a TEACHING SAMPLE for a college demo. It contains only a handful
  of well-known interactions and is nowhere near complete.
- If a pair of drugs is NOT listed here, that does NOT mean the combination
  is safe. It only means this tiny sample has no rule for it.
- Every rule cites the official product labelling it was taken from.
  Labelling is updated over time, so always check the current version.
- No severity grades are given on purpose: grading would require a
  validated clinical knowledge base, which this project does not have.
- Nothing here is advice to start, stop or change any medication.

How a rule is structured
------------------------
    "drugs":         the two drug names (lower case, generic names)
    "concern":       a short label for the type of risk
    "explanation":   plain-language description of what the source says
    "context_labs":  lab tests (names as used in synthetic_patients.py)
                     that are relevant background for this concern
    "context_vitals": vital-sign keys (see VITAL_SIGNS) that are relevant
    "sources":       list of citations for the rule (title, section, link)
"""

# ---------------------------------------------------------------------------
# Sources. Defined once and reused, so each citation is written only once.
# Links point to the official label (or a search for it) on public sites.
# ---------------------------------------------------------------------------
_WARFARIN_LABEL = {
    "title": "Coumadin (warfarin sodium) – US FDA prescribing information",
    "section": "Drug Interactions – drugs that increase bleeding risk "
               "(antiplatelet agents, NSAIDs, SSRIs)",
    "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query=COUMADIN",
}
_SERTRALINE_LABEL = {
    "title": "Zoloft (sertraline) – US FDA prescribing information",
    "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query=ZOLOFT",
}
_SUMATRIPTAN_LABEL = {
    "title": "Imitrex (sumatriptan) tablets – US FDA prescribing information",
    "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query=IMITREX",
}
_ATORVASTATIN_LABEL = {
    "title": "Lipitor (atorvastatin) – US FDA prescribing information",
    "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query=LIPITOR",
}
_LISINOPRIL_LABEL = {
    "title": "Zestril (lisinopril) – US FDA prescribing information",
    "url": "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query=ZESTRIL",
}
_SALBUTAMOL_SMPC = {
    "title": "Ventolin (salbutamol) – Summary of Product Characteristics (HPRA, Ireland)",
    "section": "4.4 Special warnings and precautions for use – hypokalaemia",
    "url": "https://assets.hpra.ie/products/Human/24545/Licence_PA1077-049-012_07102021161810.pdf",
}
_FDA_IBUPROFEN_ASPIRIN = {
    "title": "US FDA science paper: Concomitant use of ibuprofen and aspirin – "
             "potential for attenuation of the anti-platelet effect of aspirin (September 2006)",
    "section": "Whole document",
    "url": "https://www.fda.gov/media/76636/download",
}


def _source(base, section=None):
    """Copy a base citation, optionally with a more specific section."""
    citation = dict(base)
    if section:
        citation["section"] = section
    return citation


# ---------------------------------------------------------------------------
# The rules.
# ---------------------------------------------------------------------------
INTERACTION_RULES = [
    {
        "drugs": ("warfarin", "aspirin"),
        "concern": "Increased bleeding risk",
        "explanation": (
            "Warfarin slows blood clotting and aspirin makes platelets less sticky. "
            "The warfarin label lists antiplatelet agents such as aspirin among drugs "
            "that increase the risk of bleeding when taken together with warfarin."
        ),
        "context_labs": ["INR", "Platelets", "Hemoglobin"],
        "context_vitals": [],
        "sources": [_source(_WARFARIN_LABEL)],
    },
    {
        "drugs": ("warfarin", "ibuprofen"),
        "concern": "Increased bleeding risk",
        "explanation": (
            "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID). The warfarin "
            "label lists NSAIDs among drugs that increase the risk of bleeding when "
            "taken together with warfarin."
        ),
        "context_labs": ["INR", "Platelets", "Hemoglobin"],
        "context_vitals": [],
        "sources": [_source(_WARFARIN_LABEL)],
    },
    {
        "drugs": ("warfarin", "sertraline"),
        "concern": "Increased bleeding risk",
        "explanation": (
            "Sertraline is a selective serotonin reuptake inhibitor (SSRI). Both the "
            "warfarin and sertraline labels state that SSRIs taken with warfarin can "
            "increase the risk of bleeding."
        ),
        "context_labs": ["INR", "Platelets", "Hemoglobin"],
        "context_vitals": [],
        "sources": [
            _source(_WARFARIN_LABEL),
            _source(_SERTRALINE_LABEL, "Warnings and Precautions – Increased Risk of Bleeding"),
        ],
    },
    {
        "drugs": ("sertraline", "sumatriptan"),
        "concern": "Serotonin syndrome risk",
        "explanation": (
            "Both drugs act on the serotonin system. The sertraline and sumatriptan "
            "labels warn that combining an SSRI with a triptan can cause serotonin "
            "syndrome, a potentially serious reaction (for example agitation, fast "
            "heart rate, high temperature, muscle twitching)."
        ),
        "context_labs": [],
        "context_vitals": ["heart_rate", "temperature"],
        "sources": [
            _source(_SERTRALINE_LABEL, "Warnings and Precautions – Serotonin Syndrome"),
            _source(_SUMATRIPTAN_LABEL, "Warnings and Precautions – Serotonin Syndrome"),
        ],
    },
    {
        "drugs": ("sertraline", "ibuprofen"),
        "concern": "Increased bleeding risk",
        "explanation": (
            "SSRIs such as sertraline can affect platelet function. The sertraline "
            "label warns that taking it together with NSAIDs such as ibuprofen may "
            "add to the risk of bleeding."
        ),
        "context_labs": ["Platelets", "Hemoglobin"],
        "context_vitals": [],
        "sources": [_source(_SERTRALINE_LABEL, "Warnings and Precautions – Increased Risk of Bleeding")],
    },
    {
        "drugs": ("salbutamol", "prednisolone"),
        "concern": "Low potassium (hypokalaemia) risk",
        "explanation": (
            "Salbutamol can lower blood potassium. Its product information states that "
            "this effect may be increased by steroids (such as prednisolone), "
            "diuretics, xanthine derivatives and by low oxygen levels (hypoxia), and "
            "advises monitoring potassium in these situations."
        ),
        "context_labs": ["Potassium"],
        "context_vitals": ["spo2"],
        "sources": [_source(_SALBUTAMOL_SMPC)],
    },
    {
        "drugs": ("atorvastatin", "clarithromycin"),
        "concern": "Muscle damage (myopathy / rhabdomyolysis) risk",
        "explanation": (
            "Clarithromycin blocks the enzyme that breaks down atorvastatin, so "
            "atorvastatin levels rise. The atorvastatin label warns that this raises "
            "the risk of muscle damage."
        ),
        "context_labs": ["Creatine kinase (CK)"],
        "context_vitals": [],
        "sources": [_source(_ATORVASTATIN_LABEL, "Warnings and Precautions – Myopathy and "
                          "Rhabdomyolysis; Drug Interactions")],
    },
    {
        "drugs": ("lisinopril", "ibuprofen"),
        "concern": "Kidney function and blood-pressure effects",
        "explanation": (
            "The lisinopril label states that NSAIDs such as ibuprofen may reduce the "
            "blood-pressure-lowering effect of lisinopril, and in some patients "
            "(for example older adults or those with reduced kidney function) the "
            "combination may worsen kidney function."
        ),
        "context_labs": ["Creatinine", "Potassium"],
        "context_vitals": ["systolic_bp"],
        "sources": [_source(_LISINOPRIL_LABEL, "Drug Interactions – Non-Steroidal "
                          "Anti-Inflammatory Agents")],
    },
    {
        "drugs": ("aspirin", "ibuprofen"),
        "concern": "Reduced antiplatelet effect of low-dose aspirin",
        "explanation": (
            "The US FDA reported that ibuprofen can interfere with the platelet "
            "effect of low-dose aspirin, which may make aspirin less effective at "
            "its heart-protecting purpose."
        ),
        "context_labs": ["Platelets"],
        "context_vitals": [],
        "sources": [_source(_FDA_IBUPROFEN_ASPIRIN)],
    },
]


# ---------------------------------------------------------------------------
# Alternative names that map to the generic name used in the rules.
# Only a few common international / chemical names are included.
# ---------------------------------------------------------------------------
DRUG_ALIASES = {
    "albuterol": "salbutamol",               # US name for salbutamol
    "acetylsalicylic acid": "aspirin",
    "asa": "aspirin",
}

# Drugs that appear in the synthetic patients' lists but have no rule here.
# They are "known" so the checker can say "no rule in this sample" instead
# of "unknown drug". This is NOT a statement that they never interact.
OTHER_KNOWN_DRUGS = {
    "metformin",
    "ceftriaxone",
    "omeprazole",
    "levothyroxine",
}
