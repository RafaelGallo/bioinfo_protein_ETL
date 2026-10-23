"""Transform protein sequences into model features.

Transforma sequências de proteínas em features para os modelos: composição de
aminoácidos, propriedades físico-químicas e motivos de sinal de localização
nuclear. É o mesmo cálculo usado no treino, para o predict.py gerar colunas
idênticas às que o modelo viu.
"""

import numpy as np
import pandas as pd

# the 20 standard amino acids / os 20 aminoácidos padrão
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy scale / escala de hidropatia de Kyte-Doolittle
kyte_doolittle = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def clean_sequences(df, sequence_col="Sequence"):
    """Keep only rows whose sequence has the 20 standard amino acids.

    Mantém só as linhas cuja sequência tem apenas os 20 aminoácidos padrão.
    """
    df = df.dropna(subset=[sequence_col])
    valid = df[sequence_col].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    return df[valid].reset_index(drop=True)


def build_features(df, sequence_col="Sequence"):
    """Build the 28 sequence-derived features / monta as 28 features derivadas da sequência."""
    sequence = df[sequence_col]
    seq_len = sequence.str.len()
    features = pd.DataFrame(index=df.index)

    # amino acid composition (20 columns) / composição de aminoácidos (20 colunas)
    for aa in amino_acids:
        features[f"aa_{aa}"] = sequence.str.count(aa) / seq_len

    # physicochemical features from composition / features físico-químicas a partir da composição
    features["log_length"] = np.log(seq_len)
    features["gravy"] = sum(features[f"aa_{aa}"] * value for aa, value in kyte_doolittle.items())
    features["frac_positive"] = features[["aa_K", "aa_R"]].sum(axis=1)
    features["frac_negative"] = features[["aa_D", "aa_E"]].sum(axis=1)
    features["net_charge_frac"] = features["frac_positive"] - features["frac_negative"]
    features["frac_aromatic"] = features[["aa_F", "aa_W", "aa_Y"]].sum(axis=1)

    # nuclear localization signals are rich in lysine and arginine / sinais nucleares são ricos em K e R
    features["kr_motif_count"] = sequence.str.count(r"[KR]{4,}")
    features["kr_motif_pat7"] = sequence.str.count(r"P[KR]{2,}[A-Z][KR]")

    return features
