"""Split settings and function shared by the models.

Configuração e função de divisão compartilhadas pelos modelos: split por
organismo (nenhum organismo aparece em dois conjuntos), 60% treino, 20%
validação e 20% teste.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# random seed for reproducibility / semente aleatória para reprodutibilidade
seed = 42

# fraction of organisms in test, and of the remaining in validation
# fração dos organismos no teste, e do restante na validação
test_size = 0.2
val_size = 0.2


def group_split(y, groups, seed=seed, test_size=test_size, val_size=val_size):
    """Return train, validation and test row indexes, split by group.

    Retorna os índices de treino, validação e teste, divididos por grupo.
    O teste sai primeiro e a validação vem do treino restante.
    """
    n_rows = len(y)
    rows = np.arange(n_rows)
    placeholder = np.zeros((n_rows, 1))
    groups = np.asarray(groups)

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer.split(placeholder, y, groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_rel, val_rel = next(
        inner.split(placeholder[train_val_idx], np.asarray(y)[train_val_idx], groups[train_val_idx])
    )
    return rows[train_val_idx][train_rel], rows[train_val_idx][val_rel], rows[test_idx]
