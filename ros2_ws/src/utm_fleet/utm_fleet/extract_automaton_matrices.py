import numpy as np
from ultrades.automata import *
import scipy.sparse as sp 


def extract_automaton_matrices(G, k):
    """
    Versão OTIMIZADA usando SciPy Sparse (COO -> CSR).
    """
    try:
        k = int(k)
    except Exception as err:
        raise TypeError(f"'k' must be an integer (got {k!r})") from err

    # 1. Materializa e Indexa
    Q = list(states(G))
    E = list(events(G))
    T = list(transitions(G))

    n = len(Q)
    m = len(E)

    state_index = {q: i for i, q in enumerate(Q)}
    event_index = {e: i for i, e in enumerate(E)}

    # 2. Coleta de Índices para Matrizes Esparsas
    # COO (Coordinate format) é o mais rápido para construção
    A_rows, A_cols = [], []
    B_rows, B_cols = [], []
    C_rows, C_cols = [], []

    for (q_i, sigma, q_j) in T:
        i = state_index[q_i]
        j = state_index[q_j]
        eidx = event_index[sigma]

        # Matriz A (n x n): De i para j
        A_rows.append(i)
        A_cols.append(j)
        
        # Matriz B (m x n): Evento eidx alcança estado destino j
        B_rows.append(eidx)
        B_cols.append(j)
        
        # Matriz C (n x m): Estado origem i disponibiliza evento eidx
        C_rows.append(i)
        C_cols.append(eidx)

    # 3. Construção e Conversão para CSR
    # dtype=np.float32 ou np.int8 são mais rápidos que booleanos em sparse
    
    # Valores de 1 para as coordenadas
    data = np.ones(len(T), dtype=np.int8) 

    # Matriz A: (n x n)
    A_coo = sp.coo_matrix((data, (A_rows, A_cols)), shape=(n, n), dtype=np.int8)
    A_csr = A_coo.tocsr() # CSR é ótimo para slicing por linha (como em compute_reach)
    
    # Matriz B: (m x n)
    B_coo = sp.coo_matrix((data, (B_rows, B_cols)), shape=(m, n), dtype=np.int8)
    B_csr = B_coo.tocsr() 

    # Matriz C: (n x m)
    C_coo = sp.coo_matrix((data, (C_rows, C_cols)), shape=(n, m), dtype=np.int8)
    C_csr = C_coo.tocsr() 

    # 4. Custos e Separações (Mantidos em NumPy denso, já que W e D são densos)
    W = np.zeros((n, k), dtype=np.float32)
    D = np.zeros((m, m), dtype=np.float32)

    # 5. One-hots por evento (Mantido)
    event_dict = {}
    for e in E:
        onehot = np.zeros(m, dtype=np.bool_)
        onehot[event_index[e]] = True
        event_dict[str(e)] = (e, onehot)

    # Retorna matrizes esparsas!
    return A_csr, B_csr, C_csr, W, D, event_dict, state_index


def print_automaton_data(A, B, C, W, D, event_dict):
    np.set_printoptions(
        threshold=np.inf,
        linewidth=200,
        formatter={'bool': lambda x: '1' if x else '0'}
    )

    print("───── Matriz A (adjacency: n x n) ─────")
    print(A.astype(int)); print()

    print("───── Matriz B (event reachability: m x n) ─────")
    print(B.astype(int)); print()

    print("───── Matriz C (event availability: n x m) ─────")
    print(C.astype(int)); print()

    print("───── Matriz W (state costs: n x k) ─────")
    print(W); print()

    print("───── Matriz D (event separation: m x m) ─────")
    print(D); print()

    print("───── Dicionário de Eventos ─────")
    for e_str, (e_obj, onehot) in event_dict.items():
        print(f"{e_str} →")
        print(f"  Objeto: {e_obj}")
        print(f"  One-hot: {onehot.astype(int)}")
        print()


def verifica_propriedade(A, B, C):
    """
    Verifica a propriedade de determinismo algébrico:
    Para cada (estado i, evento e) disponível (C[i,e]=1), deve existir
    exatamente 1 próximo estado alcançável por e.
    """
    n, m = C.shape  # C é n x m
    # Matmul em inteiros para contar caminhos
    D = (A.astype(np.int64) @ B.T.astype(np.int64))  # n x m, D[i,e] = nº de destinos via e
    V = (C.astype(np.int64) * D)  # hadamard -> mantém contagens só onde C=1

    problemas = []
    for i in range(n):
        for e in range(m):
            if C[i, e]:
                valor_V = V[i, e]
                if valor_V != 1:
                    problemas.append({
                        'estado': i,
                        'evento': e,
                        'valor_C': int(C[i, e]),
                        'valor_V': int(valor_V),
                        'tipo': 'MÚLTIPLOS ESTADOS' if valor_V > 1 else 'NENHUM ESTADO'
                    })

    ok = np.array_equal(V, C.astype(np.int64))
    return ok, problemas


