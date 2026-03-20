# utm_fleet/milp_optimizer.py

from collections import defaultdict, deque
import threading
import warnings
import time

import numpy as np
import scipy.sparse as sp
from gurobipy import *
from ultrades.automata import dfa, event, transitions, states

from .extract_automaton_matrices import extract_automaton_matrices


GLOBAL_MILP_LOCK = threading.Lock()

GLOBAL_LAST_U_SEQUENCE = None
GLOBAL_LAST_EVENT_NAMES = None

try:
    GLOBAL_GUROBI_ENV = Env(empty=True)
    GLOBAL_GUROBI_ENV.setParam("OutputFlag", 0)
    GLOBAL_GUROBI_ENV.setParam("TimeLimit", 5.0)
    GLOBAL_GUROBI_ENV.setParam("MIPGap", 0.001)
    GLOBAL_GUROBI_ENV.start()
except Exception as e:
    print("ERRO CRÍTICO ao inicializar GLOBAL_GUROBI_ENV:", e)
    GLOBAL_GUROBI_ENV = None


BIGM = 100.0
EPSILON_W = 0.7
BETA_INCENTIVE = 10.0
ALPHA_TIME = 0.4
ALPHA_STATE = 0.3


def new_sub_automato_propriedade(G, e, Nc):
    transicoes = transitions(G)

    por_origem = defaultdict(list)
    for orig, ev, dest in transicoes:
        por_origem[orig].append((orig, ev, dest))

    estado_inicial = e
    fila = deque([(estado_inicial, 0)])
    profundidade = {estado_inicial: 0}
    recorte_trans = []

    while fila:
        estado, d = fila.popleft()
        if d == Nc:
            continue

        for trans in por_origem.get(estado, []):
            orig, ev, dest = trans
            recorte_trans.append(trans)
            if dest not in profundidade:
                profundidade[dest] = d + 1
                fila.append((dest, d + 1))

    def aplica_correcao(trans_list, depth, max_iter=20):
        for _ in range(max_iter):
            adj = defaultdict(set)
            dest_por_evento = defaultdict(set)
            delta = {}
            por_evento_dest = defaultdict(list)

            for orig, ev, dest in trans_list:
                adj[orig].add(dest)
                dest_por_evento[ev].add(dest)
                por_evento_dest[(ev, dest)].append(orig)
                if (orig, ev) not in delta:
                    delta[(orig, ev)] = dest

            to_remove = set()

            for key, dest_can in delta.items():
                orig, ev = key
                Adj_i = adj.get(orig, set())
                Dest_ev = dest_por_evento.get(ev, set())
                S = Adj_i & Dest_ev

                if len(S) <= 1:
                    continue

                extras = S - {dest_can}
                for dest_extra in extras:
                    for o2 in por_evento_dest.get((ev, dest_extra), []):
                        if depth.get(o2, 0) > depth.get(orig, 0):
                            to_remove.add((o2, ev, dest_extra))

            if not to_remove:
                break

            trans_list = [t for t in trans_list if t not in to_remove]

        return trans_list

    trans_corrigidas = aplica_correcao(recorte_trans, profundidade)

    epsolon = event("epslon", controllable=False)

    origs = set([t[0] for t in trans_corrigidas])
    estados_alcancaveis = set(profundidade.keys())
    estados_mortos = estados_alcancaveis - origs

    for st in estados_mortos:
        trans_corrigidas.append((st, epsolon, st))

    new_automaton = dfa(
        trans_corrigidas,
        estado_inicial,
        "Sub_%s_%s" % (str(estado_inicial), str(G))
    )

    return new_automaton


def compute_reach(A_csr, H, start=0, inviaveis=None):
    n_ = A_csr.shape[0]
    banned = np.zeros(n_, dtype=bool)

    if inviaveis is not None and len(inviaveis):
        banned[inviaveis] = True

    reach_ = []
    cur = np.array([start], dtype=np.int32)
    if banned[start]:
        cur = np.array([], dtype=np.int32)

    reach_.append(cur)

    for _ in range(H):
        if cur.size == 0:
            nxt = np.array([], dtype=np.int32)
        else:
            A_sub = A_csr[cur, :]
            nxt = np.unique(A_sub.indices)
            if nxt.size:
                nxt = nxt[~banned[nxt]].astype(np.int32)

        reach_.append(nxt)
        cur = nxt

    return reach_


def otimizador(Sup, estado_inicial_recorte, janela, cost_dictionary, list_eventos_interesse, list_eventos_proibidos):
    global GLOBAL_LAST_U_SEQUENCE, GLOBAL_LAST_EVENT_NAMES

    if GLOBAL_GUROBI_ENV is None:
        print("[ERRO] GLOBAL_GUROBI_ENV é None.")
        return [], -1

    H = int(janela)

    recorte = new_sub_automato_propriedade(Sup, estado_inicial_recorte, H)
    resultado_matrices = extract_automaton_matrices(recorte, 3)

    A_csr, B_csr, C_csr, W, D_np, event_dict, state_index = resultado_matrices

    n = A_csr.shape[0]
    m = C_csr.shape[1]
    event_names = list(event_dict.keys())

    Q_recorte = list(states(recorte))
    for estado in Q_recorte:
        estado_str = str(estado)
        if estado_str in cost_dictionary:
            custo_E, custo_Tf, custo_D = cost_dictionary[estado_str]
        else:
            custo_E, custo_Tf, custo_D = (0.0, 0.0, 0.0)

        i = state_index[estado]
        W[i, 0] = custo_E
        W[i, 1] = custo_Tf
        W[i, 2] = custo_D

    pesos_E_D_somados = ALPHA_TIME + ALPHA_STATE
    pesos_E_D = np.array([
        ALPHA_TIME / pesos_E_D_somados,
        ALPHA_STATE / pesos_E_D_somados,
    ])
    W_ED = W[:, [0, 2]]
    w_bar = (W_ED @ pesos_E_D).astype(np.float32)

    name_to_idx = {}
    for idx, nm in enumerate(event_names):
        name_to_idx[nm] = idx

    I_indices = np.array([name_to_idx[nm] for nm in list_eventos_interesse if nm in name_to_idx], dtype=np.int32)
    P_indices = np.array([name_to_idx[nm] for nm in list_eventos_proibidos if nm in name_to_idx], dtype=np.int32)

    m_I = len(I_indices)
    m_P = len(P_indices)

    inviaveis_cols = np.where((C_csr.indptr[1:] - C_csr.indptr[:-1]) == 0)[0].astype(np.int32)
    reach = compute_reach(A_csr, H, start=0, inviaveis=inviaveis_cols)
    pos = [{int(j): k for k, j in enumerate(reach[t])} for t in range(H + 1)]

    event_seq = []
    model_status = GRB.LOADED

    with GLOBAL_MILP_LOCK:
        model = None
        try:
            model = Model("mpsc_eoi_sem_tempo", env=GLOBAL_GUROBI_ENV)
            model.setParam("OutputFlag", 0)

            x = [model.addMVar(len(reach[t]), vtype=GRB.BINARY, name="x_%d" % t) for t in range(H + 1)]
            u = model.addMVar((H, m), vtype=GRB.BINARY, name="u")

            if m_I > 0:
                tau = model.addMVar((H, m_I), vtype=GRB.BINARY, name="tau")
            else:
                tau = None

            if 0 in pos[0]:
                model.addConstr(x[0][pos[0][0]] == 1.0, name="init_x")

            for t in range(H):
                model.addConstr(x[t].sum() == 1.0, name="state_onehot_t%d" % t)
                model.addConstr(u[t, :].sum() == 1.0, name="event_onehot_t%d" % t)

            model.addConstr(x[H].sum() == 1.0, name="state_onehot_t%d" % H)

            C_dense = np.asarray(C_csr.todense())

            for t in range(H):
                rt = reach[t]
                if len(rt) == 0:
                    continue

                C_sub_rt = C_dense[rt, :]
                model.addConstr(u[t, :] <= x[t] @ C_sub_rt, name="event_feas_t%d" % t)

            A_csr_t = A_csr.transpose().tocsr()

            for t in range(H):
                rt = reach[t]
                rtp1 = reach[t + 1]
                if len(rtp1) == 0:
                    continue

                sources_dict = defaultdict(list)

                for idx_next, state_next in enumerate(rtp1):
                    prev_states_A = A_csr_t[state_next, :].indices

                    for idx_curr, state_curr in enumerate(rt):
                        if state_curr in prev_states_A:
                            valid_events = np.nonzero(
                                np.multiply(
                                    B_csr[:, state_next].todense().A1,
                                    C_csr[state_curr, :].todense().A1
                                )
                            )[0]

                            for event_idx in valid_events:
                                sources_dict[idx_next].append((idx_curr, event_idx))

                for idx_next in range(len(rtp1)):
                    sources = sources_dict[idx_next]
                    if sources:
                        rhs = quicksum([x[t][i] * u[t][j] for i, j in sources])
                        model.addConstr(x[t + 1][idx_next] == rhs, name="dyn_t%d_s%d" % (t, idx_next))
                    else:
                        model.addConstr(x[t + 1][idx_next] == 0.0, name="dyn_unreach_t%d_s%d" % (t, idx_next))

            if m_P > 0:
                for p_idx in P_indices:
                    model.addConstr(u[:, p_idx].sum() == 0.0, name="event_prohibited_e%d" % p_idx)

            if m_I > 0 and tau is not None:
                for i_idx in range(m_I):
                    e_idx = I_indices[i_idx]
                    u_acum = 0.0

                    for t in range(H):
                        u_acum += u[t, e_idx]
                        model.addConstr(tau[t, i_idx] <= u[t, e_idx])
                        model.addConstr(tau[t, i_idx] <= 1 - u_acum + u[t, e_idx])
                        model.addConstr(tau[t, i_idx] >= u[t, e_idx] - (u_acum - u[t, e_idx]))

                    model.addConstr(tau[:, i_idx].sum() <= 1.0)

            if GLOBAL_LAST_U_SEQUENCE is not None and GLOBAL_LAST_EVENT_NAMES is not None:
                u_prev = GLOBAL_LAST_U_SEQUENCE
                event_names_prev = GLOBAL_LAST_EVENT_NAMES
                H_prev = u_prev.shape[0]

                prev_to_curr_map = {}
                for name in event_names_prev:
                    prev_to_curr_map[name] = name_to_idx.get(name)

                for t in range(H):
                    if t + 1 < H_prev:
                        event_idx_prev = int(np.argmax(u_prev[t + 1, :]))
                        event_name_prev = event_names_prev[event_idx_prev]
                        curr_idx = prev_to_curr_map.get(event_name_prev)

                        if curr_idx is not None and curr_idx < m:
                            u[t, curr_idx].Start = 1.0

            cost_states_E_D = quicksum(
                x[t][idx] * w_bar[state_global]
                for t in range(H)
                for idx, state_global in enumerate(reach[t])
            )

            if m_I > 0 and tau is not None:
                cost_incentive = quicksum(
                    -1 * (H - t) * tau[t, i_idx] * BETA_INCENTIVE
                    for i_idx in range(m_I)
                    for t in range(H)
                )
            else:
                cost_incentive = 0.0

            final_objective = ALPHA_STATE * cost_states_E_D + cost_incentive
            model.setObjective(final_objective, GRB.MINIMIZE)

            model.optimize()
            model_status = model.status

            if model_status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
                u_sol = u.X
                GLOBAL_LAST_U_SEQUENCE = u_sol
                GLOBAL_LAST_EVENT_NAMES = event_names

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    seq_idx = [int(np.argmax(u_sol[t, :])) for t in range(H)]

                event_seq = [event_names[i] for i in seq_idx]
                print("[✓] Solução encontrada:", event_seq)
            else:
                print("[×] Otimização falhou. Status:", model_status)

        except Exception as e:
            print("[ERRO NO GUROBI]", e)
            model_status = -1

        finally:
            if model is not None:
                try:
                    model.dispose()
                except Exception:
                    pass

    return event_seq, model_status