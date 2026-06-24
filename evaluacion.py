
import numpy as np

# Extremos absolutos leídos del CSV de tasa_eclosion:
#   mínimo: laúd peores condiciones = 0.40
#   máximo: golfina condiciones perfectas = 0.90
_TASA_MIN_ABS = 0.40
_TASA_MAX_ABS = 0.90
_RANGO_ABS    = _TASA_MAX_ABS - _TASA_MIN_ABS   # 0.50


def _norm_v1(v):
    """Normaliza V1 al rango [0,1] con los extremos absolutos del CSV."""
    return float(np.clip((v - _TASA_MIN_ABS) / _RANGO_ABS, 0.0, 1.0))


def _previos_por_especie(nidos_previos):
    """Dict especie → (xs, ys, profs) para nidos de jornadas previas."""
    acum = {}
    if nidos_previos:
        for n in nidos_previos:
            e = n['especie']
            acum.setdefault(e, ([], [], []))
            acum[e][0].append(float(n['x']))
            acum[e][1].append(float(n['y']))
            acum[e][2].append(float(n.get('prof', 0)))
    return acum


# ── V1 ─────────────────────────────────────────────────────────────────────────

def calcular_v1(individuo, base, nidos_previos=None):
    
    genes_nuevos = individuo.genes
    por_esp_nuevo = {}
    for i, g in enumerate(genes_nuevos):
        por_esp_nuevo.setdefault(g.especie, []).append(i)

    por_esp_prev = _previos_por_especie(nidos_previos)

    total_tasa = 0.0
    N_total    = 0

    especies = set(por_esp_nuevo.keys()) | set(por_esp_prev.keys())

    for esp in especies:
        e = base[esp]

        # Nidos nuevos de esta especie
        idxs_n = por_esp_nuevo.get(esp, [])
        gs_new = [genes_nuevos[i] for i in idxs_n]
        xs_n   = np.array([g.x    for g in gs_new]) if gs_new else np.array([])
        ys_n   = np.array([g.y    for g in gs_new]) if gs_new else np.array([])
        prfs_n = np.array([g.prof for g in gs_new]) if gs_new else np.array([])

        # Nidos previos de esta especie
        p_data = por_esp_prev.get(esp)
        if p_data and len(p_data[0]) > 0:
            xs_p   = np.array(p_data[0])
            ys_p   = np.array(p_data[1])
            prfs_p = np.array([
                float(pf) if pf != 0 else e['prof_opt']
                for pf in p_data[2]
            ])
        else:
            xs_p = ys_p = prfs_p = np.array([])

        # Corral completo de esta especie
        xs_all   = np.concatenate([xs_n,   xs_p])
        ys_all   = np.concatenate([ys_n,   ys_p])
        prfs_all = np.concatenate([prfs_n, prfs_p])
        n_all    = len(xs_all)

        if n_all == 0:
            continue

        N_total += n_all

        # factor_prof: gaussiana centrada en prof_opt, sigma del CSV
        fp_all = np.exp(
            -((prfs_all - e['prof_opt'])**2) / (2.0 * e['sigma']**2)
        )

        # factor_sep: proporción de vecinos a dist >= sep_min del CSV
        if n_all > 1:
            dx   = xs_all[:, None] - xs_all[None, :]
            dy   = ys_all[:, None] - ys_all[None, :]
            dist_sq = dx**2 + dy**2
            np.fill_diagonal(dist_sq, np.inf)
            ok     = (dist_sq >= e['sep_min']**2).sum(axis=1)
            fs_all = ok / (n_all - 1)
        else:
            fs_all = np.ones(n_all)

        # Interpolación entre promedio histórico y máximo histórico
        tasa = (e['tasa_promedio'] +
                (e['tasa_maxima'] - e['tasa_promedio']) * fp_all * fs_all)
        total_tasa += float(tasa.sum())

    if N_total == 0:
        return _TASA_MIN_ABS

    v1 = total_tasa / N_total
    return round(float(np.clip(v1, _TASA_MIN_ABS, _TASA_MAX_ABS)), 6)


# ── V2 ─────────────────────────────────────────────────────────────────────────

def calcular_v2(individuo, gestor):
   
    N = individuo.num_nidos()
    if N == 0:
        return 0.0
    fuera = sum(1 for g in individuo.genes if not g.en_zona_correcta(gestor))
    return round(fuera / N, 6)


# ── V3 ─────────────────────────────────────────────────────────────────────────

def calcular_v3(individuo, base, nidos_previos=None):
   
    genes = individuo.genes
    N     = len(genes)
    if N == 0:
        return 0.0

    por_esp_prev = _previos_por_especie(nidos_previos)
    por_esp_new  = {}
    for g in genes:
        por_esp_new.setdefault(g.especie, []).append(g)

    pares_total = 0
    pares_malos = 0

    for esp, gs in por_esp_new.items():
        n_new = len(gs)
        sep   = base[esp]['sep_min']
        xs_n  = np.array([g.x for g in gs])
        ys_n  = np.array([g.y for g in gs])

        # (a) nuevo-vs-nuevo (triángulo superior, sin diagonal)
        if n_new > 1:
            dx   = xs_n[:, None] - xs_n[None, :]
            dy   = ys_n[:, None] - ys_n[None, :]
            dist_sq = dx**2 + dy**2
            mask = np.triu(np.ones((n_new, n_new), bool), 1)
            pares_total += int(mask.sum())
            pares_malos += int((dist_sq[mask] < sep**2).sum())

        # (b) nuevo-vs-previo
        p_data = por_esp_prev.get(esp)
        if p_data and len(p_data[0]) > 0:
            xs_p = np.array(p_data[0])
            ys_p = np.array(p_data[1])
            n_p  = len(xs_p)
            dx   = xs_n[:, None] - xs_p[None, :]
            dy   = ys_n[:, None] - ys_p[None, :]
            dist_sq = dx**2 + dy**2
            pares_total += n_new * n_p
            pares_malos += int((dist_sq < sep**2).sum())

    return round(pares_malos / pares_total, 6) if pares_total > 0 else 0.0


# ── V4 ─────────────────────────────────────────────────────────────────────────

def calcular_v4(individuo, base):
   
    genes = individuo.genes
    N     = len(genes)
    if N == 0:
        return 0.0
    total = 0.0
    for g in genes:
        e     = base[g.especie]
        rango = e['prof_max'] - e['prof_min']
        desv  = abs(g.prof - e['prof_opt']) / rango if rango > 0 else 0.0
        total += min(desv, 1.0)
    return round(total / N, 6)


# ── FITNESS ─────────────────────────────────────────────────────────────────────

def calcular_fitness(individuo, gestor, base, corral, nidos_previos=None):
   
    if individuo.num_nidos() == 0:
        individuo.fitness = 0.0
        individuo.v1 = individuo.v2 = individuo.v3 = individuo.v4 = 0.0
        return 0.0

    v1_raw = calcular_v1(individuo, base, nidos_previos)
    v2     = calcular_v2(individuo, gestor)
    v3     = calcular_v3(individuo, base, nidos_previos)
    v4     = calcular_v4(individuo, base)

    v1_norm = _norm_v1(v1_raw)
    fitness = v1_norm - (v2 + v3 + v4) / 3.0

    individuo.fitness = round(float(fitness), 6)
    individuo.v1      = v1_raw
    individuo.v2      = v2
    individuo.v3      = v3
    individuo.v4      = v4
    return individuo.fitness


def evaluar_poblacion(poblacion, gestor, base, corral, nidos_previos=None):
    for ind in poblacion:
        calcular_fitness(ind, gestor, base, corral, nidos_previos)
    return poblacion
