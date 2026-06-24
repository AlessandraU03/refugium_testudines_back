import random
import math
import numpy as np
from cromosoma import Individuo


def seleccion_torneo(poblacion, k=3):
    n         = len(poblacion)
    n_parejas = max(n // 2, 1)
    parejas   = []
    for _ in range(n_parejas):
        c1 = random.choices(poblacion, k=k)
        c2 = random.choices(poblacion, k=k)
        parejas.append((
            max(c1, key=lambda i: i.fitness),
            max(c2, key=lambda i: i.fitness),
        ))
    return parejas


def cruza_por_especie(p1, p2, prob_cruza):

    if random.random() > prob_cruza:
        return p1.copia(), p2.copia()

    especies = sorted({g.especie for g in p1.genes})
    h1_genes = []
    h2_genes = []

    for esp in especies:
        g1 = [g for g in p1.genes if g.especie == esp]
        g2 = [g for g in p2.genes if g.especie == esp]
        for a, b in zip(g1, g2):
            if random.random() < 0.5:
                h1_genes.append(a.copia())
                h2_genes.append(b.copia())
            else:
                h1_genes.append(b.copia())
                h2_genes.append(a.copia())

    return Individuo(h1_genes), Individuo(h2_genes)


def mutacion_combinada(individuo, prob_mut, base, gestor, nidos_previos=None):
   
    ind   = individuo.copia()
    genes = ind.genes

    # Prebuild numpy arrays for the genes of this individual to avoid python loops
    all_xs = np.array([g.x for g in genes])
    all_ys = np.array([g.y for g in genes])
    all_esps = np.array([g.especie for g in genes])

    # Preconstruir coordenadas de previos por especie
    previos_esp = {}
    if nidos_previos:
        if isinstance(nidos_previos, dict) and "__cache__" in nidos_previos:
            previos_esp = {
                e: (p_data[0], p_data[1])
                for e, p_data in nidos_previos.items()
                if e != "__cache__"
            }
        else:
            for n in nidos_previos:
                e = n['especie']
                previos_esp.setdefault(e, ([], []))
                previos_esp[e][0].append(float(n['x']))
                previos_esp[e][1].append(float(n['y']))
            previos_esp = {
                e: (np.array(xs), np.array(ys))
                for e, (xs, ys) in previos_esp.items()
            }

    # ── Mutación 1: Profundidad ───────────────────────────────────────────────
    for gen in genes:
        if random.random() < prob_mut:
            e = base[gen.especie]
            # Perturbación: 0.5 * sigma del CSV → exploración moderada
            gen.prof = round(float(np.clip(
                gen.prof + random.gauss(0, e['sigma'] * 0.5),
                e['prof_min'], e['prof_max']
            )), 1)

    # ── Mutación 2: Posición ─────────────────────────────────────────────────
    for idx, gen in enumerate(genes):
        if random.random() >= prob_mut:
            continue

        e       = base[gen.especie]
        sep_min = e['sep_min']
        lim     = gestor.limites_zona(gen.zona_correcta())
        if not lim:
            continue

        # Vecinos nuevos de la misma especie (excluyendo este nido) usando indexación NumPy
        mask = (all_esps == gen.especie)
        mask[idx] = False
        xs_new = all_xs[mask]
        ys_new = all_ys[mask]

        # Vecinos previos de la misma especie
        xs_prev, ys_prev = previos_esp.get(gen.especie, (np.array([]), np.array([])))

        xs_all = np.concatenate([xs_new, xs_prev])
        ys_all = np.concatenate([ys_new, ys_prev])

        if len(xs_all) == 0:
            continue

        sep_min_sq = (sep_min - 0.05) ** 2
        dist_sq_actual = (gen.x - xs_all)**2 + (gen.y - ys_all)**2
        viols_actual = int((dist_sq_actual < sep_min_sq).sum())

        if viols_actual == 0:
            continue   # ya respeta sep_min, no muta posición

        cxs = np.round(np.random.uniform(lim['xmin'], lim['xmax'], size=60), 1)
        cys = np.round(np.random.uniform(lim['ymin'], lim['ymax'], size=60), 1)

        dx = cxs[:, None] - xs_all[None, :]
        dy = cys[:, None] - ys_all[None, :]
        dist_sq_cand = dx**2 + dy**2

        viols_cand = (dist_sq_cand < sep_min_sq).sum(axis=1)
        best_idx = np.argmin(viols_cand)
        mejor_viols = viols_cand[best_idx]

        if mejor_viols < viols_actual:
            new_x = float(cxs[best_idx])
            new_y = float(cys[best_idx])
            gen.x = new_x
            gen.y = new_y
            all_xs[idx] = new_x
            all_ys[idx] = new_y

    return ind


def poda_elitismo(poblacion, descendencia, tam_pob, n_elite=2):
    """Selección μ+λ: conserva los n_elite mejores y completa aleatoriamente."""
    todos = sorted(poblacion + descendencia,
                   key=lambda i: i.fitness, reverse=True)
    elite = todos[:n_elite]
    resto = todos[n_elite:]
    random.shuffle(resto)
    return elite + resto[:tam_pob - n_elite]
