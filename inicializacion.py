import random
import numpy as np
from cromosoma import Gen, Individuo


def inicializar_poblacion(tam_pob, nidos_entrada, gestor, base,
                          nidos_ocupados=None):
    n_semi      = tam_pob // 2
    n_aleatorio = tam_pob - n_semi
    poblacion   = []

    for k in range(n_semi):
        nivel = 0.10 + 0.90 * (k / max(n_semi - 1, 1))
        poblacion.append(_crear_semi(nidos_entrada, gestor, base, nivel, nidos_ocupados))

    for _ in range(n_aleatorio):
        poblacion.append(_crear_aleatorio(nidos_entrada, gestor, base, nidos_ocupados))

    return poblacion


def _crear_semi(nidos_entrada, gestor, base, nivel, nidos_ocupados):
   
    genes   = []
    por_esp = {}
    for id_nido, especie in nidos_entrada:
        por_esp.setdefault(especie, []).append(id_nido)

    for especie in sorted(por_esp):
        ids  = por_esp[especie]
        e    = base[especie]
        zona = f"zona_{especie}"
        lim  = gestor.limites_zona(zona)
        if not lim:
            continue

        pos = gestor.posiciones_cuadricula(zona, len(ids), e['sep_min'], nidos_ocupados)
        random.shuffle(pos)

        for id_nido, (bx, by) in zip(ids, pos):
            max_ruido_xy = e['sep_min'] / 10.0
            sigma_xy     = max_ruido_xy * nivel
            x = float(np.clip(
                bx + random.gauss(0, sigma_xy),
                lim['xmin'], lim['xmax']
            ))
            y = float(np.clip(
                by + random.gauss(0, sigma_xy),
                lim['ymin'], lim['ymax']
            ))

            p = float(np.clip(
                e['prof_opt'] + random.gauss(0, e['sigma'] * nivel),
                e['prof_min'], e['prof_max']
            ))

            genes.append(Gen(id_nido, especie, round(x, 1), round(y, 1), round(p, 1)))

    return Individuo(genes)


def _crear_aleatorio(nidos_entrada, gestor, base, nidos_ocupados):
    
    genes   = []
    por_esp = {}
    for id_nido, especie in nidos_entrada:
        por_esp.setdefault(especie, []).append(id_nido)

    for especie in sorted(por_esp):
        ids = por_esp[especie]
        e   = base[especie]
        lim = gestor.limites_zona(f"zona_{especie}")
        if not lim:
            continue
        pos = gestor.posiciones_cuadricula(
            f"zona_{especie}", len(ids), e['sep_min'], nidos_ocupados
        )

        for id_nido, (x, y) in zip(ids, pos):
            p = round(random.uniform(e['prof_min'], e['prof_max']), 1)
            genes.append(Gen(id_nido, especie, x, y, p))

    return Individuo(genes)
