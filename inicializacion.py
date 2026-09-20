import random
import numpy as np
from cromosoma import Gen, Individuo, ORDENES_ZONAS, resolver_gestor


def inicializar_poblacion(tam_pob, nidos_entrada, gestor, base,
                          nidos_ocupados=None):
    n_semi      = tam_pob // 2
    n_aleatorio = tam_pob - n_semi
    poblacion   = []

    # El reparto de franjas se siembra repartido entre los seis ordenes
    # posibles, no todos en el historico. Si la poblacion arrancara entera con
    # golfina-prieta-laud, el AG tendria que descubrir por mutacion que existen
    # los otros cinco, y con la malla actual ese es el peor de los seis para
    # prieta y laud.
    n_ordenes = len(ORDENES_ZONAS)

    for k in range(n_semi):
        nivel = 0.10 + 0.90 * (k / max(n_semi - 1, 1))
        idx = k % n_ordenes
        ind = _crear_semi(nidos_entrada, resolver_gestor(gestor, idx), base,
                          nivel, nidos_ocupados)
        ind.idx_orden = idx
        poblacion.append(ind)

    for j in range(n_aleatorio):
        idx = j % n_ordenes
        ind = _crear_aleatorio(nidos_entrada, resolver_gestor(gestor, idx),
                               base, nidos_ocupados)
        ind.idx_orden = idx
        poblacion.append(ind)

    return poblacion


def individuo_secuencial(nidos_entrada, gestor, base, nidos_ocupados=None):
    """La colocación que se hace hoy en el corral, sin ningún algoritmo.

    Llena la rejilla de cada zona en el orden de la serpentina, a la
    separación que fija la NOM-162, y siembra todo a la profundidad óptima de
    la especie. No es una colocación mala: respeta la norma en los dos
    parámetros que la norma regula.

    Existe para servir de línea base. El AG tiene que superarla en crías
    esperadas, y si no lo hace hay que decirlo: comparar el algoritmo contra
    una colocación al azar sería hacer trampa, porque nadie siembra al azar.
    """
    genes = []
    por_esp = {}
    for id_nido, especie in nidos_entrada:
        por_esp.setdefault(especie, []).append(id_nido)

    previos = list(nidos_ocupados or [])

    # La linea base usa SIEMPRE el reparto historico de franjas
    # (golfina, prieta, laud), porque es el que tenia el corral antes de que el
    # orden fuera una decision del algoritmo. Compararla contra un reparto
    # optimizado seria regalarle al AG parte de su propia ventaja.
    gestor = resolver_gestor(gestor, 0)

    for especie in sorted(por_esp):
        ids = por_esp[especie]
        e   = base[especie]
        # La misma rejilla que usa el término de orden para juzgar. Aprieta el
        # paso hasta que alcancen las casillas, que es lo que hace el personal
        # cuando llegan más nidos de los que caben holgados.
        #
        # Importa para que la comparación sea honesta: una versión anterior
        # apilaba los nidos sobrantes en la última casilla y producía 14
        # nidos/m². El AG le ganaba a un espantapájaros, no al procedimiento
        # real.
        slots = gestor.slots_suficientes(f"zona_{especie}", e['sep_min'],
                                         previos, len(ids))

        if not slots:
            continue

        for k, id_nido in enumerate(ids):
            # Tras el bucle esto no debería quedarse corto. Si la zona está
            # físicamente llena aun con la rejilla apretada al mínimo, se
            # reutiliza la última casilla: es un corral sin lugar, y el
            # resultado tiene que verse mal porque lo está.
            x, y = slots[k] if k < len(slots) else slots[-1]
            genes.append(Gen(id_nido, especie, x, y, e['prof_opt']))

    return Individuo(genes)


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

        # Sembrar sobre la MISMA rejilla que usan la reparacion y el juez del
        # orden. `posiciones_cuadricula` construye la suya a sep_min y, cuando
        # no alcanza, completa con muestreo aleatorio: eso arrancaba la
        # poblacion con nidos apinados a 4 nidos/m2 en un corral saturado.
        pos = list(gestor.slots_suficientes(zona, e['sep_min'],
                                            nidos_ocupados, len(ids)))
        if not pos:
            continue
        while len(pos) < len(ids):
            pos.append(pos[-1])
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
        pos = list(gestor.slots_suficientes(f"zona_{especie}", e['sep_min'],
                                            nidos_ocupados, len(ids)))
        if not pos:
            continue
        random.shuffle(pos)
        while len(pos) < len(ids):
            pos.append(pos[-1])

        for id_nido, (x, y) in zip(ids, pos):
            p = round(random.uniform(e['prof_min'], e['prof_max']), 1)
            genes.append(Gen(id_nido, especie, x, y, p))

    return Individuo(genes)
