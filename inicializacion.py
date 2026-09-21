import random
import numpy as np
from cromosoma import (Gen, Individuo, ORDENES_ZONAS, resolver_gestor,
                       posicion_mas_libre)


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
    # Si el corral ya tiene nidos enterrados, el gestor trae un solo reparto y
    # toda la poblacion nace con el: el reparto dejo de ser una decision.
    n_ordenes = len(gestor) if hasattr(gestor, '__len__') else len(ORDENES_ZONAS)

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

        lim_z    = gestor.limites_zona(f"zona_{especie}")
        colocados = [(float(p['x']), float(p['y'])) for p in previos] + list(slots)

        for k, id_nido in enumerate(ids):
            # Si la zona está llena aun con la rejilla apretada al PISO físico,
            # ya no hay casilla. Antes aquí se repetía la última:
            #     x, y = slots[k] if k < len(slots) else slots[-1]
            # y eso ponía varios nidos en las mismas coordenadas exactas, o sea
            # varias nidadas en un hoyo. Ahora se busca el hueco más holgado
            # que quede: sigue siendo un corral sin lugar y el resultado tiene
            # que verse mal, pero se puede ejecutar y se puede medir.
            if k < len(slots):
                x, y = slots[k]
            elif lim_z:
                x, y = posicion_mas_libre(lim_z, colocados)
                colocados.append((x, y))
            else:
                continue
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
        # Mismo caso que en el llenado secuencial: `pos.append(pos[-1])`
        # duplicaba coordenadas y sembraba dos nidos en el mismo punto.
        ocupadas = [(float(p['x']), float(p['y']))
                    for p in (nidos_ocupados or [])] + list(pos)
        while len(pos) < len(ids):
            hueco = posicion_mas_libre(lim, ocupadas)
            pos.append(hueco)
            ocupadas.append(hueco)
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
        # Colocación LIBRE, no sobre la rejilla.
        #
        # La mitad aleatoria de la población existe para aportar diversidad de
        # posiciones. Si nace sobre las mismas casillas que la mitad sembrada,
        # la población entera arranca en la rejilla y ninguna mutación logra
        # sacarla de ahí: medido, las 100 colocaciones quedaban en 27 columnas
        # por 10 filas, idénticas al llenado secuencial.
        #
        # Se muestrea por rechazo: puntos al azar dentro de la zona que
        # respeten la separación frente a lo ya colocado y a lo ya enterrado.
        lim = gestor.limites_zona(f"zona_{especie}")
        if not lim:
            continue
        sep = float(e['sep_min'])
        paso = gestor.separacion_efectiva.get(f"zona_{especie}")
        if paso:
            sep = min(sep, float(paso))
        # Nunca por debajo del piso fisico: la mitad aleatoria de la poblacion
        # coloca libremente, y sin este suelo podia nacer con nidos a menos de
        # lo que mide una camara de huevos.
        if hasattr(gestor, 'piso'):
            sep = max(sep, gestor.piso(especie))
        sep_sq = (sep - 0.05) ** 2

        ocupadas = [(float(o['x']), float(o['y']))
                    for o in (nidos_ocupados or [])]
        pos = []
        for _ in ids:
            colocado = None
            for _intento in range(40):
                cx = round(random.uniform(lim['xmin'], lim['xmax']), 1)
                cy = round(random.uniform(lim['ymin'], lim['ymax']), 1)
                if all((cx - ox) ** 2 + (cy - oy) ** 2 >= sep_sq
                       for ox, oy in ocupadas):
                    colocado = (cx, cy)
                    break
            if colocado is None:
                # Zona apretada: se cae a la rejilla, que siempre es válida.
                libres = [s for s in gestor.slots_suficientes(
                    f"zona_{especie}", e['sep_min'], nidos_ocupados, len(ids))
                    if s not in pos]
                if not libres:
                    break
                colocado = libres[0]
            pos.append(colocado)
            ocupadas.append(colocado)
        if not pos:
            continue

        for id_nido, (x, y) in zip(ids, pos):
            p = round(random.uniform(e['prof_min'], e['prof_max']), 1)
            genes.append(Gen(id_nido, especie, x, y, p))

    return Individuo(genes)
