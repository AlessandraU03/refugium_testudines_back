import random
import math
import numpy as np
from cromosoma import Individuo, ORDENES_ZONAS, resolver_gestor

# Fracción de la descendencia a la que se le aplica la reparación de orden.
#
# MEDIDO, no elegido a mano. Con 100 nidos nuevos sobre 70 enterrados, dos
# corridas por valor (pop=30, gen=40):
#
#   p_rep   aptitud   cumplimiento separación   nidos en riesgo letal
#   0.00    0.7272    1.000                     45.0
#   0.15    0.7119    0.990                     46.5
#   0.35    0.7059    0.990                     47.5
#   0.65    0.6991    0.980                     47.5
#   1.00    0.5524    0.880                     48.0
#
# La reparación no ayuda: perjudica de forma monótona. Aplicada siempre -que
# era el comportamiento original- pega a toda la población a la misma rejilla,
# el AG deja de colocar nidos para limitarse a elegir casillas y el
# cumplimiento de separación cae a 0.88.
#
# Queda en 0.0. El código se conserva porque el llenado en serpentina tiene un
# valor OPERATIVO que la aptitud no mide: el personal siembra recorriendo
# hileras, y una distribución dispersa es más incómoda de ejecutar en campo. Si
# alguna vez ese criterio pesa más que el biológico, subir este valor y asumir
# el costo que muestra la tabla.
_PROB_REPARACION = 0.0


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

    # El reparto de franjas tambien se hereda: es parte del cromosoma, no una
    # constante del problema. Cada hijo toma el de uno de los padres.
    if random.random() < 0.5:
        i1, i2 = p1.idx_orden, p2.idx_orden
    else:
        i1, i2 = p2.idx_orden, p1.idx_orden
    return Individuo(h1_genes, i1), Individuo(h2_genes, i2)


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

    # ── Mutación 0: Reparto de franjas ────────────────────────────────────────
    # Cambia a qué especie le toca cada franja del corral. Es la mutación de
    # mayor alcance: con la malla cubriendo sólo parte del corral, mover a
    # prieta de la franja al sol a la franja sombreada le sube la sombra de
    # 0.11 a 0.97, y con ella le cambia la proporción sexual.
    #
    # Al cambiar el reparto las coordenadas viejas quedan fuera de zona, así
    # que hay que re-sembrar: cada nido se reubica en las casillas del nuevo
    # reparto conservando su profundidad, que es lo que el cambio de zona no
    # afecta.
    # Cuantos repartos puede usar este corral. Si ya hay nidos enterrados el
    # gestor trae uno solo, el que el corral ya tiene, y esta mutacion no hace
    # nada: cambiar de franja obligaria a desenterrar lo ya sembrado.
    n_ordenes = len(gestor) if hasattr(gestor, '__len__') else 1
    if n_ordenes > 1 and random.random() < prob_mut:
        nuevo = random.randrange(n_ordenes)
        if nuevo != ind.idx_orden:
            ind.idx_orden = nuevo
            g_nuevo = resolver_gestor(gestor, nuevo)

            por_esp_re = {}
            for gen in genes:
                por_esp_re.setdefault(gen.especie, []).append(gen)

            for esp, gs in por_esp_re.items():
                xs_p, ys_p = previos_esp.get(esp, (np.array([]), np.array([])))
                prev_dicts = [{'x': float(px), 'y': float(py), 'especie': esp}
                              for px, py in zip(xs_p, ys_p)]
                slots = g_nuevo.slots_suficientes(
                    f"zona_{esp}", base[esp]['sep_min'], prev_dicts, len(gs))
                if not slots:
                    continue
                for k, gen in enumerate(gs):
                    gen.x, gen.y = slots[k] if k < len(slots) else slots[-1]

            # Las coordenadas cambiaron: los arreglos de arriba quedaron viejos.
            all_xs = np.array([g.x for g in genes])
            all_ys = np.array([g.y for g in genes])

    # A partir de aquí se trabaja sobre la geometría del reparto de ESTE individuo.
    gestor = resolver_gestor(gestor, ind.idx_orden)

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

        # TODOS los vecinos, no solo los de su especie: los demas nidos de la
        # jornada y los ya enterrados, sean de la especie que sean.
        #
        # Comparando solo contra la misma especie, esta mutacion podia mover un
        # nido justo encima de uno de otra especie ya sembrado. Medido: llego a
        # dejar un nido nuevo a 18 cm de uno enterrado. La separacion minima es
        # una norma biologica por especie, pero el hoyo ya excavado es un hecho
        # fisico que no distingue especies.
        mask = np.ones(len(genes), dtype=bool)
        mask[idx] = False
        xs_partes = [all_xs[mask]]
        ys_partes = [all_ys[mask]]
        for _e, (_px, _py) in previos_esp.items():
            xs_partes.append(_px)
            ys_partes.append(_py)

        xs_all = np.concatenate(xs_partes)
        ys_all = np.concatenate(ys_partes)

        if len(xs_all) == 0:
            continue

        # Contra la separacion ALCANZABLE, no contra la de la norma.
        #
        # Cuando el corral se satura la rejilla se aprieta (p. ej. a 80 cm para
        # 270 nidos en 240 m2). Midiendo contra los 100 cm de la norma, TODOS
        # los nidos violan y ningun candidato mejora la cuenta, asi que esta
        # mutacion dejaba de hacer efecto justo cuando mas falta hace.
        sep_efectiva = gestor.separacion_efectiva.get(
            gen.zona_correcta(), sep_min)
        sep_min_sq = (sep_efectiva - 0.05) ** 2
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

        # Entre los candidatos que menos violan, quedarse con el que MAS lejos
        # queda del vecino mas cercano. Sin este desempate, "menos violaciones"
        # admitia un candidato pegado a un solo vecino, y de ahi salian nidos a
        # 18 cm de uno enterrado: una violacion, pero fisicamente encima.
        menos = viols_cand.min()
        empatados = np.flatnonzero(viols_cand == menos)
        cercania = dist_sq_cand[empatados].min(axis=1)
        best_idx = int(empatados[int(np.argmax(cercania))])
        mejor_viols = viols_cand[best_idx]

        # Y nunca aceptar un movimiento que acerque el nido a su vecino mas
        # proximo: aunque baje la cuenta de violaciones, empeoraria el
        # solapamiento fisico.
        if float(dist_sq_cand[best_idx].min()) < float(dist_sq_actual.min()):
            continue

        if mejor_viols < viols_actual:
            new_x = float(cxs[best_idx])
            new_y = float(cys[best_idx])
            gen.x = new_x
            gen.y = new_y
            all_xs[idx] = new_x
            all_ys[idx] = new_y

    # ── Mutación 3: Orden de llenado ─────────────────────────────────────────
    # La mutación de posición sólo actúa sobre nidos que violan sep_min, así que
    # sin esto el AG nunca reordena una distribución válida pero dispersa.
    # Jala nidos fuera de la secuencia hacia las casillas libres del tramo inicial.
    por_esp = {}
    for gen in genes:
        por_esp.setdefault(gen.especie, []).append(gen)

    # NO se aplica siempre. Antes sí, y ese era el problema: pegaba a TODOS los
    # individuos a la misma rejilla, de modo que el AG dejaba de colocar nidos
    # y se limitaba a elegir casillas. Medido con 100 nidos, la población
    # entera acababa en 27 columnas por 10 filas, idéntica al llenado
    # secuencial, y por eso la ganancia contra la referencia era siempre 0.0.
    #
    # Aplicada sólo a una parte de la descendencia, sigue cumpliendo su papel
    # -mantener colocaciones ordenadas y sembrables- sin imponerle la rejilla
    # al resto, que queda libre para buscar sombra o repartir densidad.
    aplica_reparacion = random.random() < _PROB_REPARACION
    # Sólo mueve nidos a casillas de la misma rejilla, que son biológicamente
    # equivalentes (mismo sep_min, misma zona), así que no sacrifica diversidad útil.
    for esp, gs in (por_esp.items() if aplica_reparacion else ()):
        sep_min = base[esp]['sep_min']
        xs_prev, ys_prev = previos_esp.get(esp, (np.array([]), np.array([])))
        previos_dicts = [{'x': float(px), 'y': float(py), 'especie': esp}
                         for px, py in zip(xs_prev, ys_prev)]

        # La misma rejilla que usan quien coloca y quien juzga el orden. Si se
        # pide de otro modo, esta mutación repara los nidos hacia casillas que
        # el término de orden no reconoce, y el esfuerzo se pierde.
        slots = gestor.slots_suficientes(f"zona_{esp}", sep_min, previos_dicts,
                                         len(gs))
        if not slots:
            continue

        # slots_ordenados ya devuelve coordenadas redondeadas a 1 decimal, y Gen
        # redondea las suyas al construirse: basta redondear una vez por nido.
        prefijo = slots[:len(gs)]
        claves_prefijo = set(prefijo)
        claves_gs = [(round(g.x, 1), round(g.y, 1)) for g in gs]
        ocupadas = set(claves_gs)

        libres = [s for s in prefijo if s not in ocupadas]
        fuera  = [g for g, k in zip(gs, claves_gs) if k not in claves_prefijo]

        # Las casillas del prefijo ya distan sep_min entre sí y de los nidos
        # previos, así que sólo pueden chocar con nidos aún fuera de secuencia.
        if not libres or not fuera:
            continue

        pendientes = list(libres)
        fx = np.array([g.x for g in fuera], dtype=float)
        fy = np.array([g.y for g in fuera], dtype=float)
        # Igual que arriba: la prueba de aceptacion usa el paso real de la
        # rejilla que acaba de devolver slots_suficientes. Con sep_min la
        # condicion nunca se cumplia en un corral saturado y la reparacion de
        # orden no movia un solo nido.
        sep_efectiva = gestor.separacion_efectiva.get(f"zona_{esp}", sep_min)
        sep_sq = (sep_efectiva - 0.05) ** 2

        for i, gen in enumerate(fuera):
            if not pendientes:
                break
            for k, destino in enumerate(pendientes):
                d2 = (fx - destino[0])**2 + (fy - destino[1])**2
                d2[i] = np.inf   # no compararse consigo mismo
                if d2.min() >= sep_sq:
                    gen.x, gen.y = destino[0], destino[1]
                    fx[i], fy[i] = gen.x, gen.y
                    pendientes.pop(k)
                    break

    # ── Mutación 4: Colocación libre ─────────────────────────────────────────
    # Propone mover un nido a cualquier punto de su zona, no a una casilla de
    # la rejilla, y acepta el movimiento sólo si sigue respetando la separación
    # frente a TODOS los vecinos: los nidos nuevos y los ya enterrados, sean de
    # la especie que sean.
    #
    # Es lo que convierte al AG en un colocador y no en un seleccionador de
    # casillas. La rejilla garantiza la separación por construcción, pero a
    # cambio fija las posiciones; aquí la separación se verifica, así que el
    # nido puede ir a donde la sombra o la densidad lo hagan rendir más.
    xs_otros = [np.array([g.x for g in genes])]
    ys_otros = [np.array([g.y for g in genes])]
    for e, (px, py) in previos_esp.items():
        xs_otros.append(px)
        ys_otros.append(py)
    todos_x = np.concatenate(xs_otros)
    todos_y = np.concatenate(ys_otros)

    for idx, gen in enumerate(genes):
        if random.random() >= prob_mut:
            continue
        lim = gestor.limites_zona(gen.zona_correcta())
        if not lim:
            continue

        # Separación exigida: la de la norma, o la del paso real de la rejilla
        # si el corral está tan saturado que hubo que apretarla.
        sep_req = float(base[gen.especie]['sep_min'])
        paso = gestor.separacion_efectiva.get(gen.zona_correcta())
        if paso:
            sep_req = min(sep_req, float(paso))
        sep_sq = (sep_req - 0.05) ** 2

        cx = round(random.uniform(lim['xmin'], lim['xmax']), 1)
        cy = round(random.uniform(lim['ymin'], lim['ymax']), 1)

        d2 = (todos_x - cx) ** 2 + (todos_y - cy) ** 2
        d2[idx] = np.inf                       # su propia posición actual
        if float(d2.min()) < sep_sq:
            continue                           # invadiría a un vecino

        todos_x[idx] = cx
        todos_y[idx] = cy
        gen.x, gen.y = cx, cy

    return ind


def poda_elitismo(poblacion, descendencia, tam_pob, n_elite=2):
    """Selección μ+λ: conserva los n_elite mejores y completa aleatoriamente."""
    todos = sorted(poblacion + descendencia,
                   key=lambda i: i.fitness, reverse=True)
    elite = todos[:n_elite]
    resto = todos[n_elite:]
    random.shuffle(resto)
    return elite + resto[:tam_pob - n_elite]
