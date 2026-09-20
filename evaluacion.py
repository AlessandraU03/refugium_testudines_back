
import numpy as np
from datetime import datetime, timedelta

# Extremos absolutos leídos del CSV de tasa_eclosion:
#   mínimo: laúd peores condiciones = 0.40
#   máximo: golfina condiciones perfectas = 0.90
_TASA_MIN_ABS = 0.40
_TASA_MAX_ABS = 0.90


def calcular_pts_window(fecha_siembra_dt, especie, base):
    """
    Calcula las fechas de inicio y fin del Período Termosensible (PTS).
    El PTS es el período crítico donde la temperatura determina el sexo.

    Args:
        fecha_siembra_dt: datetime del día de siembra
        especie: nombre de especie (golfina, prieta, laud)
        base: diccionario base de conocimiento con parámetros de especie

    Returns:
        dict con: pts_inicio, pts_fin, pts_inicio_dia, pts_fin_dia, semana_critica
    """
    e = base.get(especie, {})

    dias_prom = e.get('dias_prom', 50)
    pts_inicio_dia = e.get('pts_inicio_dia', int(dias_prom / 3))
    pts_fin_dia = e.get('pts_fin_dia', int(2 * dias_prom / 3))
    semana_critica = e.get('semana_critica', 3)

    pts_inicio = fecha_siembra_dt + timedelta(days=pts_inicio_dia)
    pts_fin = fecha_siembra_dt + timedelta(days=pts_fin_dia)

    return {
        'pts_inicio': pts_inicio.strftime('%Y-%m-%d'),
        'pts_fin': pts_fin.strftime('%Y-%m-%d'),
        'pts_inicio_dia': pts_inicio_dia,
        'pts_fin_dia': pts_fin_dia,
        'semana_critica': semana_critica
    }


def calcular_semana_incubacion(fecha_siembra_dt, fecha_actual_dt, base, especie):
    """
    Calcula en qué semana de incubación se encuentra un nido actualmente.

    Args:
        fecha_siembra_dt: datetime del día de siembra
        fecha_actual_dt: datetime actual
        base: diccionario base de conocimiento
        especie: nombre de especie

    Returns:
        dict con: semana (int), dias_transcurridos, dias_faltantes, en_pts (bool)
    """
    dias_transcurridos = (fecha_actual_dt - fecha_siembra_dt).days

    e = base.get(especie, {})
    dias_prom = e.get('dias_prom', 50)

    semana = max(1, (dias_transcurridos // 7) + 1)

    pts_inicio = e.get('pts_inicio_dia', int(dias_prom / 3))
    pts_fin = e.get('pts_fin_dia', int(2 * dias_prom / 3))
    en_pts = pts_inicio <= dias_transcurridos <= pts_fin

    dias_faltantes = max(0, dias_prom - dias_transcurridos)

    return {
        'semana': semana,
        'dias_transcurridos': dias_transcurridos,
        'dias_faltantes': dias_faltantes,
        'en_pts': en_pts,
        'semana_critica': (semana == e.get('semana_critica', 3))
    }


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

    if isinstance(nidos_previos, dict) and "__cache__" in nidos_previos:
        por_esp_prev = nidos_previos
    else:
        por_esp_prev = _previos_por_especie(nidos_previos)

    total_tasa = 0.0
    N_total    = 0

    especies = set(por_esp_nuevo.keys()) | set(por_esp_prev.keys())
    if "__cache__" in especies:
        especies.remove("__cache__")

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
            xs_p   = p_data[0] if isinstance(p_data[0], np.ndarray) else np.array(p_data[0])
            ys_p   = p_data[1] if isinstance(p_data[1], np.ndarray) else np.array(p_data[1])
            if isinstance(p_data[2], np.ndarray):
                prfs_p = p_data[2]
            else:
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


# ── FITNESS ──────────────────────────────────────────────────────

def calcular_fitness(individuo, gestor, base, corral, nidos_previos=None,
                     mes=None):
    """Aptitud de una colocación.

    Aquí convivían dos funciones de aptitud: ésta y otra, llamada tradicional,
    que restaba tres variables con pesos elegidos a mano. El AG nunca la
    llamaba, pero era el valor por omisión del parámetro, así que una llamada
    distraída evaluaba con pesos inventados y daba otro resultado. El sistema
    tiene ahora una sola función de aptitud y todos sus coeficientes citados.
    """
    return calcular_fitness_cientifico(individuo, gestor, base, corral,
                                       nidos_previos, mes)


def evaluar_poblacion(poblacion, gestor, base, corral, nidos_previos=None,
                      mes=None):
    for ind in poblacion:
        calcular_fitness(ind, gestor, base, corral, nidos_previos, mes)
    return poblacion


# ── FUNCIONES CIENTÍFICAS (Coeficientes de Literatura) ──────────────────────────
# Basadas en compilación de papers: Lepidochelys olivacea eclosión factors
# Fuentes: Springer Nature, MTN, ResearchGate, SciELO
# Ver: scratchpad/tortuga_datos_cuantitativos.md

def calcular_efecto_profundidad_cientifico(profundidad_cm, prof_min_cm, prof_max_cm):
    """Cumplimiento del rango de profundidad de la NOM-162-SEMARNAT-2012.

    La NOM-162 (sección 6.8.4, Tabla 1) fija para la tortuga golfina una
    profundidad total de nido de 40 a 50 cm. La norma da un RANGO, no una
    pérdida por centímetro, y la literatura revisada tampoco documenta esa
    pendiente para la especie: el único estudio de profundidad en reubicación
    encontrado (Najwa-Sawawi et al. 2021, Saudi J. Biol. Sci. 28(9):5053-5060)
    es de tortuga verde, a unos 80 cm, y no halló diferencia significativa de
    eclosión.

    Por eso dentro del rango el efecto vale 1.0: toda profundidad que cumple
    la norma es igual de válida. Fuera del rango la penalización no es una
    estimación biológica sino una restricción que empuja la solución de vuelta
    al rango. Su escala es la mitad del propio rango (5 cm para golfina), de
    modo que no introduce ningún número ajeno a la norma.

    Sustituye un coeficiente anterior de 0.01 por cm sin fuente, cuyo
    comentario además contradecía la fórmula: decía que 10 cm costaban 15 % y
    el cálculo daba 10 %.

    Returns:
        float [0.0, 1.0]: 1.0 dentro del rango de la norma.
    """
    p = float(profundidad_cm)
    lo, hi = float(prof_min_cm), float(prof_max_cm)
    if lo <= p <= hi:
        return 1.0
    semi_rango = (hi - lo) / 2.0
    if semi_rango <= 0:
        return 0.0
    exceso = (lo - p) if p < lo else (p - hi)
    return max(0.0, 1.0 - exceso / semi_rango)


def cumplimiento_separacion(genes, base, nidos_previos=None, sep_alcanzable=None):
    """Fracción de nidos nuevos que respetan la separación contra TODO vecino.

    Se mide contra los nidos nuevos y contra los ya enterrados, sin importar la
    especie: la separación mínima es una norma biológica, pero el espacio
    ocupado es un hecho físico. Entre dos especies distintas se exige la mayor
    de sus dos separaciones, que es la condición más restrictiva de las dos.

    Este término es lo que le permite al AG COLOCAR en vez de limitarse a
    elegir casillas de una rejilla: mientras la separación se garantizaba sólo
    por construcción de la cuadrícula, cualquier posición fuera de ella era
    inválida y el algoritmo no tenía libertad real. Ahora puede mover un nido a
    donde le convenga -por sombra, por densidad- y la aptitud le cobra si
    invade el espacio de otro.

    Returns:
        float [0.0, 1.0]: 1.0 = ningún nido invade la separación de otro.
    """
    if not genes:
        return 1.0

    previos = _previos_lista(nidos_previos)
    vecinos = ([(float(g.x), float(g.y), g.especie) for g in genes] +
               [(float(p['x']), float(p['y']), p.get('especie', 'golfina'))
                for p in previos])

    # Separación exigible: la de la norma, salvo que el corral esté tan
    # saturado que la rejilla haya tenido que apretarse. En ese caso se exige
    # la ALCANZABLE, no la de la norma.
    #
    # Es la lección del término que se retiró: midiendo contra los 100 cm de la
    # norma en un corral comprimido a 80, ningún nido puede cumplir, el término
    # vale 0 para todas las colocaciones y deja de distinguir la buena de la
    # mala. Contra la alcanzable, el reparto uniforme obtiene 1.0 y cualquier
    # apiñamiento innecesario baja.
    seps = {}
    for e in base:
        norma = float(base[e].get('sep_min', 100.0))
        alc = (sep_alcanzable or {}).get(e)
        seps[e] = min(norma, float(alc)) if alc else norma

    xs = np.array([v[0] for v in vecinos])
    ys = np.array([v[1] for v in vecinos])

    cumplen = 0
    for i, g in enumerate(genes):
        dx = xs - float(g.x)
        dy = ys - float(g.y)
        d2 = dx * dx + dy * dy
        d2[i] = np.inf                      # no compararse consigo mismo
        # Separación exigida frente a cada vecino: la mayor de las dos especies
        req = np.array([max(seps.get(g.especie, 100.0), seps.get(v[2], 100.0))
                        for v in vecinos])
        if np.all(d2 >= (req - 0.05) ** 2):
            cumplen += 1

    return cumplen / float(len(genes))


def calcular_efecto_separacion_cientifico(nidos, base):
    """
    Factor de separación: proporción de nidos cuyo vecino más cercano de la
    misma especie respeta la separación mínima documentada.

    La separación sale de separacion_minima.csv por especie (NOM-162-SEMARNAT-2012):
    golfina 100 cm, prieta 120 cm, laúd 150 cm.

    Args:
        nidos: lista de genes con (x, y) en cm
        base: base de conocimiento por especie

    Returns:
        float [0.0, 1.0]: 1.0 = ningún nido invade la separación de otro
    """
    if len(nidos) <= 1:
        return 1.0

    por_esp = {}
    for n in nidos:
        por_esp.setdefault(n.especie, []).append(n)

    violaciones = 0
    for esp, gs in por_esp.items():
        if len(gs) <= 1:
            continue
        sep_min = base[esp]['sep_min']
        xs = np.array([g.x for g in gs])
        ys = np.array([g.y for g in gs])
        dist_sq = (xs[:, None] - xs[None, :])**2 + (ys[:, None] - ys[None, :])**2
        np.fill_diagonal(dist_sq, np.inf)
        violaciones += int((dist_sq.min(axis=1) < sep_min**2).sum())

    return 1.0 - violaciones / len(nidos)


# NOTA: aquí existía calcular_efecto_densidad_cientifico(), eliminada el
# 2026-09-06. Usaba un umbral de 25 nidos/m² que no proviene de ninguna fuente:
# la revisión de literatura concluyó que no hay densidad máxima segura
# documentada para corrales de incubación. Además el término era inerte —
# devolvía 1.0 incluso a 3.5x la capacidad máxima del corral (400 nidos en
# 1400 m² = 0.29 nidos/m²), así que sólo sumaba una constante al fitness.


def _previos_lista(nidos_previos):
    """Normaliza nidos_previos (lista de dicts o caché NumPy) a lista de dicts."""
    if not nidos_previos:
        return []
    if isinstance(nidos_previos, dict) and "__cache__" in nidos_previos:
        salida = []
        for esp, p in nidos_previos.items():
            if esp == "__cache__":
                continue
            salida.extend({'x': float(x), 'y': float(y), 'especie': esp}
                          for x, y in zip(p[0], p[1]))
        return salida
    return list(nidos_previos)


def calcular_orden_operativo(individuo, gestor, base, nidos_previos=None):
    """
    Fracción de nidos que caen dentro del tramo inicial de la secuencia de
    llenado de su zona. 1.0 = el corral se llenó en orden, sin huecos.

    NO es un criterio biológico: mide si la distribución se puede sembrar
    siguiendo la serpentina en vez de ir buscando casillas sueltas. Por eso
    entra al fitness sólo como desempate (ver _EPS_ORDEN).
    """
    genes = individuo.genes
    if not genes:
        return 1.0

    # Cada individuo puede llevar su propio reparto de franjas, asi que la
    # rejilla contra la que se juzga su orden de llenado es la de SU reparto.
    from cromosoma import resolver_gestor
    gestor = resolver_gestor(gestor, getattr(individuo, 'idx_orden', 0))

    por_esp = {}
    for g in genes:
        por_esp.setdefault(g.especie, []).append(g)

    previos = _previos_lista(nidos_previos)
    en_orden = 0

    for esp, gs in por_esp.items():
        # La misma rejilla que usa quien coloca. Pedirla de otro modo hacia que
        # este termino le diera 0.0000 al llenado secuencial.
        slots = gestor.slots_suficientes(f"zona_{esp}", base[esp]['sep_min'],
                                         previos, len(gs))
        prefijo = {(round(x, 1), round(y, 1)) for x, y in slots[:len(gs)]}
        en_orden += sum(1 for g in gs
                        if (round(g.x, 1), round(g.y, 1)) in prefijo)

    return en_orden / len(genes)


# Peso del término operativo. NO es un coeficiente biológico: sólo hace que el
# orden de llenado desempate entre distribuciones biológicamente equivalentes.
# Al ser 0.01, ninguna ganancia de orden puede compensar una diferencia
# biológica mayor a ese margen (escalarización lexicográfica).
_EPS_ORDEN = 0.01

# Regiones sombreadas del corral, en centímetros, leídas de csv/sombra_corral.csv.
#
# La sombra es POSICIONAL: `termico.en_sombra(x, y, ...)` decide nido por nido.
# Si la malla cubre el corral entero, todos los nidos reciben el mismo trato y
# la posición no influye en el sexo. Si cubre sólo una parte, dónde queda cada
# nido decide su proporción sexual, y ahí el AG tiene un problema real que
# resolver: en septiembre el rango entre estar bajo la malla o no son 45 puntos
# porcentuales de hembras.
#
# La cobertura real del corral de Puerto Arista NO está confirmada. El CSV
# asume el corral completo y lo declara.
def _cargar_sombras(carpeta_csv=None):
    """Lee las regiones sombreadas. Lista vacía si no hay archivo."""
    import os
    from cromosoma import cargar_csv

    # Una sola fuente: la malla de csv/sitio.csv. Antes existia ademas
    # sombra_corral.csv, que decia cobertura completa mientras sitio.csv decia
    # 0-1800 cm; dos configuraciones de la misma malla que se contradecian.
    try:
        import solar
        m = solar.cargar_sitio(carpeta_csv)['malla']
        return [dict(m)]
    except Exception:
        return []


SOMBRAS = _cargar_sombras()


# Proporción sexual de referencia, en porcentaje de hembras.
#
# NO es un 50/50 por simetría: las poblaciones de tortuga marina son
# naturalmente sesgadas hacia hembras, y forzar el equilibrio no tiene respaldo.
# Es la proporción que implica la temperatura MEDIDA en un corral mexicano
# comparable: de la Torre-Robles, Buenrostro-Silva y García-Grajales (2017)
# registraron 30.1 °C durante el segundo tercio de la incubación en nidos de
# golfina a ~45 cm en el corral de San Juan Chacahua, Oaxaca, con 86.6 % de
# eclosión. Esa temperatura, en la ecuación de Girondot con los parámetros de
# Sandoval et al. (2020), da 55.9 % de hembras.
#
# ADVERTENCIA: el modelo estima 32.01 °C para Puerto Arista en septiembre,
# casi dos grados por encima de esa medición. De esa diferencia depende que el
# corral produzca 96 % o 56 % de hembras. Mientras no se mida la arena del
# corral, este objetivo orienta la búsqueda pero sus cifras no son un hallazgo.
OBJETIVO_PCT_HEMBRAS = 55.9

# Peso del objetivo de sexo. Misma escalarización lexicográfica que _EPS_ORDEN:
# la proporción sexual sólo desempata entre colocaciones que ya son
# equivalentes en eclosión. El orden no es arbitrario: una cría que no eclosiona
# no tiene sexo, así que el sexo no puede comprarse con crías perdidas.
_EPS_SEXO = 0.05


_SITIO_CACHE = None


def _sitio():
    """Configuracion del sitio (latitud, orientacion, malla), leida una vez.

    Devuelve None si no hay csv/sitio.csv, y entonces el modelo termico cae al
    criterio binario de sombra que habia antes. Asi el sistema sigue corriendo
    en una instalacion que no haya declarado su geometria.
    """
    global _SITIO_CACHE
    if _SITIO_CACHE is None:
        try:
            import solar
            _SITIO_CACHE = solar.cargar_sitio()
        except Exception:
            _SITIO_CACHE = False
    return _SITIO_CACHE or None


def _evaluar_colocacion_termica(individuo, base, nidos_previos=None, mes=None):
    """Corre el modelo térmico sobre esta colocación y devuelve todo de una vez.

    De aquí salen los tres términos biológicos de la aptitud: cuántas crías
    sobreviven al hacinamiento, qué proporción sexual resulta, y cuántos nidos
    quedan por encima del límite letal. Se calculan juntos porque los tres
    salen del mismo recorrido por los nidos.

    Los nidos ya sembrados en jornadas anteriores cuentan como vecinos: un
    nido nuevo no puede ignorar lo que ya está enterrado a su lado.

    Sobre `mes`: antes se deducía de `genes[0].fecha_siembra`, que SIEMPRE vale
    None en los genes que construye el AG. El resultado es que la aptitud
    evaluaba septiembre sin importar la fecha de la jornada. Para la eclosión
    daba igual, porque sólo depende de la densidad; para la proporción sexual
    habría sido un error grave. Ahora se recibe explícitamente.
    """
    import termico

    genes = list(individuo.genes)
    if not genes:
        return {'indice_eclosion': 1.0,
                'resumen': termico.predecir_conjunto([]), 'por_nido': []}

    previos = _previos_lista(nidos_previos)
    vecinos = genes + [
        type('N', (), {'x': float(p['x']), 'y': float(p['y']),
                       'especie': p.get('especie', 'golfina'),
                       'num_huevas': p.get('num_huevas', 0)})()
        for p in previos
    ]

    tasas = {e: base[e].get('tasa_eclosion', 0.75) for e in base}
    if mes is None and getattr(genes[0], 'fecha_siembra', None):
        try:
            mes = datetime.strptime(genes[0].fecha_siembra, '%Y-%m-%d').month
        except (ValueError, TypeError):
            mes = None

    # Pivote y parametro de forma propios de cada especie. Sin esto, prieta y
    # laud se evaluaban con la pivote de golfina, que esta 0.75 C mas arriba
    # que la de prieta: mas de veinte puntos de diferencia en la proporcion de
    # hembras dentro de la zona empinada de la curva.
    params_especie = {
        e: {'pivote': base[e].get('pivote_temp'),
            's': base[e].get('s_parameter')}
        for e in base
    }

    # Dia del anio para la geometria solar. Se toma el dia 15 del mes de
    # siembra: la declinacion solar cambia unos 0.3 grados por dia, asi que
    # dentro de un mes la trayectoria del sol se mueve poco, y la arena a 45 cm
    # responde al promedio de varios dias, no a la insolacion de una fecha
    # exacta. Usar el mes evita arrastrar la fecha completa por las firmas del
    # AG; si algun dia hace falta precision diaria, hay que threadearla.
    dia = None
    sitio = _sitio()
    if mes is not None:
        dia = (int(mes) - 1) * 30 + 15

    return termico.evaluar_colocacion(
        vecinos, tasas, mes=mes, rectangulos_sombra=SOMBRAS,
        huevos_por_defecto=termico.huevos_del_mes(mes) if mes else None,
        sitio=sitio, dia_del_anio=dia, params_especie=params_especie)


def _indice_eclosion(individuo, base, nidos_previos=None, mes=None):
    """Sólo el índice de eclosión. Se conserva por compatibilidad."""
    return _evaluar_colocacion_termica(
        individuo, base, nidos_previos, mes)['indice_eclosion']


def calcular_fitness_cientifico(individuo, gestor, base, corral,
                                nidos_previos=None, mes=None):
    """
    Aptitud de una colocación, con todos los coeficientes citados.

        eclosión = crías que sobreviven al hacinamiento (Honarvar et al. 2008)
        prof     = cumplimiento del rango de la NOM-162 (40–50 cm en golfina)
        letal    = fracción de nidos por debajo de los 36 °C del último tercio
        sexo     = cercanía a la proporción sexual de referencia

        biológico = eclosión × prof × letal
        fitness   = (biológico + _EPS_SEXO × sexo + _EPS_ORDEN × orden)
                    / (1 + _EPS_SEXO + _EPS_ORDEN)

    POR QUÉ SE QUITÓ EL TÉRMINO DE SEPARACIÓN
    -----------------------------------------
    Antes la aptitud era `eclosión × (prof + separación) / 2`. Ese término
    contaba qué fracción de nidos respeta los 100 cm de la NOM-162, y se
    volvía perverso justo cuando más importa. Medido con 270 nidos en el
    corral de 240 m²: la rejilla tiene que apretarse a 88.9 cm, de modo que
    NINGÚN nido puede cumplir la norma y el llenado uniforme —que es el óptimo
    biológico, densidad 1.00 nidos/m² en todo el corral— obtenía separación
    0.000 y aptitud 0.500. El AG subía ese conteo a 0.72 separando unos nidos
    y apiñando otros hasta 3 nidos/m², y alcanzaba aptitud 0.86 produciendo
    DIEZ CRÍAS MENOS. Un término no biológico le estaba ganando a la curva
    medida de Honarvar.

    La separación sigue siendo obligatoria, pero donde corresponde: la rejilla
    de `slots_ordenados` la impone al construir las casillas, y aquí se reporta
    como métrica de cumplimiento (`individuo.v2`), no como objetivo.

    SOBRE LOS DOS EPSILON
    ---------------------
    Escalarización lexicográfica, el mismo patrón que ya usaba el orden de
    llenado. No son pesos elegidos a mano para balancear objetivos: son
    márgenes tan pequeños que sólo desempatan entre colocaciones ya
    equivalentes en lo biológico. El orden de prioridad tiene una razón, no una
    preferencia: una cría que no eclosiona no tiene sexo, así que la proporción
    sexual nunca puede comprarse con crías perdidas.

    Returns:
        float [0.0, 1.0]
    """
    if individuo.num_nidos() == 0:
        individuo.fitness = 0.0
        individuo.v1 = individuo.v2 = individuo.v3 = 0.0
        return 0.0

    # Un solo recorrido del modelo térmico da los tres términos biológicos.
    r = _evaluar_colocacion_termica(individuo, base, nidos_previos, mes)
    resumen = r['resumen']
    eclosion = r['indice_eclosion']

    # Riesgo letal: los nidos que rebasarían 36 °C en el último tercio no son
    # un matiz, son nidada perdida. Entra como factor, no como desempate.
    n_nidos = max(1, resumen['n_nidos'])
    letal = 1.0 - resumen['nidos_en_riesgo_termico'] / float(n_nidos)

    # Sexo: distancia a la proporción de referencia, normalizada a [0,1].
    sexo = 1.0 - abs(resumen['pct_hembras'] - OBJETIVO_PCT_HEMBRAS) / 100.0
    sexo = float(np.clip(sexo, 0.0, 1.0))

    efecto_prof = np.mean([
        calcular_efecto_profundidad_cientifico(
            g.prof, base[g.especie]['prof_min'], base[g.especie]['prof_max'])
        for g in individuo.genes
    ]) if individuo.genes else 1.0

    # Cumplimiento de separación contra TODO vecino, incluidos los nidos ya
    # enterrados y los de otras especies. Vuelve a ser objetivo, pero medido
    # contra la separación alcanzable, que es lo que lo hacía degenerar antes.
    from cromosoma import resolver_gestor
    g_ind = resolver_gestor(gestor, getattr(individuo, 'idx_orden', 0))
    sep_alc = {}
    for nombre, paso in getattr(g_ind, 'separacion_efectiva', {}).items():
        sep_alc[nombre.replace('zona_', '')] = paso
    cumple_sep = cumplimiento_separacion(individuo.genes, base, nidos_previos,
                                         sep_alc)

    # Se conserva la métrica anterior sólo para reportarla en la interfaz.
    efecto_sep = calcular_efecto_separacion_cientifico(individuo.genes, base)

    # Cumplimiento de la densidad máxima recomendada: 1 nido/m²
    # (Best Practices IOTN 2018; NOM-162-SEMARNAT-2012).
    #
    # Hace falta porque la curva de Honarvar es PLANA por debajo de 2 nidos/m²:
    # su densidad más baja ensayada fue 2, así que no tiene nada que decir de
    # lo que pasa entre 1 y 2. Sin este término el AG puede duplicar la norma
    # del corral sin pagar nada, y medido hacía justo eso: derivaba a 2.0
    # nidos/m² mientras el llenado uniforme se mantenía en 1.0.
    #
    # Se mide como densidad y no como distancia entre pares, que es lo que
    # hacía el término de separación que se retiró. La diferencia es que esto
    # NO es degenerado cuando el corral se satura: la fracción de nidos dentro
    # de la norma es continua y la maximiza precisamente el llenado uniforme,
    # así que empuja hacia el óptimo biológico en vez de pelearse con él.
    import termico
    dens_nidos = [p['densidad_m2'] for p in r['por_nido']]
    cumple_dens = (sum(1 for d in dens_nidos
                       if d <= termico.DENSIDAD_MAX_NIDOS_M2)
                   / float(len(dens_nidos))) if dens_nidos else 1.0

    biologico = float(np.clip(
        eclosion * efecto_prof * letal * cumple_dens * cumple_sep, 0.0, 1.0))
    orden = calcular_orden_operativo(individuo, gestor, base, nidos_previos)
    fitness_final = ((biologico + _EPS_SEXO * sexo + _EPS_ORDEN * orden)
                     / (1.0 + _EPS_SEXO + _EPS_ORDEN))

    individuo.fitness = round(float(fitness_final), 6)
    individuo.indice_eclosion = round(float(eclosion), 4)
    individuo.v1 = calcular_v1(individuo, base, nidos_previos)
    individuo.v2 = round(1.0 - cumple_sep, 6)   # incumplimiento de separación
    individuo.v3 = 1.0 - efecto_prof
    individuo.orden = orden

    return individuo.fitness
