
import numpy as np
from datetime import datetime, timedelta

# Extremos absolutos leídos del CSV de tasa_eclosion:
#   mínimo: laúd peores condiciones = 0.40
#   máximo: golfina condiciones perfectas = 0.90
_TASA_MIN_ABS = 0.40
_TASA_MAX_ABS = 0.90
_RANGO_ABS    = _TASA_MAX_ABS - _TASA_MIN_ABS   # 0.50


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


# ── V2 (Antes V3) — Violaciones de Separación Mínima ───────────────────────────

def calcular_v2(individuo, base, nidos_previos=None):
   
    genes = individuo.genes
    N     = len(genes)
    if N == 0:
        return 0.0

    if isinstance(nidos_previos, dict) and "__cache__" in nidos_previos:
        por_esp_prev = nidos_previos
    else:
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
            xs_p = p_data[0] if isinstance(p_data[0], np.ndarray) else np.array(p_data[0])
            ys_p = p_data[1] if isinstance(p_data[1], np.ndarray) else np.array(p_data[1])
            n_p  = len(xs_p)
            dx   = xs_n[:, None] - xs_p[None, :]
            dy   = ys_n[:, None] - ys_p[None, :]
            dist_sq = dx**2 + dy**2
            pares_total += n_new * n_p
            pares_malos += int((dist_sq < sep**2).sum())

    return round(pares_malos / pares_total, 6) if pares_total > 0 else 0.0


# ── V3 — Desviación de Profundidad ────────────────────────────────────────────

def calcular_v3(individuo, base):
   
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

def calcular_fitness(individuo, gestor, base, corral, nidos_previos=None, modo='tradicional'):

    if individuo.num_nidos() == 0:
        individuo.fitness = 0.0
        individuo.v1 = individuo.v2 = individuo.v3 = 0.0
        return 0.0

    if modo == 'cientifico':
        return calcular_fitness_cientifico(individuo, gestor, base, corral, nidos_previos)

    # Modo tradicional (default): usa pesos arbitrarios
    v1_raw = calcular_v1(individuo, base, nidos_previos)
    v2     = calcular_v2(individuo, base, nidos_previos)
    v3     = calcular_v3(individuo, base)

    v1_norm = _norm_v1(v1_raw)
    fitness = v1_norm - (v2 + v3) / 2.0

    individuo.fitness = round(float(fitness), 6)
    individuo.v1      = v1_raw
    individuo.v2      = v2
    individuo.v3      = v3
    return individuo.fitness


def evaluar_poblacion(poblacion, gestor, base, corral, nidos_previos=None, modo='tradicional'):
    for ind in poblacion:
        calcular_fitness(ind, gestor, base, corral, nidos_previos, modo=modo)
    return poblacion


# ── FUNCIONES CIENTÍFICAS (Coeficientes de Literatura) ──────────────────────────
# Basadas en compilación de papers: Lepidochelys olivacea eclosión factors
# Fuentes: Springer Nature, MTN, ResearchGate, SciELO
# Ver: scratchpad/tortuga_datos_cuantitativos.md

def calcular_efecto_profundidad_cientifico(profundidad_cm, prof_opt_cm):
    """
    Penalidad por desviación de profundidad óptima.
    Basado en: ±5cm = -5% éxito, ±10cm = -15% éxito

    Args:
        profundidad_cm: profundidad del nido en cm
        prof_opt_cm: profundidad óptima de la especie (profundidad_siembra.csv)

    Returns:
        float [0.0, 1.0]: factor de penalidad (1.0 = sin penalidad, 0.0 = máxima)
    """
    desviacion = abs(profundidad_cm - prof_opt_cm)

    # Modelo lineal: cada cm de desviación = ~1% reducción
    # ±5cm → 5% reducción, ±10cm → 10-15% reducción
    # Usar escala conservadora: -0.01 por cm
    penalidad = min(desviacion * 0.01, 0.50)  # Cap a 50% máximo
    return max(0.0, 1.0 - penalidad)


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

    por_esp = {}
    for g in genes:
        por_esp.setdefault(g.especie, []).append(g)

    previos = _previos_lista(nidos_previos)
    en_orden = 0

    for esp, gs in por_esp.items():
        slots = gestor.slots_ordenados(f"zona_{esp}", base[esp]['sep_min'], previos)
        prefijo = {(round(x, 1), round(y, 1)) for x, y in slots[:len(gs)]}
        en_orden += sum(1 for g in gs
                        if (round(g.x, 1), round(g.y, 1)) in prefijo)

    return en_orden / len(genes)


# Peso del término operativo. NO es un coeficiente biológico: sólo hace que el
# orden de llenado desempate entre distribuciones biológicamente equivalentes.
# Al ser 0.01, ninguna ganancia de orden puede compensar una diferencia
# biológica mayor a ese margen (escalarización lexicográfica).
_EPS_ORDEN = 0.01


def calcular_fitness_cientifico(individuo, gestor, base, corral, nidos_previos=None):
    """
    Fitness basado en parámetros documentados, más un desempate operativo.

        biológico = V1_norm × (efecto_prof + efecto_sep) / 2
        fitness   = (biológico + _EPS_ORDEN × orden) / (1 + _EPS_ORDEN)

    Componentes:
      - V1: tasa de eclosión estimada (tasa_eclosion.csv)
      - efecto_prof: desviación respecto a la profundidad óptima de la especie
        (profundidad_siembra.csv, NOM-162-SEMARNAT-2012)
      - efecto_sep: separación mínima por especie (separacion_minima.csv)
      - orden: llenado secuencial, criterio operativo, no biológico

    Advertencia sobre el alcance real: con las dimensiones documentadas del
    corral y la rejilla basada en sep_min, `efecto_sep` vale 1.0 en la práctica
    (la rejilla ya garantiza la separación). El único factor que varía es
    `efecto_prof`. No presentar esto como una función multi-factor sin aclararlo.

    Returns:
        float [0.0, 1.0]
    """
    if individuo.num_nidos() == 0:
        individuo.fitness = 0.0
        individuo.v1 = individuo.v2 = individuo.v3 = 0.0
        return 0.0

    # V1: tasa de eclosión (sin cambios)
    v1_raw = calcular_v1(individuo, base, nidos_previos)
    v1_norm = _norm_v1(v1_raw)

    # Efecto profundidad: promedio de factor por cada nido
    efecto_prof = np.mean([
        calcular_efecto_profundidad_cientifico(g.prof, base[g.especie]['prof_opt'])
        for g in individuo.genes
    ]) if individuo.genes else 1.0

    # Efecto separación: separación mínima documentada por especie
    efecto_sep = calcular_efecto_separacion_cientifico(individuo.genes, base)

    # Combinar factores: V1 penalizada por profundidad y separación
    factor_combinado = (efecto_prof + efecto_sep) / 2.0
    biologico = float(np.clip(v1_norm * factor_combinado, 0.0, 1.0))

    # El orden de llenado sólo desempata; nunca desplaza al criterio biológico
    orden = calcular_orden_operativo(individuo, gestor, base, nidos_previos)
    fitness_final = (biologico + _EPS_ORDEN * orden) / (1.0 + _EPS_ORDEN)

    individuo.fitness = round(float(fitness_final), 6)
    individuo.v1 = v1_raw
    individuo.v2 = 1.0 - efecto_sep  # Inversión: v2 original era violaciones
    individuo.v3 = 1.0 - efecto_prof
    individuo.orden = orden

    return individuo.fitness
