# -*- coding: utf-8 -*-
"""
Modelo termico del corral de incubacion.

Cierra el circuito que hasta ahora estaba abierto en el sistema:

    colocacion + densidad  ->  temperatura  -> +-> tasa de eclosion -> crias
                                                +-> modelo de Girondot -> sexo

Antes de este modulo la temperatura de un nido dependia solo del mes y de la
profundidad, de modo que mover un nido de lugar no cambiaba ningun resultado y
el algoritmo genetico no tenia nada real que optimizar.

--------------------------------------------------------------------------
DOS DECISIONES DE MODELADO QUE HAY QUE ENTENDER ANTES DE TOCAR ESTE ARCHIVO
--------------------------------------------------------------------------

1. La densidad NO entra en la temperatura del PTS.

   Honarvar et al. (2008) encontraron diferencias significativas de
   temperatura entre densidades unicamente en los dias 37, 40 y 47 de una
   incubacion de ~47 dias, es decir en el ultimo tercio. En su Figura 3a las
   tres densidades van practicamente encimadas hasta el dia 30.

   El periodo termosensible de la golfina va del dia 17 al 33. Cuando el
   hacinamiento calienta la arena, el sexo ya quedo determinado. Por eso el
   apinamiento castiga la ECLOSION, no la proporcion sexual.

   La feminizacion proviene de la temperatura base de la arena (mes,
   profundidad, exposicion al sol), y la palanca que la corrige es la SOMBRA.

2. La densidad afecta la eclosion por la curva de Honarvar, no recalculando
   desde la temperatura.

   Esa curva ya incorpora el mecanismo completo que midieron: temperatura,
   caida de O2 y acumulacion de CO2. Derivar la eclosion otra vez a partir de
   la temperatura contaria dos veces el mismo efecto.

--------------------------------------------------------------------------
"""

import math

# =====================================================================
#  COEFICIENTES DOCUMENTADOS
# =====================================================================

# --- Honarvar, S., O'Connor, M. P. y Spotila, J. R. (2008). Density-dependent
#     effects on hatching success of the olive ridley turtle, Lepidochelys
#     olivacea. Oecologia, 157(2), 221-230.
#
#     Diseno: parcelas de 1 m x 1 m, cinco bloques, 70 huevos por nidada,
#     Playa Nancite, Costa Rica. Cuatro tratamientos: control (0 nidos),
#     baja (2), moderada (5) y alta (9).
CURVA_ECLOSION_DENSIDAD = [
    (2.0, 0.716),   # densidad baja
    (5.0, 0.559),   # densidad moderada
    (9.0, 0.295),   # densidad alta
]
DENSIDAD_REFERENCIA_M2 = 2.0   # densidad mas baja ensayada; sirve de base 1.0

#     Temperatura en playa natural: zona de baja densidad 32.7 C contra zona
#     de alta densidad 35.3 C. Los autores no precisan la densidad numerica de
#     esas dos zonas, asi que el gradiente por nido/m2 es DERIVADO, no citado.
#     Se deja explicito y ajustable para el analisis de sensibilidad.
DELTA_T_BAJA_ALTA_C = 2.6
DELTA_T_POR_NIDO_M2 = DELTA_T_BAJA_ALTA_C / (9.0 - 2.0)   # ~0.371 C, DERIVADO

# --- Carbonell Ellgutter, J. A., Bik, I. M., Renssen, H., Rosell, F.,
#     Hawkes, L. A. y Reinhardt, S. (2025). Temperature Conditions in
#     Artificial Sea Turtle Nests: Toward Optimized Hatchery Management.
#     Ecology and Evolution, 15(7), e71750.
#
#     Modelo GAMM sobre 22 nidos de L. olivacea en corral, Pacifico de
#     Guatemala. R2 ajustado = 0.748.
CALOR_METABOLICO_PTS_C   = 1.113   # coeficiente del tercio medio
CALOR_METABOLICO_FINAL_C = 3.607   # coeficiente del tercio final
DELTA_T_POR_HUEVO_C      = 0.018   # por huevo adicional en la nidada
LIMITE_LETAL_C           = 36.0    # tolerancia termica del embrion

#     Salto metabolico entre el tercio medio y el final. Se usa la DIFERENCIA
#     de los dos coeficientes anteriores, no el valor absoluto: son
#     coeficientes de un GAMM medidos contra el intercepto de ese modelo y no
#     pueden sumarse tal cual a una temperatura observada. Ver temperatura_pts.
SALTO_METABOLICO_FINAL_C = CALOR_METABOLICO_FINAL_C - CALOR_METABOLICO_PTS_C

#     Tamano de nidada de referencia: 91.4 huevos, medido EN EL SITIO.
#
#     Corzo-Dominguez y Romero-Berny (2025) reportan el promedio de huevos por
#     nido mes a mes para Puerto Arista durante 2022 (Fig. 4B). Ponderado por
#     el numero de nidos de cada mes (Fig. 3A) da 91.4 huevos por nido.
#     Sustituye al valor anterior de 100, que venia de Playa Nancite, Costa
#     Rica (Honarvar et al. 2008). La serie mensual completa vive en
#     csv/fecundidad_mensual.csv y es preferible usarla: la nidada varia de 77
#     huevos en junio a 93 en enero y agosto.
#
#     NOTA SOBRE UNA INCONSISTENCIA DE LA FUENTE. En la Fig. 4A el total de
#     huevos de agosto aparece como 19,722, pero los 1,015 nidos de ese mes
#     (Fig. 3A) por sus 93 huevos/nido (Fig. 4B) dan 94,395. Los otros once
#     meses cuadran dentro del 1 %. Ademas, sumando los doce meses con el
#     valor recalculado se obtienen 313,497 huevos, que coincide con el total
#     por zonas que el propio articulo reporta en la pagina 1277 (209,099 +
#     93,602 + 10,877 = 313,578) con 0.03 % de diferencia; con el 19,722
#     impreso la suma no cuadra. Se toma por tanto el valor recalculado y se
#     deja constancia aqui.
HUEVOS_REFERENCIA = 91.4

#     Efecto de la distancia al muro de concreto (referencia: 50 cm del muro).
#     El corral de Puerto Arista no es de concreto, asi que este termino queda
#     disponible pero desactivado por omision.
DELTA_T_DISTANCIA_MURO_C = {
     50: 0.000,
     90: -0.312,
    130: -0.742,
    170: -0.965,
    210: -0.723,
    250: -0.683,
    290: -0.568,
}

# --- Hill, J. E., Paladino, F. V., Spotila, J. R. y Santidrian Tomillo, P.
#     (2015). Shading and Watering as a Tool to Mitigate the Impacts of
#     Climate Change in Sea Turtle Nests. PLOS ONE, 10(6), e0129528.
#
#     Medido a 45 cm de profundidad, que es la profundidad de siembra
#     documentada para Puerto Arista.
DELTA_T_SOMBRA_C        = -2.2   # sombra sola
DELTA_T_RIEGO_C         = -2.3   # riego promedio (323 mm)
DELTA_T_SOMBRA_RIEGO_C  = -4.0   # sombra + riego

# --- Sandoval, S., Gomez-Munoz, V. M. y Porta-Gandara, M. A. (2020). Nuevo
#     metodo para simplificar la estimacion de la proporcion sexual en crias
#     de tortuga marina utilizando datos de temperatura en corrales de
#     incubacion. Investigacion y Ciencia, 28(80), 14-21.
#
#     L. olivacea en corrales de incubacion, Playa Ceuta, Sinaloa (Pacifico
#     mexicano). Ecuacion de Girondot (1999).
PIVOTE_C    = 29.95
S_GIRONDOT  = -0.63

# --- Temperatura base de la arena por mes.
#     ADVERTENCIA: esta tabla venia en api.py atribuida a "Sandoval 2020 /
#     de la Torre 2017" pero no se ha podido verificar en ninguna de las dos
#     fuentes. Es el unico insumo del modelo sin respaldo confirmado. Se
#     mantiene para no romper el comportamiento existente, pero cualquier
#     resultado que dependa fuertemente de ella debe reportarse como
#     provisional hasta sustituirla por mediciones del sitio.
TEMP_BASE_MES_C = {
    1: 25.5, 2: 26.0, 3: 27.0, 4: 28.5,  5: 30.0,  6: 31.8,
    7: 33.2, 8: 32.8, 9: 31.9, 10: 30.1, 11: 27.2, 12: 25.0,
}
TEMP_BASE_FUENTE = "PENDIENTE DE VERIFICAR - sustituir por medicion en sitio"

# --- Gradiente termico por profundidad de siembra.
#     ADVERTENCIA: 0.08 C por cada centimetro por encima de los 45 cm venia
#     en api.py SIN FUENTE. No se ha localizado respaldo documental. Se trae
#     aqui, en lugar de dejarlo disperso, para que sea visible y ajustable:
#     ponerlo en 0.0 desactiva el efecto de la profundidad.
#
#     Su peso real es pequeno mientras se siembre a la profundidad
#     documentada: el AG coloca los nidos a ~45 cm, asi que la correccion
#     queda por debajo de 0.1 C. Importa solo si alguien siembra fuera de esa
#     profundidad.
DELTA_T_POR_CM_PROFUNDIDAD = 0.08
PROFUNDIDAD_REFERENCIA_CM  = 45.0

def fecundidad_mensual(carpeta_csv=None):
    """Huevos por nido de cada mes, medidos en Puerto Arista durante 2022.

    Devuelve {mes: huevos_por_nido}. Los meses sin nidos registrados (mayo)
    quedan fuera del diccionario, para que el llamador caiga en el valor de
    referencia en vez de usar un cero.
    """
    import csv as _csv
    import os as _os
    if carpeta_csv is None:
        carpeta_csv = _os.path.join(_os.path.dirname(__file__), 'csv')
    ruta = _os.path.join(carpeta_csv, 'fecundidad_mensual.csv')
    if not _os.path.exists(ruta):
        return {}
    with open(ruta, newline='', encoding='utf-8') as f:
        filas = list(_csv.DictReader(f))
    salida = {}
    for r in filas:
        h = float(r['huevos_por_nido'])
        if h > 0:
            salida[int(r['mes'])] = h
    return salida


def huevos_del_mes(mes, carpeta_csv=None):
    """Huevos por nido del mes dado; HUEVOS_REFERENCIA si no hay dato."""
    return fecundidad_mensual(carpeta_csv).get(int(mes), HUEVOS_REFERENCIA)


# Lado de la vecindad usada para medir densidad local, en centimetros.
# Corresponde a la parcela de 1 m x 1 m de Honarvar et al. (2008).
LADO_VECINDAD_CM = 100.0


# =====================================================================
#  DENSIDAD
# =====================================================================

def densidad_local(x_cm, y_cm, nidos, lado_cm=LADO_VECINDAD_CM):
    """Nidos por metro cuadrado alrededor de (x_cm, y_cm), el propio incluido.

    Replica la parcela de 1 m x 1 m de Honarvar et al. (2008): cuenta cuantos
    nidos caen dentro del cuadrado centrado en la posicion dada.

    `nidos` es una lista de pares (x, y) en centimetros.
    """
    mitad = lado_cm / 2.0
    n = 0
    for nx, ny in nidos:
        if abs(nx - x_cm) <= mitad and abs(ny - y_cm) <= mitad:
            n += 1
    # el area de la vecindad en m2
    area_m2 = (lado_cm / 100.0) ** 2
    return n / area_m2


def densidades_locales(coords, lado_cm=LADO_VECINDAD_CM):
    """Densidad local de TODOS los nidos de una vez, en nidos/m2.

    Version por lotes de densidad_local. La version nido a nido es O(n) y
    llamarla para cada nido la vuelve O(n^2); dentro del AG eso se multiplica
    por poblacion y generaciones y el corral saturado se vuelve inviable.

    Aqui los nidos se reparten en celdas de `lado_cm`. La vecindad cuadrada de
    un nido nunca se sale de las 9 celdas que lo rodean, asi que basta revisar
    esas. El resultado es identico al de densidad_local; solo cambia el costo,
    que pasa a ser proporcional al numero de nidos.
    """
    n = len(coords)
    if n == 0:
        return []

    mitad = lado_cm / 2.0
    area_m2 = (lado_cm / 100.0) ** 2

    celdas = {}
    for i, (x, y) in enumerate(coords):
        clave = (int(x // lado_cm), int(y // lado_cm))
        celdas.setdefault(clave, []).append(i)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    densidades = []
    for i, (x, y) in enumerate(coords):
        cx, cy = int(x // lado_cm), int(y // lado_cm)
        cuenta = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in celdas.get((cx + dx, cy + dy), ()):
                    if abs(xs[j] - x) <= mitad and abs(ys[j] - y) <= mitad:
                        cuenta += 1
        densidades.append(cuenta / area_m2)
    return densidades


def factor_eclosion_por_densidad(densidad_m2):
    """Factor multiplicativo sobre la tasa de eclosion base de la especie.

    Interpola linealmente la curva de Honarvar et al. (2008) y la normaliza
    contra su densidad mas baja ensayada (2 nidos/m2), que se toma como 1.0.

    Por debajo de 2 nidos/m2 no se aplica castigo: el estudio no ensayo
    densidades menores y la norma de manejo (1 nido/m2) esta por debajo del
    rango experimental.

        2 nidos/m2 -> 1.000
        5 nidos/m2 -> 0.781
        9 nidos/m2 -> 0.412
    """
    d = float(densidad_m2)
    puntos = CURVA_ECLOSION_DENSIDAD
    base = dict(puntos)[DENSIDAD_REFERENCIA_M2]

    if d <= puntos[0][0]:
        absoluto = puntos[0][1]
    elif d >= puntos[-1][0]:
        absoluto = puntos[-1][1]
    else:
        absoluto = puntos[-1][1]
        for (d0, e0), (d1, e1) in zip(puntos, puntos[1:]):
            if d0 <= d <= d1:
                t = (d - d0) / (d1 - d0)
                absoluto = e0 + t * (e1 - e0)
                break

    return absoluto / base


# =====================================================================
#  TEMPERATURA
# =====================================================================

def temperatura_base(mes):
    """Temperatura base de la arena para el mes dado. Ver advertencia arriba."""
    return TEMP_BASE_MES_C.get(int(mes), 30.0)


def temperatura_pts(t_base_c, n_huevos=None, sombra=False, riego=False,
                    prof_cm=None):
    """Temperatura media durante el periodo termosensible.

    Los coeficientes de Carbonell Ellgutter et al. (2025) provienen de un GAMM
    y estan medidos RESPECTO AL INTERCEPTO DE SU MODELO, no como incrementos
    absolutos que puedan sumarse a cualquier temperatura. Por eso aqui se
    aplican como DESVIACIONES respecto a un nido de referencia:

      - `t_base_c` se interpreta como la temperatura del nido durante el PTS
        para una nidada de tamano de referencia, sin sombra ni riego. No se le
        suma el calor metabolico del tercio medio, porque una temperatura de
        nido observada ya lo contiene.
      - el termino por huevo se aplica solo a la diferencia contra
        HUEVOS_REFERENCIA.

    La densidad NO entra aqui a proposito: Honarvar et al. (2008) solo
    hallaron diferencias significativas por densidad en el ultimo tercio de la
    incubacion, cuando el sexo ya quedo determinado. Ver el encabezado.
    """
    n = HUEVOS_REFERENCIA if n_huevos in (None, 0) else float(n_huevos)
    t = float(t_base_c)
    t += DELTA_T_POR_HUEVO_C * (n - HUEVOS_REFERENCIA)
    if prof_cm is not None:
        t += (PROFUNDIDAD_REFERENCIA_CM - float(prof_cm)) * DELTA_T_POR_CM_PROFUNDIDAD
    t += _delta_mitigacion(sombra, riego)
    return round(t, 2)


def temperatura_ultimo_tercio(t_base_c, n_huevos=None, densidad_m2=0.0,
                              sombra=False, riego=False,
                              delta_por_densidad=DELTA_T_POR_NIDO_M2,
                              prof_cm=None):
    """Temperatura media durante el ultimo tercio, donde si pesa la densidad.

    Se construye sobre la del PTS sumando dos cosas:

      - el salto metabolico entre el tercio medio y el final, que es la
        DIFERENCIA de los dos coeficientes de Carbonell Ellgutter et al.
        (2025): 3.607 - 1.113 = 2.494 C. Se usa la diferencia y no el
        coeficiente absoluto por la misma razon explicada en temperatura_pts.
      - el efecto de la densidad, por encima de la densidad de referencia.

    `delta_por_densidad` es el parametro DERIVADO de Honarvar: se expone como
    argumento justamente para poder recorrer su rango en el analisis de
    sensibilidad.
    """
    t = temperatura_pts(t_base_c, n_huevos, sombra, riego, prof_cm)
    t += SALTO_METABOLICO_FINAL_C
    exceso = max(0.0, float(densidad_m2) - DENSIDAD_REFERENCIA_M2)
    t += delta_por_densidad * exceso
    return round(t, 2)


def _delta_mitigacion(sombra, riego):
    if sombra and riego:
        return DELTA_T_SOMBRA_RIEGO_C
    if sombra:
        return DELTA_T_SOMBRA_C
    if riego:
        return DELTA_T_RIEGO_C
    return 0.0


# =====================================================================
#  SEXO Y SUPERVIVENCIA
# =====================================================================

def proporcion_sexual(t_pts_c):
    """Ecuacion de Girondot con los parametros de Sandoval et al. (2020).

        Pm = 1 / (1 + exp((P - T) / S))     P = 29.95 C,  S = -0.63

    Devuelve el porcentaje de machos y de hembras.
    """
    try:
        pm = 1.0 / (1.0 + math.exp((PIVOTE_C - float(t_pts_c)) / S_GIRONDOT))
    except OverflowError:
        pm = 0.0 if float(t_pts_c) > PIVOTE_C else 1.0
    pm = max(0.0, min(1.0, pm))
    return {
        'temp_pts_c':  round(float(t_pts_c), 2),
        'pct_machos':  round(pm * 100.0, 1),
        'pct_hembras': round((1.0 - pm) * 100.0, 1),
    }


def riesgo_letal(t_ultimo_tercio_c):
    """Margen contra el limite de tolerancia termica del embrion (36 C)."""
    t = float(t_ultimo_tercio_c)
    return {
        'temp_ultimo_tercio_c': round(t, 2),
        'limite_letal_c':       LIMITE_LETAL_C,
        'margen_c':             round(LIMITE_LETAL_C - t, 2),
        'en_riesgo':            t >= LIMITE_LETAL_C,
    }


# =====================================================================
#  PREDICCION
# =====================================================================

def predecir_nido(mes, n_huevos, tasa_base_especie, densidad_m2=0.0,
                  sombra=False, riego=False, t_base_c=None,
                  delta_por_densidad=DELTA_T_POR_NIDO_M2, prof_cm=None):
    """Prediccion completa para un nido: crias esperadas y sexo.

    Args:
        mes: mes de siembra (1-12)
        n_huevos: tamano de la nidada
        tasa_base_especie: tasa de eclosion de la especie sin hacinamiento
                           (tasa_eclosion.csv)
        densidad_m2: nidos por metro cuadrado alrededor de este nido
        sombra, riego: intervenciones aplicadas sobre este nido
        t_base_c: temperatura base; si es None se toma del mes
    """
    t_base = temperatura_base(mes) if t_base_c is None else float(t_base_c)

    t_pts   = temperatura_pts(t_base, n_huevos, sombra, riego, prof_cm)
    t_final = temperatura_ultimo_tercio(t_base, n_huevos, densidad_m2,
                                        sombra, riego, delta_por_densidad,
                                        prof_cm)

    factor = factor_eclosion_por_densidad(densidad_m2)
    tasa   = max(0.0, min(1.0, float(tasa_base_especie) * factor))
    crias  = float(n_huevos or 0) * tasa

    sexo = proporcion_sexual(t_pts)
    riesgo = riesgo_letal(t_final)

    return {
        'densidad_m2':        round(float(densidad_m2), 2),
        'temp_base_c':        round(t_base, 2),
        'factor_densidad':    round(factor, 3),
        'tasa_eclosion':      round(tasa, 3),
        'crias_esperadas':    round(crias, 1),
        'crias_hembras':      round(crias * sexo['pct_hembras'] / 100.0, 1),
        'crias_machos':       round(crias * sexo['pct_machos']  / 100.0, 1),
        'sombra':             bool(sombra),
        'riego':              bool(riego),
        'proporcion_sexual':  sexo,
        'riesgo_termico':     riesgo,
    }


def predecir_conjunto(predicciones):
    """Agrega una lista de predicciones de nido en indicadores de cohorte."""
    if not predicciones:
        return {
            'n_nidos': 0, 'crias_esperadas': 0.0,
            'crias_hembras': 0.0, 'crias_machos': 0.0,
            'pct_hembras': 0.0, 'pct_machos': 0.0,
            'nidos_en_riesgo_termico': 0, 'temp_pts_media_c': 0.0,
        }

    crias   = sum(p['crias_esperadas'] for p in predicciones)
    hembras = sum(p['crias_hembras']   for p in predicciones)
    machos  = sum(p['crias_machos']    for p in predicciones)
    riesgo  = sum(1 for p in predicciones if p['riesgo_termico']['en_riesgo'])
    t_media = sum(p['proporcion_sexual']['temp_pts_c']
                  for p in predicciones) / len(predicciones)

    return {
        'n_nidos':                 len(predicciones),
        'crias_esperadas':         round(crias, 1),
        'crias_hembras':           round(hembras, 1),
        'crias_machos':            round(machos, 1),
        'pct_hembras':             round(100.0 * hembras / crias, 1) if crias else 0.0,
        'pct_machos':              round(100.0 * machos  / crias, 1) if crias else 0.0,
        'nidos_en_riesgo_termico': riesgo,
        'temp_pts_media_c':        round(t_media, 2),
    }


# =====================================================================
#  INTEGRACION CON EL ALGORITMO GENETICO
# =====================================================================

def en_sombra(x_cm, y_cm, rectangulos_sombra):
    """True si la posicion cae dentro de alguna region sombreada.

    `rectangulos_sombra` es una lista de dicts con xmin, ymin, xmax, ymax
    en centimetros. Una lista vacia significa corral sin sombra, que es el
    estado actual de Puerto Arista.
    """
    for r in rectangulos_sombra or ():
        if (r['xmin'] <= x_cm <= r['xmax']) and (r['ymin'] <= y_cm <= r['ymax']):
            return True
    return False


def evaluar_colocacion(nidos, tasa_base_por_especie, mes=None,
                       rectangulos_sombra=None, huevos_por_defecto=None,
                       delta_por_densidad=DELTA_T_POR_NIDO_M2):
    """Evalua una colocacion completa de nidos.

    `nidos` es una lista de objetos con atributos x, y, especie y num_huevas
    (los genes del cromosoma), o de dicts con esas mismas claves.

    Devuelve un dict con el rendimiento de la cohorte y, sobre todo, con
    `indice_eclosion`: la fraccion de crias que se conservan respecto al
    mismo conjunto de nidos sin hacinamiento. Ese indice es el que consume la
    funcion de aptitud, porque esta acotado a (0, 1] y solo depende de la
    curva medida por Honarvar, sin parametros inciertos.
    """
    if not nidos:
        return {'indice_eclosion': 1.0, 'resumen': predecir_conjunto([]),
                'por_nido': []}

    def _attr(n, nombre, alterno=None):
        if isinstance(n, dict):
            return n.get(nombre, alterno)
        return getattr(n, nombre, alterno)

    coords = [(float(_attr(n, 'x', 0.0)), float(_attr(n, 'y', 0.0)))
              for n in nidos]

    huevos_def = huevos_por_defecto or HUEVOS_REFERENCIA
    predicciones = []
    crias_sin_hacinamiento = 0.0

    dens_todas = densidades_locales(coords)
    for i, (n, (x, y)) in enumerate(zip(nidos, coords)):
        especie = _attr(n, 'especie', 'golfina')
        huevos  = _attr(n, 'num_huevas', 0) or huevos_def
        tasa    = float(tasa_base_por_especie.get(especie, 0.75))
        sombra  = en_sombra(x, y, rectangulos_sombra)
        d       = dens_todas[i]

        p = predecir_nido(mes=mes or 9, n_huevos=huevos,
                          tasa_base_especie=tasa, densidad_m2=d,
                          sombra=sombra,
                          delta_por_densidad=delta_por_densidad)
        predicciones.append(p)
        crias_sin_hacinamiento += float(huevos) * tasa

    resumen = predecir_conjunto(predicciones)
    indice = (resumen['crias_esperadas'] / crias_sin_hacinamiento
              if crias_sin_hacinamiento > 0 else 1.0)

    return {
        'indice_eclosion': round(min(1.0, max(0.0, indice)), 6),
        'crias_sin_hacinamiento': round(crias_sin_hacinamiento, 1),
        'resumen': resumen,
        'por_nido': predicciones,
    }
