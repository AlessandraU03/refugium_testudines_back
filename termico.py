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

# --- Densidad maxima recomendada en corral. No es una distancia sino una
#     densidad, y el motivo documentado es triple: calor metabolico entre
#     nidos vecinos, disponibilidad de gases respiratorios, y espacio para que
#     el personal circule.
#       "Maintain a density of 1 nest/m2 to minimise the effects of adjacent
#        nests on temperature and respiratory gas availability, and allow
#        space for hatchery workers to move."
#       Best Practices in Sea Turtle Hatchery Management for South Asia
#       (2018), citando a Mortimer et al. (1999), Shanker et al. (2003),
#       Ahmad et al. (2004) y Maulany et al. (2012).
#     Coincide con la separacion de 1 m que la NOM-162-SEMARNAT-2012 fija para
#     golfina y que Corzo-Dominguez y Romero-Berny (2025) documentan en el
#     corral de Puerto Arista. Es el umbral contra el que el AG compara la
#     densidad local que produce cada colocacion.
DENSIDAD_MAX_NIDOS_M2 = 1.0

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

#     El efecto de la sombra DEPENDE DE LA PROFUNDIDAD, y Hill et al. lo
#     midieron a las dos profundidades que importan aqui:
#         45 cm (golfina) -> -2.2 C
#         75 cm (laud)    -> -1.3 C
#     Entre ambas se interpola; para laud a 80 cm es una extrapolacion de 5 cm.
#
#     Consistencia interna que conviene notar: en las parcelas expuestas la
#     arena marco 31.7 C a 45 cm y 30.8 C a 75 cm, pero en las SOMBREADAS
#     marco 29.5 C a las dos profundidades. Es decir, bajo sombra el gradiente
#     por profundidad se desvanece, y eso es exactamente lo que produce
#     interpolar una reduccion mayor en lo somero.
DELTA_T_SOMBRA_POR_PROF_C = [(45.0, -2.2), (75.0, -1.3)]

#     DOSIS DE RIEGO. Hill et al. regaron a diario por la tarde, en cantidades
#     iguales cada dia durante 31 dias, con tres volumenes tomados de la
#     pluviometria historica del sitio:
#
#         100 mm (ano de El Nino)   -> -1.8 C a 45 cm, -0.6 C a 75 cm
#         323 mm (ano neutro)       -> -2.3 C a 45 cm, -0.7 C a 75 cm
#         721 mm (ano de La Nina)   -> -2.4 C a 45 cm, -0.7 C a 75 cm
#
#     La curva tiene rendimientos decrecientes muy marcados: triplicar el agua
#     de 100 a 323 mm gana medio grado, y duplicarla otra vez gana una decima.
#     Por eso el riego admite una DOSIS y no un si/no: con presupuesto limitado
#     la dosis baja rinde muchisimo mas por litro, y esa es una recomendacion
#     que el sistema puede hacer con respaldo.
CURVA_RIEGO_MM = [
    (0.0,   {45.0: 0.0,  75.0: 0.0}),
    (100.0, {45.0: -1.8, 75.0: -0.6}),
    (323.0, {45.0: -2.3, 75.0: -0.7}),
    (721.0, {45.0: -2.4, 75.0: -0.7}),
]
RIEGO_DOSIS_REFERENCIA_MM = 323.0   # la que corresponde al anclaje combinado
RIEGO_DIAS = 31                     # duracion del riego en el experimento

#     ATENUACION DE REFERENCIA. SUPUESTO DECLARADO, no un dato del articulo.
#
#     Hill et al. (2015) no reportan el porcentaje de sombreo de sus parcelas:
#     solo distinguen "sombreada" de "expuesta". No existe por tanto un mapeo
#     documentado de porcentaje de malla a grados centigrados. Se asume que su
#     condicion sombreada equivale a una malla que intercepta el 80 % de la
#     radiacion directa, y el enfriamiento escala linealmente entre 0 y ese
#     anclaje. Un nido bajo una malla del 80 % todo el dia recibe exactamente
#     la reduccion medida por Hill; medio dia bajo ella, la mitad.
#
#     Cambiar este valor reescala todo el efecto de la sombra, asi que es el
#     primer parametro que hay que someter a analisis de sensibilidad.
ATENUACION_REFERENCIA_HILL = 0.80

# --- Sandoval, S., Gomez-Munoz, V. M. y Porta-Gandara, M. A. (2020). Nuevo
#     metodo para simplificar la estimacion de la proporcion sexual en crias
#     de tortuga marina utilizando datos de temperatura en corrales de
#     incubacion. Investigacion y Ciencia, 28(80), 14-21.
#
#     L. olivacea en corrales de incubacion, Playa Ceuta, Sinaloa (Pacifico
#     mexicano). Ecuacion de Girondot (1999).
PIVOTE_C    = 29.95
S_GIRONDOT  = -0.63

# --- Situacion del corral.
#     Dato de campo: el corral de Puerto Arista tiene malla sombra. No se
#     conoce el porcentaje de cobertura (el vivero de referencia en Guatemala
#     usaba 50-80 %; Hill et al. midieron sombra completa). Las predicciones
#     del "estado actual" usan este valor; el escenario sin sombra queda como
#     contrafactual.
CORRAL_CON_MALLA_SOMBRA = True

# --- Temperatura base de la arena por mes.
#     Sustituye una tabla que venia en api.py atribuida a "Sandoval 2020 /
#     de la Torre 2017", que no aparecia en ninguna de las dos fuentes y que
#     tenia la forma estacional de un clima templado (de 25.0 C en diciembre a
#     33.2 C en julio, 8.2 C de oscilacion), no la de la costa de Chiapas.
#
#     1) Forma: normales climatologicas del SMN, temperatura media del aire,
#        promedio de las tres estaciones costeras mas cercanas a Puerto
#        Arista: Tonala 7168 (1971-2000; las dos de Tonala estan
#        suspendidas), Arriaga 7182 (1991-2020) y Pijijiapan 7129
#        (1991-2020). Oscila apenas 2.6 C en el ano, con pico en abril-mayo.
#
#     2) Nivel: Carbonell Ellgutter et al. (2025) midieron en un vivero con
#        malla sombra de la costa pacifica de Guatemala una temperatura
#        media de nido de 30.9 C en el tercio medio (sep-dic), con aire
#        regional de 27.0 C de media anual. El PTS de un corral con malla
#        queda asi 3.9 C por encima del aire.
#
#     3) Esta tabla es la temperatura SIN sombra, que es como la define el
#        resto del modelo; por eso se le suman los 2.2 C que la sombra resta
#        segun Hill et al. (2015). Con sombra=True el modelo regresa al nivel
#        medido en el vivero con malla.
#
#     Incertidumbre que debe reportarse: el aire de referencia es la media
#     anual y no la del periodo de estudio; la cobertura de la malla de
#     Puerto Arista no se conoce; y otra medicion cercana, en un corral de San
#     Juan Chacahua, Oaxaca, a la misma latitud (de la Torre-Robles et al.
#     2017, PTS 30.1 C en temporada seca, sin dato de sombra) quedaria solo
#     1.05 C sobre el aire. La proporcion sexual es muy sensible a esta
#     diferencia porque el PTS ronda la temperatura pivote. La correccion
#     definitiva es medir la arena del corral.
AIRE_MEDIO_COSTA_CHIAPAS_C = {
    1: 27.40, 2: 28.00, 3: 28.97, 4: 29.97,  5: 29.80,  6: 28.50,
    7: 28.70, 8: 28.63, 9: 28.10, 10: 28.27, 11: 28.30, 12: 27.73,
}
OFFSET_PTS_CON_MALLA_C = 30.9 - 27.0
TEMP_BASE_MES_C = {
    mes: round(aire + OFFSET_PTS_CON_MALLA_C - DELTA_T_SOMBRA_C, 2)
    for mes, aire in AIRE_MEDIO_COSTA_CHIAPAS_C.items()
}
TEMP_BASE_FUENTE = ("Aire: normales SMN Tonala 7168, Arriaga 7182 y Pijijiapan 7129. "
                    "Nivel: vivero con malla sombra, Carbonell Ellgutter et al. (2025). "
                    "Aproximado: confirmar midiendo la arena del corral.")

# --- Gradiente termico por profundidad de siembra.
#     Hill et al. (2015), PLOS ONE 10(6):e0129528, Tablas 1 y 2: en las
#     parcelas de control (sin sombra ni riego) de Playa Grande, Costa Rica,
#     la arena marco 31.7 +- 0.3 C a 45 cm y 30.8 +- 0.2 C a 75 cm. Son
#     0.9 C en 30 cm: 0.03 C por centimetro, mas calor cuanto mas somero.
#
#     Sustituye un 0.08 C/cm que venia en api.py sin fuente y que casi
#     triplicaba el gradiente medido.
#
#     Alcance: se midio entre 45 y 75 cm, y aqui se aplica alrededor de 45 cm
#     dentro del rango de 40 a 50 cm de la NOM-162, asi que por el lado somero
#     es una extrapolacion corta. Mientras se siembre a la profundidad
#     documentada la correccion no pasa de 0.15 C.
DELTA_T_POR_CM_PROFUNDIDAD = 0.03
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
    t += _delta_mitigacion(sombra, riego, prof_cm)
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


_ATENUACION_SITIO = None


def atenuacion_malla_del_sitio():
    """Opacidad de la malla instalada, leida de csv/sitio.csv una sola vez.

    Se resuelve aqui, como valor por omision, en vez de arrastrarla por la
    firma de cada funcion del modulo. Si no hay archivo de sitio se devuelve la
    atenuacion de referencia, con lo que `sombra=True` sigue equivaliendo al
    efecto pleno que midio Hill.
    """
    global _ATENUACION_SITIO
    if _ATENUACION_SITIO is None:
        try:
            import solar
            _ATENUACION_SITIO = float(
                solar.cargar_sitio().get('atenuacion_malla',
                                         ATENUACION_REFERENCIA_HILL))
        except Exception:
            _ATENUACION_SITIO = ATENUACION_REFERENCIA_HILL
    return _ATENUACION_SITIO


def _fraccion_sombra(sombra):
    """Normaliza el argumento `sombra` a una fraccion en [0, 1].

    Acepta un booleano, por compatibilidad con las llamadas que solo distinguen
    corral con malla de corral sin malla, o una fraccion continua, que es lo
    que entrega solar.atenuacion_efectiva() para un nido concreto. Asi la
    sombra dejo de ser un interruptor y paso a ser una magnitud que depende de
    DONDE esta el nido, sin romper a quien ya llamaba con True o False.
    """
    if sombra is None or sombra is False:
        return 0.0
    if sombra is True:
        return 1.0
    return max(0.0, min(1.0, float(sombra)))


def dosis_riego_mm(riego):
    """Normaliza el argumento `riego` a milimetros de agua.

    Acepta un booleano -por compatibilidad con las llamadas que solo distinguen
    regado de no regado, donde True significa la dosis media de Hill- o un
    numero de milimetros, que es lo que permite recomendar CUANTO regar.
    """
    if riego is None or riego is False:
        return 0.0
    if riego is True:
        return RIEGO_DOSIS_REFERENCIA_MM
    return max(0.0, float(riego))


_CACHE_RIEGO = {}


def _riego_por_dosis(mm, prof_cm):
    """Enfriamiento del riego a esa dosis y profundidad, interpolando la curva.

    Memorizado: la curva es una constante del modulo y las profundidades caen
    en una rejilla de un decimal, asi que el resultado se repite. Sin cache se
    reinterpolaba 1.3 millones de veces por corrida.
    """
    clave = (round(float(mm), 1), None if prof_cm is None else round(float(prof_cm), 1))
    if clave in _CACHE_RIEGO:
        return _CACHE_RIEGO[clave]
    _CACHE_RIEGO[clave] = _r = _riego_por_dosis_calc(mm, prof_cm)
    return _r


def _riego_por_dosis_calc(mm, prof_cm):
    p = 45.0 if prof_cm is None else float(prof_cm)

    def a_prof(tabla):
        # Entre las dos profundidades medidas, 45 y 75 cm, se interpola.
        d45, d75 = tabla[45.0], tabla[75.0]
        if p <= 45.0:
            return d45
        if p >= 75.0:
            return d75
        return d45 + (p - 45.0) / 30.0 * (d75 - d45)

    puntos = [(mm_i, a_prof(t)) for mm_i, t in CURVA_RIEGO_MM]
    if mm <= puntos[0][0]:
        return puntos[0][1]
    if mm >= puntos[-1][0]:
        return puntos[-1][1]
    for (m0, d0), (m1, d1) in zip(puntos, puntos[1:]):
        if m0 <= mm <= m1:
            return d0 + (mm - m0) / (m1 - m0) * (d1 - d0)
    return puntos[-1][1]


def _sombra_plena_por_prof(prof_cm):
    """Reduccion que da la sombra plena a esa profundidad, interpolando Hill."""
    if prof_cm is None:
        return DELTA_T_SOMBRA_C
    p = float(prof_cm)
    (p0, d0), (p1, d1) = DELTA_T_SOMBRA_POR_PROF_C
    if p <= p0:
        return d0
    if p >= p1:
        # Extrapolacion corta hacia lo profundo, con la misma pendiente.
        return d1 + (p - p1) * (d1 - d0) / (p1 - p0)
    t = (p - p0) / (p1 - p0)
    return d0 + t * (d1 - d0)


def _delta_mitigacion(sombra, riego, prof_cm=None, atenuacion_malla=None):
    """Enfriamiento por sombra y riego, en grados centigrados (negativo).

    La sombra entra como fraccion continua: un nido que pasa el 40 % de la
    insolacion del dia bajo la malla recibe el 40 % del efecto medido, escalado
    ademas por lo opaca que sea la malla frente a la de referencia de Hill.

    El riego sigue siendo binario porque asi se midio: parcelas regadas contra
    no regadas. Sombra y riego juntos NO se suman (2.2 + 2.3 = 4.5 C, pero
    Hill midio 4.0 C al combinarlos), asi que se interpola desde riego solo
    hacia el anclaje combinado conforme aumenta la sombra.
    """
    f = _fraccion_sombra(sombra)
    if atenuacion_malla is None:
        atenuacion_malla = atenuacion_malla_del_sitio()
    if atenuacion_malla:
        f *= float(atenuacion_malla) / ATENUACION_REFERENCIA_HILL
    f = max(0.0, min(1.0, f))

    sombra_plena = _sombra_plena_por_prof(prof_cm)
    dosis = dosis_riego_mm(riego)

    if dosis <= 0:
        return f * sombra_plena

    riego_solo = _riego_por_dosis(dosis, prof_cm)

    # Sombra y riego juntos NO suman sus efectos: Hill midio -2.2 y -2.3 por
    # separado pero -4.0 al combinarlos, o sea 1.7 C de aporte extra de la
    # sombra sobre la arena ya regada. Ese aporte se midio con la dosis media,
    # y aqui se SUPONE igual para las demas dosis, que es la unica manera de
    # extender la interaccion sin inventar una superficie de respuesta.
    aporte_sombra = DELTA_T_SOMBRA_RIEGO_C - DELTA_T_RIEGO_C
    return riego_solo + f * aporte_sombra


# =====================================================================
#  SEXO Y SUPERVIVENCIA
# =====================================================================

def proporcion_sexual(t_pts_c, pivote=None, s=None):
    """Ecuacion de Girondot.

        Pm = 1 / (1 + exp((P - T) / S))

    P y S son ESPECIFICOS DE CADA ESPECIE y se leen de pts_termosensible.csv:

        golfina  P = 29.95 C   Sandoval, Gomez-Munoz y Porta-Gandara,
                               Lepidochelys olivacea, Playa Ceuta, Sinaloa
        prieta   P = 29.2  C   Godfrey y Mrosovsky (2006), Chelonia mydas,
                               Suriname (poblacion atlantica, importada)
        laud     P = 29.4  C   extremo inferior del rango 29.4-29.8 C
                               documentado para Dermochelys coriacea

    Antes esta funcion usaba siempre las constantes de golfina, de modo que
    prieta y laud se evaluaban con una pivote que no es la suya. Una diferencia
    de 0.75 C en P, como la que hay entre golfina y prieta, mueve la proporcion
    de hembras mas de veinte puntos en la zona empinada de la curva.

    Los valores por omision son los de golfina, por compatibilidad.
    """
    p = PIVOTE_C if pivote is None else float(pivote)
    sp = S_GIRONDOT if s is None else float(s)
    try:
        pm = 1.0 / (1.0 + math.exp((p - float(t_pts_c)) / sp))
    except OverflowError:
        pm = 0.0 if float(t_pts_c) > p else 1.0
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
                  delta_por_densidad=DELTA_T_POR_NIDO_M2, prof_cm=None,
                  pivote=None, s_girondot=None):
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

    sexo = proporcion_sexual(t_pts, pivote, s_girondot)
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
                       delta_por_densidad=DELTA_T_POR_NIDO_M2,
                       sitio=None, dia_del_anio=None, params_especie=None,
                       detalle=True):
    """Version vectorizada. Ver _evaluar_colocacion_lento para la original.

    Hace con numpy, sobre todos los nidos a la vez, lo que antes era un bucle
    de Python nido por nido. La funcion de aptitud llama a esto unas 5 000
    veces por corrida y cada llamada recorria 170 nidos: 657 000 pasadas por
    toda la cadena termica. Esa era la mitad del tiempo de una corrida.

    `detalle=False` evita construir los diccionarios por nido, que son 860 000
    por corrida y solo hacen falta cuando alguien va a leerlos uno por uno.
    Los arreglos quedan siempre en la clave 'arrays'.

    Los redondeos se replican paso a paso -no por prolijidad, sino porque el
    original redondea la temperatura a dos decimales y las crias a uno ANTES
    de sumarlas, de modo que omitirlos cambiaria los totales.
    """
    import numpy as np

    if not nidos:
        return {'indice_eclosion': 1.0, 'resumen': predecir_conjunto([]),
                'por_nido': [], 'arrays': {}}

    es_dict = isinstance(nidos[0], dict)

    def col(nombre, alterno):
        if es_dict:
            return [n.get(nombre, alterno) for n in nidos]
        return [getattr(n, nombre, alterno) for n in nidos]

    x = np.array([float(v) for v in col('x', 0.0)])
    y = np.array([float(v) for v in col('y', 0.0)])
    especies = col('especie', 'golfina')
    huevos_def = huevos_por_defecto or HUEVOS_REFERENCIA
    huevos = np.array([float(h) if h else float(huevos_def)
                       for h in col('num_huevas', 0)])
    tasa = np.array([float(tasa_base_por_especie.get(e, 0.75)) for e in especies])

    prof_raw = col('prof', None)
    prof_valida = np.array([p is not None for p in prof_raw])
    prof = np.array([float(p) if p is not None else PROFUNDIDAD_REFERENCIA_CM
                     for p in prof_raw])

    pe = params_especie or {}
    pivote = np.array([float((pe.get(e) or {}).get('pivote') or PIVOTE_C)
                       for e in especies])
    s_gir = np.array([float((pe.get(e) or {}).get('s') or S_GIRONDOT)
                      for e in especies])

    dens = np.array(densidades_locales(list(zip(x, y))))

    # --- sombra y riego, nido por nido -----------------------------------
    if sitio is not None and dia_del_anio is not None:
        import solar
        sombra = np.array([solar.fraccion_sombra_dia(xi, yi, dia_del_anio, sitio)
                           for xi, yi in zip(x, y)])
    else:
        sombra = np.array([1.0 if en_sombra(xi, yi, rectangulos_sombra) else 0.0
                           for xi, yi in zip(x, y)])

    if sitio and sitio.get('riego_activo'):
        r = sitio['riego']
        riego = ((x >= r['xmin']) & (x <= r['xmax']) &
                 (y >= r['ymin']) & (y <= r['ymax']))
    else:
        riego = np.zeros(len(nidos), dtype=bool)

    # --- mitigacion: sombra y riego combinados ---------------------------
    f = np.clip(sombra * (atenuacion_malla_del_sitio() / ATENUACION_REFERENCIA_HILL),
                0.0, 1.0)
    # sombra plena por profundidad: -2.2 C a 45 cm, con pendiente 0.03 C/cm
    # hacia lo profundo; equivalente a interpolar y extrapolar la tabla.
    sombra_plena = DELTA_T_SOMBRA_C + np.maximum(0.0, prof - 45.0) * 0.03

    dosis = RIEGO_DOSIS_REFERENCIA_MM
    nodos_mm = [m for m, _ in CURVA_RIEGO_MM]
    nodos_val = []
    for _m, tabla in CURVA_RIEGO_MM:
        d45, d75 = tabla[45.0], tabla[75.0]
        nodos_val.append(np.where(prof <= 45.0, d45,
                         np.where(prof >= 75.0, d75,
                                  d45 + (prof - 45.0) / 30.0 * (d75 - d45))))
    riego_solo = np.zeros_like(prof)
    for k in range(len(nodos_mm) - 1):
        if nodos_mm[k] <= dosis <= nodos_mm[k + 1]:
            t = (dosis - nodos_mm[k]) / (nodos_mm[k + 1] - nodos_mm[k])
            riego_solo = nodos_val[k] + t * (nodos_val[k + 1] - nodos_val[k])
            break

    aporte_sombra = DELTA_T_SOMBRA_RIEGO_C - DELTA_T_RIEGO_C
    delta = np.where(riego, riego_solo + f * aporte_sombra, f * sombra_plena)

    # --- temperaturas ----------------------------------------------------
    t_base = float(temperatura_base(mes or 9))
    t_pts = t_base + DELTA_T_POR_HUEVO_C * (huevos - HUEVOS_REFERENCIA)
    t_pts = t_pts + np.where(prof_valida,
                             (PROFUNDIDAD_REFERENCIA_CM - prof) * DELTA_T_POR_CM_PROFUNDIDAD,
                             0.0)
    t_pts = np.round(t_pts + delta, 2)

    exceso = np.maximum(0.0, dens - DENSIDAD_REFERENCIA_M2)
    t_fin = np.round(t_pts + SALTO_METABOLICO_FINAL_C + delta_por_densidad * exceso, 2)

    # --- eclosion y sexo --------------------------------------------------
    xs_c = [d for d, _ in CURVA_ECLOSION_DENSIDAD]
    ys_c = [e for _, e in CURVA_ECLOSION_DENSIDAD]
    base_curva = ys_c[0]
    absoluto = np.interp(dens, xs_c, ys_c, left=ys_c[0], right=ys_c[-1])
    factor = absoluto / base_curva

    # El original redondea la tasa SOLO para reportarla: las crias se calculan
    # con la tasa exacta, y las crias por sexo con las crias exactas. Replicar
    # ese orden importa, porque redondear antes cambia los totales.
    tasa_exacta = np.clip(tasa * factor, 0.0, 1.0)
    tasa_i = np.round(tasa_exacta, 3)
    crias_exactas = huevos * tasa_exacta
    crias = np.round(crias_exactas, 1)

    with np.errstate(over='ignore'):
        pm = 1.0 / (1.0 + np.exp((pivote - t_pts) / s_gir))
    pm = np.clip(np.nan_to_num(pm, nan=0.0), 0.0, 1.0)
    pct_machos = np.round(pm * 100.0, 1)
    pct_hembras = np.round((1.0 - pm) * 100.0, 1)

    crias_h = np.round(crias_exactas * pct_hembras / 100.0, 1)
    crias_m = np.round(crias_exactas * pct_machos / 100.0, 1)
    margen = np.round(LIMITE_LETAL_C - t_fin, 2)
    en_riesgo = t_fin >= LIMITE_LETAL_C

    # --- agregados --------------------------------------------------------
    total_crias = float(crias.sum())
    resumen = {
        'n_nidos': len(nidos),
        'crias_esperadas': round(total_crias, 1),
        'crias_hembras': round(float(crias_h.sum()), 1),
        'crias_machos': round(float(crias_m.sum()), 1),
        'pct_hembras': round(100.0 * float(crias_h.sum()) / total_crias, 1) if total_crias else 0.0,
        'pct_machos': round(100.0 * float(crias_m.sum()) / total_crias, 1) if total_crias else 0.0,
        'nidos_en_riesgo_termico': int(en_riesgo.sum()),
        'temp_pts_media_c': round(float(t_pts.mean()), 2),
    }
    crias_sin_hacinamiento = float((huevos * tasa).sum())
    indice = (resumen['crias_esperadas'] / crias_sin_hacinamiento
              if crias_sin_hacinamiento > 0 else 1.0)

    arrays = {
        'densidad_m2': dens, 'sombra': sombra, 'riego': riego,
        'temp_pts_c': t_pts, 'temp_final_c': t_fin,
        'pct_hembras': pct_hembras, 'pct_machos': pct_machos,
        'margen_c': margen, 'en_riesgo': en_riesgo,
        'crias': crias, 'tasa_eclosion': tasa_i, 'factor_densidad': factor,
    }

    por_nido = []
    if detalle:
        for i in range(len(nidos)):
            por_nido.append({
                'densidad_m2':     round(float(dens[i]), 2),
                'temp_base_c':     round(t_base, 2),
                'factor_densidad': round(float(factor[i]), 3),
                'tasa_eclosion':   float(tasa_i[i]),
                'crias_esperadas': float(crias[i]),
                'crias_hembras':   float(crias_h[i]),
                'crias_machos':    float(crias_m[i]),
                'sombra':          float(sombra[i]),
                'riego':           bool(riego[i]),
                'proporcion_sexual': {
                    'temp_pts_c':  float(t_pts[i]),
                    'pct_machos':  float(pct_machos[i]),
                    'pct_hembras': float(pct_hembras[i]),
                },
                'riesgo_termico': {
                    'temp_ultimo_tercio_c': float(t_fin[i]),
                    'limite_letal_c':       LIMITE_LETAL_C,
                    'margen_c':             float(margen[i]),
                    'en_riesgo':            bool(en_riesgo[i]),
                },
            })

    return {
        'indice_eclosion': round(min(1.0, max(0.0, indice)), 6),
        'crias_sin_hacinamiento': round(crias_sin_hacinamiento, 1),
        'resumen': resumen,
        'por_nido': por_nido,
        'arrays': arrays,
    }


def _evaluar_colocacion_lento(nidos, tasa_base_por_especie, mes=None,
                              rectangulos_sombra=None, huevos_por_defecto=None,
                              delta_por_densidad=DELTA_T_POR_NIDO_M2,
                              sitio=None, dia_del_anio=None, params_especie=None):
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
        d       = dens_todas[i]

        # Sombra del nido. Con configuracion de sitio se calcula por geometria
        # solar: cuanta de la insolacion del dia le intercepta la malla, dada
        # su posicion, la orientacion del corral, la altura de la malla, la
        # fecha y la latitud. Sin ella se cae al criterio binario de
        # rectangulos, que es lo que habia antes.
        if sitio is not None and dia_del_anio is not None:
            import solar
            sombra = solar.fraccion_sombra_dia(x, y, dia_del_anio, sitio)
        else:
            sombra = en_sombra(x, y, rectangulos_sombra)

        # Riego: segunda intervencion documentada. No depende del sol, solo de
        # si ese punto de la arena se riega. Sombra y riego juntos NO suman sus
        # efectos (2.2 + 2.3 = 4.5 C, pero Hill et al. midieron 4.0 al
        # combinarlos); esa interaccion la resuelve _delta_mitigacion.
        riego = bool(sitio and sitio.get('riego_activo')
                     and en_sombra(x, y, [sitio['riego']]))

        # Profundidad y pivote propias de la especie. Antes no se pasaba la
        # profundidad del nido -se evaluaba todo como si estuviera a 45 cm- y
        # la pivote era siempre la de golfina.
        prof = _attr(n, 'prof', None)
        pe = (params_especie or {}).get(especie, {})

        p = predecir_nido(mes=mes or 9, n_huevos=huevos,
                          tasa_base_especie=tasa, densidad_m2=d,
                          sombra=sombra, riego=riego, prof_cm=prof,
                          pivote=pe.get('pivote'), s_girondot=pe.get('s'),
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
