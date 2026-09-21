# -*- coding: utf-8 -*-
"""
Geometria solar del corral de incubacion.

Responde una sola pregunta, nido por nido: cuanta radiacion solar directa
recibe a lo largo del dia, dado donde esta la malla sombra, que tan alta es,
como esta orientado el corral y en que fecha y latitud nos encontramos.

    posicion del nido + orientacion + malla + fecha + latitud
        -> fraccion de insolacion directa interceptada  [0, 1]

Esa fraccion es lo que convierte la COLOCACION en temperatura, y por tanto en
proporcion sexual. Sin ella la temperatura del nido solo dependia del mes y de
la profundidad: mover un nido de lugar no cambiaba su sexo, y el algoritmo
genetico no tenia nada espacial que optimizar.

--------------------------------------------------------------------------
ALCANCE Y LIMITES DEL MODELO
--------------------------------------------------------------------------

Lo que modela:
  - Trayectoria solar real para la fecha y la latitud (declinacion, ecuacion
    del tiempo, angulo horario).
  - Sombra proyectada por una malla horizontal a cierta altura, teniendo en
    cuenta que la sombra SE CORRE conforme baja el sol: un nido junto al borde
    de la malla esta sombreado al mediodia y expuesto a media tarde.
  - Orientacion del corral, que decide hacia donde se corre esa sombra.
  - Ponderacion por la altura solar, porque un rayo rasante aporta mucho menos
    energia a una superficie horizontal que uno vertical.

Lo que NO modela, y hay que decirlo:
  - Radiacion difusa. Solo se sigue el haz directo. Un nido "sombreado" sigue
    recibiendo cielo difuso, asi que la sombra real nunca es total.
  - Nubosidad. La costa de Chiapas concentra lluvias de julio a octubre, justo
    en la temporada de anidacion, de modo que la insolacion efectiva es menor
    que la de cielo despejado que se calcula aqui.
  - Inercia termica de la arena. La temperatura de un nido a 45 cm responde al
    promedio de varios dias, no a la insolacion instantanea. Por eso el
    resultado se usa como fraccion promedio del dia, no como curva horaria.
  - Sombras de arboles, muros o del propio poste de la malla.

Algoritmo de posicion solar: formulas estandar de astronomia de posicion
(declinacion y ecuacion del tiempo segun las aproximaciones de Spencer, 1971,
Fourier series representation of the position of the sun). Precision del orden
de centesimas de grado en declinacion, mas que suficiente aqui: la
incertidumbre del modelo la domina la nubosidad, no la efemeride.
"""

import math


# =====================================================================
#  POSICION SOLAR
# =====================================================================

def declinacion_grados(dia_del_anio):
    """Declinacion solar en grados para el dia juliano dado.

    Serie de Fourier de Spencer (1971). Positiva en verano boreal.
    """
    g = 2.0 * math.pi * (dia_del_anio - 1) / 365.0
    d = (0.006918
         - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
         - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
         - 0.002697 * math.cos(3 * g) + 0.001480 * math.sin(3 * g))
    return math.degrees(d)


def ecuacion_del_tiempo_min(dia_del_anio):
    """Diferencia entre el tiempo solar verdadero y el medio, en minutos."""
    g = 2.0 * math.pi * (dia_del_anio - 1) / 365.0
    e = (0.000075
         + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
         - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    return 229.18 * e


def posicion_solar(lat_grados, lon_grados, dia_del_anio, hora_local,
                   huso_horas):
    """Altura y azimut del sol.

    Args:
        hora_local: hora decimal del reloj local (13.5 = 13:30)
        huso_horas: desplazamiento del huso respecto a UTC (Chiapas: -6)

    Returns:
        (altura_grados, azimut_grados). El azimut se mide desde el norte en
        sentido del reloj: 90 es este, 180 sur, 270 oeste. La altura es
        negativa cuando el sol esta bajo el horizonte.
    """
    dec = math.radians(declinacion_grados(dia_del_anio))
    lat = math.radians(lat_grados)

    # Hora solar verdadera. La correccion por longitud vale 4 minutos por grado
    # de diferencia contra el meridiano central del huso.
    meridiano_central = 15.0 * huso_horas
    correccion_min = 4.0 * (lon_grados - meridiano_central) + \
        ecuacion_del_tiempo_min(dia_del_anio)
    hora_solar = hora_local + correccion_min / 60.0

    # Angulo horario: 0 al mediodia solar, 15 grados por hora, positivo tarde.
    h = math.radians(15.0 * (hora_solar - 12.0))

    sen_alt = (math.sin(lat) * math.sin(dec) +
               math.cos(lat) * math.cos(dec) * math.cos(h))
    sen_alt = max(-1.0, min(1.0, sen_alt))
    altura = math.asin(sen_alt)

    # Azimut por atan2, que resuelve el cuadrante sin casos especiales.
    y = -math.sin(h) * math.cos(dec)
    x = (math.sin(dec) * math.cos(lat) -
         math.cos(dec) * math.sin(lat) * math.cos(h))
    azimut = math.degrees(math.atan2(y, x)) % 360.0

    return math.degrees(altura), azimut


# =====================================================================
#  SOMBRA DE LA MALLA
# =====================================================================

def punto_bajo_malla(x_cm, y_cm, altura_sol, azimut_sol, orientacion_grados,
                     malla, altura_malla_cm):
    """True si el rayo que llega a ese nido atraviesa la malla.

    Geometria: el rayo que va del nido hacia el sol cruza el plano de la malla
    a una distancia horizontal `d = altura_malla / tan(altura_sol)`, en la
    direccion del azimut solar. Si ese punto de cruce cae dentro del rectangulo
    de la malla, el nido esta sombreado.

    Todo se resuelve en coordenadas LOCALES del corral, que es donde viven las
    posiciones de los nidos. Para eso basta rotar el azimut solar al marco del
    corral: si el eje +x del corral apunta al azimut `orientacion`, entonces un
    desplazamiento de magnitud d y azimut A se ve en coordenadas locales como
    (d*cos(A - orientacion), d*sin(A - orientacion)).
    """
    if altura_sol <= 0.5:
        # Sol en el horizonte: la sombra se alarga sin limite y el aporte
        # energetico es despreciable. Se trata como sin sol.
        return False

    d = altura_malla_cm / math.tan(math.radians(altura_sol))
    rel = math.radians(azimut_sol - orientacion_grados)

    cx = x_cm + d * math.cos(rel)
    cy = y_cm + d * math.sin(rel)

    return (malla['xmin'] <= cx <= malla['xmax'] and
            malla['ymin'] <= cy <= malla['ymax'])


def fraccion_sombra_dia(x_cm, y_cm, dia_del_anio, sitio, malla=None):
    """Fraccion de la insolacion directa del dia que la malla intercepta.

    Devuelve 0.0 para un nido a pleno sol todo el dia y 1.0 para uno que
    permanece bajo la malla de la salida a la puesta del sol. Los valores
    intermedios son los interesantes: un nido cerca del borde de la malla.

    Cada instante se pondera por el seno de la altura solar, que es la
    proyeccion del haz directo sobre una superficie horizontal. Asi las horas
    centrales del dia, que es cuando de verdad se calienta la arena, pesan
    mucho mas que las primeras y ultimas.
    """
    if malla is None:
        malla = sitio.get('malla')
    if not malla:
        return 0.0

    # --- Dos cachés, porque esto se llama millones de veces -----------------
    #
    # La funcion de aptitud evalua unas 10 000 colocaciones por corrida, cada
    # una con decenas de nidos, y cada nido recorria las 144 posiciones del sol
    # del dia: del orden de 10^8 calculos de efemeride por corrida. Medido: una
    # jornada de 30 nidos tardaba 142.7 s.
    #
    # La trayectoria del sol NO depende del nido, solo de la fecha y el sitio,
    # asi que se calcula una vez por dia y la reusan todos los nidos. Y como
    # los nidos caen en una rejilla discreta, el resultado por nido tambien se
    # memoriza.
    firma = (sitio['latitud'], sitio['longitud'], sitio['huso'],
             sitio['orientacion'], sitio['altura_malla'],
             sitio.get('paso_minutos', 10),
             malla['xmin'], malla['ymin'], malla['xmax'], malla['ymax'])

    clave_nido = (round(float(x_cm), 1), round(float(y_cm), 1),
                  int(dia_del_anio), firma)
    if clave_nido in _CACHE_NIDO:
        return _CACHE_NIDO[clave_nido]

    trayectoria = _trayectoria_dia(int(dia_del_anio), sitio)

    total = 0.0
    tapado = 0.0
    for alt, azi, peso in trayectoria:
        total += peso
        if punto_bajo_malla(x_cm, y_cm, alt, azi, sitio['orientacion'],
                            malla, sitio['altura_malla']):
            tapado += peso

    r = (tapado / total) if total > 0 else 0.0
    _CACHE_NIDO[clave_nido] = r
    return r


def fracciones_sombra_dia(xs_cm, ys_cm, dia_del_anio, sitio, malla=None):
    """Como `fraccion_sombra_dia`, pero para TODOS los nidos de una vez.

    POR QUE HACE FALTA. La version por nido recorre en Python los 144 instantes
    del dia y para cada uno resuelve una tangente y dos trigonometricas. Tiene
    memoria, pero el AG mueve los nidos por posiciones CONTINUAS, asi que casi
    ninguna consulta acierta: cada generacion pregunta por coordenadas nuevas.
    Medido en un proceso limpio, una jornada de 270 nidos tardaba 108 s, por
    encima del minuto que tiene que tardar.

    Aqui la geometria se resuelve como una matriz de nidos por instantes. Para
    270 nidos son 39 000 elementos, una operacion de numpy, en lugar de 39 000
    vueltas de interprete.

    Devuelve un arreglo con la fraccion de la insolacion del dia que la malla
    intercepta en cada nido, en el mismo orden en que llegaron.
    """
    import numpy as np

    xs = np.asarray(xs_cm, dtype=float)
    ys = np.asarray(ys_cm, dtype=float)
    if xs.size == 0:
        return np.empty(0, dtype=float)

    if malla is None:
        malla = sitio.get('malla')
    if not malla:
        return np.zeros(xs.shape, dtype=float)

    d, cos_rel, sin_rel, peso, total = _trayectoria_arreglos(
        int(dia_del_anio), sitio)
    if total <= 0 or d.size == 0:
        return np.zeros(xs.shape, dtype=float)

    # Donde cruza el plano de la malla el rayo que va de cada nido al sol, en
    # cada instante del dia. Misma geometria que punto_bajo_malla, en matriz.
    cx = xs[:, None] + d[None, :] * cos_rel[None, :]
    cy = ys[:, None] + d[None, :] * sin_rel[None, :]

    bajo = ((cx >= malla['xmin']) & (cx <= malla['xmax']) &
            (cy >= malla['ymin']) & (cy <= malla['ymax']))

    return (bajo * peso[None, :]).sum(axis=1) / total


_CACHE_NIDO = {}
_CACHE_TRAYECTORIA = {}
_CACHE_ARREGLOS = {}


def _trayectoria_arreglos(dia_del_anio, sitio):
    """La trayectoria del dia como arreglos listos para la matriz.

    Precalcula lo que no depende del nido: la distancia horizontal a la que el
    rayo cruza el plano de la malla, el coseno y el seno del azimut relativo a
    la orientacion del corral, y el peso energetico de cada instante. Depende
    solo de la fecha y del sitio, asi que se memoriza.

    Se descartan los instantes con el sol por debajo de 0.5 grados sobre el
    horizonte, igual que hace punto_bajo_malla: ahi la tangente se va a cero y
    la sombra proyectada seria de kilometros.
    """
    import numpy as np

    clave = (int(dia_del_anio), sitio['latitud'], sitio['longitud'],
             sitio['huso'], sitio.get('paso_minutos', 10),
             sitio['orientacion'], sitio['altura_malla'])
    if clave in _CACHE_ARREGLOS:
        return _CACHE_ARREGLOS[clave]

    tray = _trayectoria_dia(int(dia_del_anio), sitio)
    if not tray:
        vacio = (np.empty(0), np.empty(0), np.empty(0), np.empty(0), 0.0)
        _CACHE_ARREGLOS[clave] = vacio
        return vacio

    alt = np.array([p[0] for p in tray], dtype=float)
    azi = np.array([p[1] for p in tray], dtype=float)
    pes = np.array([p[2] for p in tray], dtype=float)

    # El total se calcula sobre TODOS los instantes con sol, tambien los que se
    # descartan por altura: son parte de la insolacion del dia, y quitarlos del
    # denominador inflaria la fraccion de sombra.
    total = float(pes.sum())

    util = alt > 0.5
    alt, azi, pes = alt[util], azi[util], pes[util]

    d = float(sitio['altura_malla']) / np.tan(np.radians(alt))
    rel = np.radians(azi - float(sitio['orientacion']))

    res = (d, np.cos(rel), np.sin(rel), pes, total)
    _CACHE_ARREGLOS[clave] = res
    return res


def _trayectoria_dia(dia_del_anio, sitio):
    """Posiciones del sol sobre el horizonte ese dia, con su peso energetico.

    Devuelve [(altura, azimut, peso)], donde el peso es el seno de la altura:
    la proyeccion del haz directo sobre una superficie horizontal. No depende
    de ningun nido, asi que se calcula una vez por dia y sitio.
    """
    clave = (dia_del_anio, sitio['latitud'], sitio['longitud'],
             sitio['huso'], sitio.get('paso_minutos', 10))
    if clave in _CACHE_TRAYECTORIA:
        return _CACHE_TRAYECTORIA[clave]

    paso = float(sitio.get('paso_minutos', 10))
    puntos = []
    minutos = 0.0
    while minutos < 24 * 60:
        alt, azi = posicion_solar(sitio['latitud'], sitio['longitud'],
                                  dia_del_anio, minutos / 60.0, sitio['huso'])
        if alt > 0.0:
            puntos.append((alt, azi, math.sin(math.radians(alt))))
        minutos += paso

    _CACHE_TRAYECTORIA[clave] = puntos
    return puntos


def atenuacion_efectiva(x_cm, y_cm, dia_del_anio, sitio, malla=None):
    """Atenuacion real de la radiacion directa sobre ese nido, en [0, 1].

    Combina cuanto tiempo el nido pasa bajo la malla con lo opaca que la malla
    es. Una malla de sombreo del 80 por ciento que solo cubre al nido la mitad
    de la jornada atenua 0.40, no 0.80.

    Este es el numero que consume el modelo termico para escalar el
    enfriamiento medido por Hill et al. (2015).
    """
    f = fraccion_sombra_dia(x_cm, y_cm, dia_del_anio, sitio, malla)
    return f * float(sitio.get('atenuacion_malla', 0.0))


# =====================================================================
#  CARGA DE LA CONFIGURACION DEL SITIO
# =====================================================================

def cargar_sitio(carpeta_csv=None):
    """Lee csv/sitio.csv y devuelve la configuracion lista para usar.

    Devuelve tambien `supuestos`: la lista de campos que NO provienen de una
    medicion. La interfaz los marca y el protocolo los declara. Un supuesto
    declarado se puede defender; un numero inventado que se presento como
    medicion, no.
    """
    import os
    from cromosoma import cargar_csv

    if carpeta_csv is None:
        carpeta_csv = os.path.join(os.path.dirname(__file__), 'csv')
    filas = cargar_csv(os.path.join(carpeta_csv, 'sitio.csv'))

    v = {f['campo']: f['valor'] for f in filas}
    supuestos = [{'campo': f['campo'], 'valor': f['valor'],
                  'fuente': f.get('fuente', '')}
                 for f in filas if f.get('supuesto') == 'si']

    return {
        'latitud':          float(v['latitud_grados']),
        'longitud':         float(v['longitud_grados']),
        'huso':             float(v['huso_horario_horas']),
        'orientacion':      float(v['orientacion_grados']),
        'altura_malla':     float(v['malla_altura_cm']),
        'atenuacion_malla': float(v['malla_atenuacion']),
        'paso_minutos':     float(v['paso_minutos']),
        # Corral sobre el que se midieron los rectangulos de abajo. Permite
        # reescalarlos si se evalua un corral de otro tamano (ajustar_a_corral).
        'corral_ref_largo_cm': float(v.get('corral_ref_largo_cm', 0) or 0),
        'corral_ref_ancho_cm': float(v.get('corral_ref_ancho_cm', 0) or 0),
        'malla': {
            'xmin': float(v['malla_xmin_cm']), 'ymin': float(v['malla_ymin_cm']),
            'xmax': float(v['malla_xmax_cm']), 'ymax': float(v['malla_ymax_cm']),
        },
        # Region regada. El riego es la segunda intervencion documentada y la
        # unica que, combinada con la sombra, alcanza los 4 C de enfriamiento
        # que midieron Hill et al. (2015). A diferencia de la malla no depende
        # de la trayectoria del sol: se aplica o no se aplica sobre la arena.
        'riego_activo': str(v.get('riego_activo', '0')).strip() in ('1', 'si', 'true'),
        'riego': {
            'xmin': float(v.get('riego_xmin_cm', 0)),
            'ymin': float(v.get('riego_ymin_cm', 0)),
            'xmax': float(v.get('riego_xmax_cm', 0)),
            'ymax': float(v.get('riego_ymax_cm', 0)),
        },
        'supuestos': supuestos,
    }


def ajustar_a_corral(sitio, largo_cm, ancho_cm):
    """Reescala la malla y el riego al corral que se esta evaluando.

    POR QUE HACE FALTA
    ------------------
    En sitio.csv la malla y el riego se declaran en centimetros absolutos
    (0-3000 x 0-800), medidos sobre el corral que existe en campo. Si el corral
    cambia de tamano -la interfaz permite pedir otras dimensiones- esos
    rectangulos se quedan donde estaban y dejan de cubrirlo: con un corral de
    30 x 40 m, una malla de 0-800 cm de ancho cubriria la quinta parte de la
    superficie y el resto quedaria a pleno sol SIN QUE NADIE LO PIDIERA.

    El dato de campo es que el toldo cubre el corral COMPLETO. Esa relacion
    -cubre todo- es la que hay que conservar al cambiar de tamano, no los
    centimetros. Se reescala en proporcion al corral de referencia declarado
    en el CSV: lo que cubria la mitad del ancho sigue cubriendo la mitad, y lo
    que cubria todo sigue cubriendo todo.

    No inventa cobertura: si el CSV declara una malla parcial, el resultado
    sigue siendo parcial en la misma proporcion.
    """
    if sitio is None:
        return None

    ref_largo = float(sitio.get('corral_ref_largo_cm') or 0)
    ref_ancho = float(sitio.get('corral_ref_ancho_cm') or 0)
    if ref_largo <= 0 or ref_ancho <= 0:
        return sitio

    fx = float(largo_cm) / ref_largo
    fy = float(ancho_cm) / ref_ancho
    if abs(fx - 1.0) < 1e-9 and abs(fy - 1.0) < 1e-9:
        return sitio

    def escalar(r):
        return {'xmin': r['xmin'] * fx, 'xmax': r['xmax'] * fx,
                'ymin': r['ymin'] * fy, 'ymax': r['ymax'] * fy}

    ajustado = dict(sitio)
    ajustado['malla'] = escalar(sitio['malla'])
    ajustado['riego'] = escalar(sitio['riego'])
    ajustado['corral_largo_cm'] = float(largo_cm)
    ajustado['corral_ancho_cm'] = float(ancho_cm)
    # La altura del toldo NO se escala: es una medida fisica de los postes, no
    # una proporcion del corral. Un corral cinco veces mas ancho no tiene
    # postes cinco veces mas altos, y de esa altura depende cuanto sol entra
    # por los lados abiertos.
    return ajustado
