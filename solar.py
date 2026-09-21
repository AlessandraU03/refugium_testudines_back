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


_CACHE_NIDO = {}
_CACHE_TRAYECTORIA = {}


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
