import os, sys, math
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from cromosoma import (cargar_base_conocimiento, GestorZonas, GestoresZonas,
                       RegistroJornadas, orden_establecido, admitir_nidos,
                       piso_separacion)
from ag import ejecutar_ag
from inicializacion import individuo_secuencial
from evaluacion import (calcular_pts_window, calcular_semana_incubacion,
                        calcular_fitness)

# Servidor API Flask - Refugium Testudinis
app  = Flask(__name__)
CORS(app)

CSV_DIR       = os.path.join(os.path.dirname(__file__), 'csv')
JORNADAS_FILE = os.path.join(os.path.dirname(__file__), 'jornadas.json')

BASE, CORRAL = cargar_base_conocimiento(CSV_DIR)
REGISTRO     = RegistroJornadas(JORNADAS_FILE)

# Configuracion del sitio: latitud, longitud, orientacion del corral y
# geometria de la malla sombra. Es lo que vuelve general al programa: cambiando
# este CSV el sistema sirve para cualquier corral, no solo para Puerto Arista.
try:
    import solar
    SITIO = solar.cargar_sitio(CSV_DIR)
except Exception as _e:
    print('[aviso] no se pudo leer csv/sitio.csv (%s). La sombra se evaluara '
          'de forma binaria, sin geometria solar.' % _e)
    SITIO = None

# Pivote y parametro de forma de la ecuacion de Girondot, por especie.
PARAMS_ESPECIE = {
    e: {'pivote': BASE[e].get('pivote_temp'), 's': BASE[e].get('s_parameter')}
    for e in BASE
}


def _etiqueta_columna(idx):
    """Índice 0-based a etiqueta tipo hoja de cálculo: A..Z, AA, AB, ..."""
    etiqueta = ""
    idx += 1
    while idx > 0:
        idx, resto = divmod(idx - 1, 26)
        etiqueta = chr(65 + resto) + etiqueta
    return etiqueta


def sector_grid(corral=None):
    """Número de columnas y filas de sector que caben en el corral."""
    c = corral if corral is not None else CORRAL
    celda = float(c.get('sector_celda_cm') or 200)
    n_cols  = max(1, math.ceil(float(c['largo_cm']) / celda))
    n_filas = max(1, math.ceil(float(c['ancho_cm']) / celda))
    return celda, n_cols, n_filas


def obtener_sector_fisico(x_cm, y_cm, corral=None):
    """
    Traduce coordenadas en cm a la etiqueta del sector físico (p. ej. "F-7").

    La rejilla se deriva de largo_cm, ancho_cm y sector_celda_cm de
    corral_incubacion.csv, así que se adapta a cualquier corral sin tocar código.
    """
    celda, n_cols, n_filas = sector_grid(corral)
    col_idx = min(max(0, int(float(x_cm) / celda)), n_cols  - 1)
    row_idx = min(max(0, int(float(y_cm) / celda)), n_filas - 1)
    return f"{_etiqueta_columna(col_idx)}-{row_idx + 1}"


def calcular_modelo_girondot(f_siembra_dt, prof_cm, n_huevos=None,
                             densidad_m2=0.0, sombra=None, especie='golfina',
                             x_cm=None, y_cm=None, riego=None, sitio=None):
    """Proporcion sexual de un nido. Delega en termico.py.

    Antes esta funcion tenia su propia copia del modelo: su tabla de
    temperaturas, su gradiente por profundidad y su ecuacion de Girondot.
    termico.py implementa lo mismo con los coeficientes documentados, y tener
    dos copias hacia que el sistema se contradijera: para un nido a 40 cm en
    junio, una decia 97.3 % de hembras y la otra 92.6 %.

    Se conserva el nombre y la forma de la respuesta para no tocar a quienes
    la consumen, pero el calculo es ahora el unico del sistema, y ademas toma
    en cuenta el tamano de la nidada y la sombra, que la version anterior
    ignoraba.
    """
    import termico

    mes = f_siembra_dt.month

    # Sitio de ESTA jornada: si el corral no es el medido en campo, la malla y
    # el riego vienen reescalados a su tamano. Sin esto, un corral mas ancho se
    # evaluaba con la malla del corral chico y el resto quedaba a pleno sol.
    if sitio is None:
        sitio = SITIO

    # Sombra del nido. Con coordenadas y configuracion de sitio se resuelve por
    # geometria solar: la fraccion de la insolacion del dia que la malla le
    # intercepta, dada su posicion, la orientacion del corral, la altura de la
    # malla, la fecha y la latitud. Dos nidos del mismo corral pueden diferir
    # mas de cuarenta puntos de hembras solo por donde quedaron.
    if sombra is None:
        if sitio is not None and x_cm is not None and y_cm is not None:
            sombra = solar.fraccion_sombra_dia(
                float(x_cm), float(y_cm), f_siembra_dt.timetuple().tm_yday,
                sitio)
        else:
            sombra = termico.CORRAL_CON_MALLA_SOMBRA

    # Riego: la segunda intervencion del corral. Sin esto, esta funcion
    # devolvia la proporcion sexual de un corral SIN regar mientras el bloque
    # de rendimiento la calculaba CON riego, y la misma jornada aparecia con
    # 96.9 % de hembras en el diagrama y 68.3 % en el resumen.
    if riego is None:
        riego = bool(sitio and sitio.get('riego_activo')
                     and (x_cm is None or y_cm is None
                          or termico.en_sombra(float(x_cm), float(y_cm),
                                               [sitio['riego']])))

    pe = PARAMS_ESPECIE.get(especie, {})
    huevos = termico.huevos_del_mes(mes) if n_huevos in (None, 0) else n_huevos
    t_pts = termico.temperatura_pts(termico.temperatura_base(mes),
                                    n_huevos=huevos, sombra=sombra,
                                    riego=riego, prof_cm=prof_cm)
    sexo = termico.proporcion_sexual(t_pts, pe.get('pivote'), pe.get('s'))
    pct_hembra = sexo['pct_hembras']
    pct_macho = sexo['pct_machos']
    return {
        'temp_estimada_pts': sexo['temp_pts_c'],
        'pct_machos': pct_macho,
        'pct_hembras': pct_hembra,
        'huevos_por_nido': round(float(huevos), 1),
        'sesgo': ('Feminizado (Predominio Hembras)' if pct_hembra > 65.0
                  else ('Masculinizado (Predominio Machos)' if pct_macho > 65.0
                        else 'Equilibrado (~50/50)')),
    }


def obtener_clustering(nidos):
    """Agrupa los nidos y mide la densidad real de cada uno.

    El agrupamiento k-means sirve para ver por donde se concentra la siembra,
    pero por si solo no dice si una zona esta sobrepoblada: un grupo de doce
    nidos puede estar holgado o apinado segun la superficie que ocupe.

    Por eso cada nido lleva ahora su densidad local en nidos/m2, medida sobre
    la vecindad de 1 m x 1 m que usaron Honarvar et al. (2008), y el factor de
    eclosion que esa densidad implica segun su curva. Asi la alerta deja de
    depender de un conteo arbitrario y pasa a compararse contra la densidad
    maxima documentada.
    """
    if not nidos:
        return {'centroids': [], 'labels': [], 'clusters': [], 'densidad': None}

    import termico
    from termico import DENSIDAD_MAX_NIDOS_M2

    activos = [n for n in nidos if not n.get('eclosionado', False)]
    pares = [(float(n['x']), float(n['y'])) for n in activos]
    dens_activos = termico.densidades_locales(pares) if pares else []
    por_id = {id(n): d for n, d in zip(activos, dens_activos)}

    for n in nidos:
        d = por_id.get(id(n))
        if d is None:
            n['densidad_m2'] = None
            n['factor_eclosion'] = None
        else:
            n['densidad_m2'] = round(d, 2)
            n['factor_eclosion'] = round(termico.factor_eclosion_por_densidad(d), 3)

    coords = np.array([[float(n['x']), float(n['y'])] for n in nidos], dtype=float)
    N = len(nidos)
    K = max(1, N // 15)
    if K > N:
        K = N
    
    np.random.seed(42)
    idx_centroids = np.random.choice(N, K, replace=False)
    centroids = coords[idx_centroids].copy()
    labels = np.zeros(N, dtype=int)
    
    for _ in range(30):
        dists = np.sum((coords[:, None, :] - centroids[None, :, :])**2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for i in range(K):
            mask = (labels == i)
            if np.sum(mask) > 0:
                centroids[i] = np.mean(coords[mask], axis=0)
                
    clusters_list = []
    for i in range(K):
        indices = np.where(labels == i)[0]
        nidos_cluster = [nidos[idx] for idx in indices]
        conteo_activos = sum(1 for n in nidos_cluster if not n.get('eclosionado', False))
        dens_cluster = [n['densidad_m2'] for n in nidos_cluster
                        if n.get('densidad_m2') is not None]
        dens_max = max(dens_cluster) if dens_cluster else 0.0
        dens_media = (sum(dens_cluster) / len(dens_cluster)) if dens_cluster else 0.0
        # Sobrepasar la densidad maxima documentada, no un conteo inventado
        es_hotspot = dens_max > DENSIDAD_MAX_NIDOS_M2
        
        clusters_list.append({
            'id': i,
            'cx': float(centroids[i][0]),
            'cy': float(centroids[i][1]),
            'nidos': nidos_cluster,
            'conteo_total': len(nidos_cluster),
            'conteo_activos': conteo_activos,
            'densidad_media_m2': round(dens_media, 2),
            'densidad_maxima_m2': round(dens_max, 2),
            'es_hotspot': es_hotspot
        })
        
    if dens_activos:
        sobre_norma = sum(1 for d in dens_activos if d > DENSIDAD_MAX_NIDOS_M2)
        sobre_ensayado = sum(1 for d in dens_activos
                             if d > termico.DENSIDAD_REFERENCIA_M2)
        factores = [termico.factor_eclosion_por_densidad(d) for d in dens_activos]
        resumen_dens = {
            'n_activos':            len(dens_activos),
            'densidad_norma_m2':    DENSIDAD_MAX_NIDOS_M2,
            'densidad_media_m2':    round(sum(dens_activos) / len(dens_activos), 2),
            'densidad_maxima_m2':   round(max(dens_activos), 2),
            'nidos_sobre_norma':    sobre_norma,
            'nidos_sobre_ensayado': sobre_ensayado,
            'umbral_ensayado_m2':   termico.DENSIDAD_REFERENCIA_M2,
            'indice_eclosion':      round(sum(factores) / len(factores), 3),
            'crias_perdidas_pct':   round((1 - sum(factores) / len(factores)) * 100, 1),
            'fuente':               "Honarvar, O'Connor y Spotila (2008), Oecologia 157:221-230",
        }
    else:
        resumen_dens = None

    return {
        'centroids': centroids.tolist(),
        'labels': labels.tolist(),
        'clusters': clusters_list,
        'densidad': resumen_dens,
    }


@app.route('/api/base-conocimiento')
def base_conocimiento():
    """Parametros del sistema, y cuales de ellos son supuestos.

    `supuestos` lista los campos que NO provienen de una medicion ni de una
    fuente publicada. Se expone para que la interfaz los marque y el protocolo
    los declare: un supuesto declarado se puede defender, un numero inventado
    que se presento como medicion, no.
    """
    return jsonify({
        'base':   BASE,
        'corral': CORRAL,
        'sitio':  SITIO,
        'supuestos': (SITIO or {}).get('supuestos', []),
        'params_especie': PARAMS_ESPECIE,
    })


NOMBRE_BANDA = {
    'norma':    'Conforme a la NOM-162',
    'apretado': 'Cabe, pero por debajo de la norma',
    'excedido': 'Excede la capacidad fisica del corral',
}

CATEGORIA_UICN = {
    'CR': 'En Peligro Critico',
    'EN': 'En Peligro',
    'VU': 'Vulnerable',
}


def _serializar_cupo(cupo, gestores=None):
    """Prepara el resultado del cupo para la interfaz.

    Incluye la BANDA, que es lo que hay que leer primero: si la jornada cabe
    conforme a la norma, si cabe apretando, o si no cabe. Las tres son
    respuestas legitimas; la tercera es la que antes el sistema no sabia dar y
    resolvia encimando nidos.
    """
    d = dict(cupo)
    d['banda_nombre'] = NOMBRE_BANDA.get(cupo['banda'], cupo['banda'])
    d['categorias_nombre'] = {e: CATEGORIA_UICN.get(c, c)
                              for e, c in cupo.get('categorias', {}).items()}
    if gestores is not None and hasattr(gestores, 'zonas_sin_lugar'):
        d['zonas_sin_lugar'] = gestores.zonas_sin_lugar
    return d


@app.route('/api/ejecutar', methods=['POST'])
def ejecutar():
    data = request.get_json()

    n_g   = max(0, int(data.get('n_golfina', 0)))
    n_p   = max(0, int(data.get('n_prieta',  0)))
    n_l   = max(0, int(data.get('n_laud',    0)))
    fecha = data.get('fecha', datetime.today().strftime('%Y-%m-%d'))

    if n_g + n_p + n_l == 0:
        return jsonify({'error': 'Ingresa al menos 1 nido de alguna especie'}), 400

    # Medidas del corral. Vienen de csv/corral_incubacion.csv, pero la jornada
    # puede pedir otras: asi se puede ensayar que pasaria con un corral mas
    # largo o mas ancho sin tocar ningun archivo. Es parte de que el programa
    # sea general y no el de un solo sitio.
    corral = dict(CORRAL)
    try:
        if data.get('largo_m'):
            corral['largo_cm'] = max(100.0, float(data['largo_m']) * 100.0)
        if data.get('ancho_m'):
            corral['ancho_cm'] = max(100.0, float(data['ancho_m']) * 100.0)
    except (TypeError, ValueError):
        pass
    area_m2 = float(corral['largo_cm']) * float(corral['ancho_cm']) / 10000.0
    corral['area_m2'] = round(area_m2, 1)
    corral['capacidad_maxima_nidos_simultaneos'] = int(area_m2)  # 1 nido/m2

    # Malla y riego reescalados al corral de esta jornada. En sitio.csv estan
    # en centimetros del corral medido en campo (30 x 8 m); si la jornada pide
    # otro tamano, esos centimetros no lo siguen. El dato de campo es que el
    # toldo cubre el corral COMPLETO, y esa es la relacion que hay que
    # conservar, no los centimetros: con un corral de 30 x 40 m sin reescalar,
    # la malla cubriria la quinta parte y el 80 % de los nidos se evaluaria
    # como si no hubiera malla.
    sitio = (solar.ajustar_a_corral(SITIO, corral['largo_cm'], corral['ancho_cm'])
             if SITIO is not None else None)

    # Los seis repartos posibles de las franjas. El AG elige uno: con la malla
    # sombra cubriendo parte del corral, ese reparto decide a que especie le
    # toca el fresco. Medido en septiembre, prieta pasa de 0.11 a 0.97 de
    # sombra segun el reparto.
    # Las tres especies conviven en el corral, cada una en su zona y con sus
    # propios parametros: profundidad y separacion de la NOM-162, dias de
    # incubacion, ventana del PTS y pivote de determinacion sexual.
    try:
        nidos_activos = REGISTRO.obtener_nidos_activos(fecha, BASE)
    except Exception:
        nidos_activos = []

    nidos_ocupados = [{'x': n['x'], 'y': n['y'], 'especie': n['especie']}
                      for n in nidos_activos]

    # CUPO DEL CORRAL. Se resuelve antes de armar la entrada del AG.
    #
    # Con el piso de separacion puesto, el corral tiene un limite real y puede
    # llegar mas nido del que cabe. Antes eso no se detectaba: la rejilla se
    # apretaba sin fondo y los sobrantes se sembraban repitiendo la ultima
    # casilla, o sea varias nidadas en las mismas coordenadas. Ahora la jornada
    # cae en una de tres bandas -conforme a la norma, apretada por debajo de
    # ella, o excedida- y en la tercera se decide que nidos entran por
    # prioridad de riesgo de extincion, con el laud del Pacifico oriental
    # primero.
    #
    # Al AG solo se le entregan los nidos ADMITIDOS. Pedirle que coloque lo que
    # no cabe es pedirle que devuelva una colocacion imposible.
    cupo = admitir_nidos({'golfina': n_g, 'prieta': n_p, 'laud': n_l},
                         BASE, corral, nidos_ocupados)
    n_g = cupo['admitidos'].get('golfina', 0)
    n_p = cupo['admitidos'].get('prieta', 0)
    n_l = cupo['admitidos'].get('laud', 0)

    if n_g + n_p + n_l == 0:
        return jsonify({
            'error': 'El corral no admite ningun nido mas: ya esta lleno al '
                     'limite fisico de separacion (%s). Los %d nidos de esta '
                     'jornada necesitan otro destino.'
                     % (', '.join('%s %.0f cm' % (e, v)
                                  for e, v in sorted(cupo['piso'].items())),
                        cupo['total_demanda']),
            'cupo': cupo,
        }), 409

    # Los seis repartos posibles de las franjas. El AG elige uno: con la malla
    # sombra cubriendo parte del corral, ese reparto decide a que especie le
    # toca el fresco. Medido en septiembre, prieta pasa de 0.11 a 0.97 de
    # sombra segun el reparto.
    # Las tres especies conviven en el corral, cada una en su zona y con sus
    # propios parametros: profundidad y separacion de la NOM-162, dias de
    # incubacion, ventana del PTS y pivote de determinacion sexual.
    nidos_entrada = ([(i, 'golfina') for i in range(n_g)] +
                     [(n_g + i, 'prieta') for i in range(n_p)] +
                     [(n_g + n_p + i, 'laud') for i in range(n_l)])

    # El reparto de franjas se decide UNA VEZ, con el corral vacio. Despues
    # queda fijo: los nidos ya enterrados no se pueden cambiar de franja, y
    # pedirle al personal que mueva una especie al extremo opuesto del corral
    # no es una recomendacion sino una imposibilidad. Se deduce de donde estan
    # los nidos previos y se bloquea.
    orden_fijo = orden_establecido(nidos_ocupados)
    gestor = GestoresZonas(corral, n_g, n_p, n_l, BASE,
                           ordenes=[orden_fijo] if orden_fijo else None)

    # La fecha se parsea ANTES de correr el AG. De su mes dependen la
    # temperatura base de la arena y el tamano de la nidada, y por tanto la
    # proporcion sexual que el AG intenta alcanzar.
    try:
        f_siembra = datetime.strptime(fecha, '%Y-%m-%d')
    except ValueError:
        f_siembra = datetime.today()

    gestores = gestor
    mejor, top3, historial = ejecutar_ag(
        nidos_entrada, gestores, BASE, corral, nidos_ocupados,
        mes=f_siembra.month)

    # De aqui en adelante, la geometria de zonas del individuo GANADOR: es la
    # que hay que dibujar y reportar.
    gestor = gestores.para(mejor.idx_orden)
    orden_elegido = list(gestor.orden)

    # Ventana del PTS por especie: cada una tiene su propio tercio medio
    # (golfina dias 17-33, prieta 20-40, laud 22-43).
    pts_por_especie = {e: calcular_pts_window(f_siembra, e, BASE) for e in BASE}

    def ser_ind(ind):
        return {
            'fitness': round(ind.fitness, 6),
            # Que reparto de franjas usa este individuo. Sin esto no se puede
            # ver en el Top 3 que el AG probo distintos repartos.
            'idx_orden': getattr(ind, 'idx_orden', 0),
            'orden_zonas': list(gestores.para(getattr(ind, 'idx_orden', 0)).orden),
            'v1': round(ind.v1, 4),
            'v2': round(ind.v2, 4),
            'v3': round(ind.v3, 4),
            'orden': round(ind.orden, 4) if ind.orden is not None else None,
            'indice_eclosion': getattr(ind, 'indice_eclosion', None),
            'genes': [{
                'id':               g.id_nido,
                'especie':          g.especie,
                'x':                g.x,
                'y':                g.y,
                'prof':             g.prof,
                'sector':           obtener_sector_fisico(g.x, g.y, corral),
                'zona':             g.zona_correcta(),
                'num_huevas':       g.num_huevas if hasattr(g, 'num_huevas') else 0,
                'orden_incubacion': g.orden_incubacion if hasattr(g, 'orden_incubacion') else g.id_nido,
                'fecha_siembra':    fecha,
                'pts':              pts_por_especie.get(g.especie),
                'semana_info':      calcular_semana_incubacion(f_siembra, datetime.today(), BASE, g.especie),
                'sombra_solar':     (round(solar.fraccion_sombra_dia(
                                        g.x, g.y, f_siembra.timetuple().tm_yday,
                                        sitio), 3)
                                     if sitio is not None else None),
                'proporcion_sexual': calcular_modelo_girondot(
                                        f_siembra, g.prof, especie=g.especie,
                                        x_cm=g.x, y_cm=g.y, sitio=sitio)
            } for g in ind.genes],
        }

    # Fechas de eclosion y PTS, una fila por especie presente.
    #
    # Cada especie incuba distinto -golfina 45-55 dias, prieta 55-65, laud
    # 60-70- y su ventana del PTS sale de pts_termosensible.csv. Antes este
    # bloque suponia golfina para todo y calculaba el rango 45-55 a mano.
    fechas = []
    for esp in ('golfina', 'prieta', 'laud'):
        gs = [g for g in mejor.genes if g.especie == esp]
        if not gs:
            continue
        e  = BASE[esp]
        dp = e['dias_prom']
        w  = pts_por_especie[esp]

        # Posicion representativa de la especie para resolver su sombra. Es el
        # centroide de su zona: sirve para la fila resumen, mientras que cada
        # nido lleva su propia sombra y su propio sexo en ser_ind().
        prof_media = float(np.mean([g.prof for g in gs]))
        x_media    = float(np.mean([g.x for g in gs]))
        y_media    = float(np.mean([g.y for g in gs]))

        fechas.append({
            'especie':            esp,
            'fecha_siembra':      f_siembra.strftime('%Y-%m-%d'),
            'dias_incubacion':    dp,
            'nidos':              len(gs),
            'fecha_eclosion':     (f_siembra + timedelta(days=dp)).strftime('%Y-%m-%d'),
            'fecha_eclosion_min': (f_siembra + timedelta(days=e['dias_min'])).strftime('%Y-%m-%d'),
            'fecha_eclosion_max': (f_siembra + timedelta(days=e['dias_max'])).strftime('%Y-%m-%d'),
            'pts_inicio':         w['pts_inicio'],
            'pts_fin':            w['pts_fin'],
            'pts_dias':           'Días %d al %d de incubación'
                                  % (w['pts_inicio_dia'], w['pts_fin_dia']),
            'proporcion_sexual':  calcular_modelo_girondot(
                                      f_siembra, prof_media, especie=esp,
                                      x_cm=x_media, y_cm=y_media, sitio=sitio),
        })

    # Validación: tasa estimada contra la histórica, una fila por especie.
    validacion = []
    for esp in ('golfina', 'prieta', 'laud'):
        gs_esp = [g for g in mejor.genes if g.especie == esp]
        if not gs_esp:
            continue
        e    = BASE[esp]
        prfs = np.array([g.prof for g in gs_esp])
        fp   = np.exp(-((prfs - e['prof_opt'])**2) / (2 * e['sigma']**2))
        xs_n = np.array([g.x for g in gs_esp])
        ys_n = np.array([g.y for g in gs_esp])
        n_e  = len(gs_esp)

        xs_prev = np.array([n['x'] for n in nidos_ocupados if n['especie'] == esp])
        ys_prev = np.array([n['y'] for n in nidos_ocupados if n['especie'] == esp])
        xs_all  = np.concatenate([xs_n, xs_prev])
        ys_all  = np.concatenate([ys_n, ys_prev])
        n_all   = len(xs_all)

        if n_all > 1:
            dx   = xs_n[:, None] - xs_all[None, :]
            dy   = ys_n[:, None] - ys_all[None, :]
            dist_sq = dx**2 + dy**2
            for li in range(n_e):
                dist_sq[li, li] = np.inf
            ok = (dist_sq >= e['sep_min']**2).sum(axis=1)
            fs = ok / (n_all - 1)
        else:
            fs = np.ones(n_e)

        tasas = e['tasa_promedio'] + (e['tasa_maxima'] - e['tasa_promedio']) * fp * fs
        fila = {
            'especie':   esp,
            'nidos':     n_e,
            'historico': round(float(e['tasa_promedio']), 4),
            'estimado':  round(float(tasas.mean()), 4),
            'maximo':    round(float(e['tasa_maxima']), 4),
            'pivote_c':  e.get('pivote_temp'),
        }
        # Los contrastes de campo son de un corral mexicano real y solo
        # existen para las especies que ese estudio midio.
        if esp == 'golfina':
            fila.update({
                'benchmark_oaxaca_eclosion': 0.866,   # de la Torre-Robles et al. (2017)
                'benchmark_oaxaca_emergencia': 0.827, # de la Torre-Robles et al. (2017)
                'benchmark_oaxaca_mortalidad': 0.053, # de la Torre-Robles et al. (2017)
                'benchmark_sinaloa_pivote': 29.95,    # Sandoval et al., Playa Ceuta
            })
        elif esp == 'laud':
            fila.update({
                'benchmark_oaxaca_eclosion': 0.474,   # de la Torre-Robles et al. (2017)
                'benchmark_oaxaca_emergencia': 0.453, # de la Torre-Robles et al. (2017)
                'benchmark_oaxaca_mortalidad': 0.202, # de la Torre-Robles et al. (2017)
            })
        validacion.append(fila)

    zonas = [{
        'nombre':  nombre,
        'especie': lim['especie'],
        'xmin': lim['xmin'], 'xmax': lim['xmax'],
        'ymin': lim['ymin'], 'ymax': lim['ymax'],
    } for nombre, lim in gestor.limites.items()]

    nidos_previos_serial = [{
        'id':             n['id'],
        'especie':        n['especie'],
        'x':              n['x'],
        'y':              n['y'],
        'prof':           n['prof'],
        'fecha_siembra':  n['fecha_siembra'],
        'fecha_eclosion': n['fecha_eclosion'],
        'eclosionado':    n.get('eclosionado', False),
        'zonas_jornada':  n.get('zonas_jornada', {}),
        'jornada_previa': True,
    } for n in nidos_activos]

    total_corral = len(mejor.genes) + len(nidos_activos)

    # Preparar datos para clustering (previos + nuevos)
    todos_para_clustering = []
    for g in mejor.genes:
        todos_para_clustering.append({
            'id':             g.id_nido,
            'especie':        g.especie,
            'x':              g.x,
            'y':              g.y,
            'prof':           g.prof,
            'fecha_siembra':  fecha,
            'fecha_eclosion': (f_siembra + timedelta(days=BASE[g.especie]['dias_prom'])).strftime('%Y-%m-%d'),
            'eclosionado':    False,
            'tipo':           'nuevo'
        })
    for n in nidos_activos:
        todos_para_clustering.append({
            'id':             n['id'],
            'especie':        n['especie'],
            'x':              n['x'],
            'y':              n['y'],
            'prof':           n['prof'],
            'fecha_siembra':  n['fecha_siembra'],
            'fecha_eclosion': n['fecha_eclosion'],
            'eclosionado':    n.get('eclosionado', False),
            'tipo':           'previo'
        })

    clustering = obtener_clustering(todos_para_clustering)

    # Alerta de calor
    nidos_activos_incubando = [n for n in todos_para_clustering if not n.get('eclosionado', False)]
    capacidad_max = int(corral.get('capacidad_maxima_nidos_simultaneos', 400))
    pct = (len(nidos_activos_incubando) / capacidad_max) * 100.0 if capacidad_max > 0 else 0.0
    alerta_calor = {
        'activada': pct >= 75.0,
        # El hacinamiento NO feminiza. Honarvar et al. (2008) hallaron
        # diferencias de temperatura entre densidades solo en los dias 37, 40
        # y 47; el periodo termosensible de la golfina va del dia 17 al 33, de
        # modo que cuando el apinamiento calienta la arena el sexo ya quedo
        # determinado. Lo que cuesta el hacinamiento es ECLOSION, y en el
        # tercio final, riesgo de rebasar el limite letal.
        'mensaje': ("Atención: ocupación del corral por encima del 75 %. Al apretarse "
                    "los nidos sube la densidad local y cada nido pierde eclosión según "
                    "la curva de Honarvar et al. (2008); además el calor metabólico del "
                    "último tercio acerca la arena al límite letal de 36 °C. No altera la "
                    "proporción sexual: el sexo se determina entre los días 17 y 33, antes "
                    "de que el hacinamiento caliente la arena.") if pct >= 75.0 else None
    }

    # Separacion realmente usada. Si al pedir mas nidos que casillas la rejilla
    # tuvo que apretarse, aqui se ve: apretar cuesta eclosion y no debe pasar
    # inadvertido para quien siembra.
    # Cada especie tiene su propia separacion de norma (golfina 100 cm, prieta
    # 120, laud 150) y su propia rejilla, asi que se reporta por especie.
    conteo_esp = {'golfina': n_g, 'prieta': n_p, 'laud': n_l}
    por_especie = {}
    comprimida_alguna = False
    peor = None
    for esp, n_esp in conteo_esp.items():
        if not n_esp:
            continue
        s_norma = float(BASE[esp]['sep_min'])
        s_real  = gestor.separacion_efectiva.get('zona_%s' % esp, s_norma)
        comp    = s_real < s_norma - 0.05
        comprimida_alguna = comprimida_alguna or comp
        por_especie[esp] = {
            'norma_cm':    round(s_norma, 1),
            'efectiva_cm': round(s_real, 1),
            'comprimida':  comp,
            'nidos':       n_esp,
        }
        if comp and (peor is None or s_norma - s_real > peor[1]):
            peor = (esp, s_norma - s_real, n_esp, s_norma, s_real)

    # Se conservan las claves planas de la especie mas comprimida para no
    # romper a quien ya consumia este bloque.
    esp_ref = peor[0] if peor else ('golfina' if n_g else
                                    ('prieta' if n_p else 'laud'))
    ref = por_especie.get(esp_ref, {'norma_cm': 0, 'efectiva_cm': 0})
    separacion = {
        'por_especie': por_especie,
        'norma_cm':    ref['norma_cm'],
        'efectiva_cm': ref['efectiva_cm'],
        'comprimida':  comprimida_alguna,
        'mensaje': (
            'No cabían %d nidos de %s a %.0f cm. La rejilla se apretó a %.0f cm: '
            'cada nido pierde eclosión por hacinamiento.'
            % (peor[2], peor[0], peor[3], peor[4])
        ) if peor else None,
    }
    sep_norma = ref['norma_cm']

    # ------------------------------------------------------------------
    #  Que rinde esta colocacion, y cuanto de eso lo puso el AG
    # ------------------------------------------------------------------
    # Un fitness de 0.87 no le dice nada a quien siembra. Lo que si dice algo
    # es cuantas crias salen de esta colocacion, y sobre todo cuantas mas que
    # con el procedimiento actual.
    #
    # La referencia es la siembra secuencial: llenar la rejilla en serpentina
    # a la separacion de la norma, todo a la profundidad optima. Es lo que se
    # hace hoy en el corral sin ningun algoritmo, asi que es contra eso que el
    # AG tiene que demostrar que sirve. Se evalua con la misma funcion de
    # aptitud y el mismo modelo termico, sobre el mismo corral y los mismos
    # nidos previos: la unica diferencia es donde quedo cada nido.
    import termico
    referencia = individuo_secuencial(nidos_entrada, gestor, BASE, nidos_ocupados)
    calcular_fitness(referencia, gestor, BASE, corral, nidos_ocupados,
                     mes=f_siembra.month)

    tasas_especie = {e: BASE[e].get('tasa_eclosion', 0.75) for e in BASE}
    huevos_mes = termico.huevos_del_mes(f_siembra.month)

    # La malla sombra del corral, como una region que cubre toda su superficie.
    #
    # Hay que pasarla explicitamente: TEMP_BASE_MES_C esta construida como
    # corral DESCUBIERTO (le suma de vuelta el efecto de la malla) para que
    # aplicar sombra sea una resta. Evaluar sin este rectangulo describe un
    # corral sin malla, y da 34.2 C en el PTS con todos los nidos en riesgo
    # letal, que no es la situacion de Puerto Arista.
    # Respaldo binario por si no hay geometria solar: la misma malla declarada
    # en sitio.csv, no un corral cubierto por completo.
    sombra_corral = ([dict(sitio['malla'])] if sitio is not None else
                     ([{'xmin': 0.0, 'ymin': 0.0,
                        'xmax': float(corral['largo_cm']),
                        'ymax': float(corral['ancho_cm'])}]
                      if termico.CORRAL_CON_MALLA_SOMBRA else []))

    def rendir(ind):
        """Crias y sexo que produce una colocacion, con los nidos ya sembrados.

        Los nidos previos entran al calculo porque la densidad que rodea a un
        nido nuevo depende de lo que ya esta enterrado a su lado. Son los
        mismos en las dos colocaciones, asi que la diferencia entre ellas es
        atribuible al algoritmo.
        """
        nidos = list(ind.genes) + [
            {'x': n['x'], 'y': n['y'], 'especie': n['especie'], 'num_huevas': 0}
            for n in nidos_ocupados
        ]
        r = termico.evaluar_colocacion(nidos, tasas_especie,
                                       mes=f_siembra.month,
                                       rectangulos_sombra=sombra_corral,
                                       huevos_por_defecto=huevos_mes,
                                       sitio=sitio,
                                       dia_del_anio=f_siembra.timetuple().tm_yday,
                                       params_especie=PARAMS_ESPECIE)
        res = r['resumen']
        dens = [q['densidad_m2'] for q in r['por_nido']]
        return {
            'fitness':            round(float(ind.fitness), 6),
            'indice_eclosion':    r['indice_eclosion'],
            'crias_esperadas':    res['crias_esperadas'],
            'crias_hembras':      res['crias_hembras'],
            'crias_machos':       res['crias_machos'],
            'pct_hembras':        res['pct_hembras'],
            'temp_pts_media_c':   res['temp_pts_media_c'],
            'nidos_en_riesgo':    res['nidos_en_riesgo_termico'],
            'densidad_maxima_m2': round(max(dens), 2) if dens else 0.0,
        }

    rend_ag  = rendir(mejor)
    rend_sec = rendir(referencia)
    rendimiento = {
        'mes':               f_siembra.month,
        'nidos_evaluados':   len(mejor.genes) + len(nidos_ocupados),
        'nidos_nuevos':      len(mejor.genes),
        'ag':                rend_ag,
        'secuencial':        rend_sec,
        'ganancia_crias':    round(rend_ag['crias_esperadas'] - rend_sec['crias_esperadas'], 1),
        'ganancia_fitness':  round(rend_ag['fitness'] - rend_sec['fitness'], 6),
        'densidad_norma_m2': 1.0,
        'referencia': ('Llenado secuencial de la rejilla a %.0f cm de separacion '
                       'y profundidad optima: el procedimiento actual, sin algoritmo.'
                       % sep_norma),
    }

    return jsonify({
        'historial':     historial,
        'top3':          [ser_ind(i) for i in top3],
        'mejor':         ser_ind(mejor),
        'separacion':    separacion,
        'rendimiento':   rendimiento,
        'fechas':        fechas,
        'validacion':    validacion,
        'zonas':         zonas,
        'corral':        corral,
        'n_golfina':     n_g,
        'n_prieta':      n_p,
        'n_laud':        n_l,
        'orden_zonas':   orden_elegido,
        'orden_base':    ['golfina', 'prieta', 'laud'],
        # Si el corral ya tenia nidos, el reparto venia impuesto por ellos y el
        # AG no pudo elegirlo. La interfaz debe decirlo, para que nadie crea
        # que el algoritmo propone mover especies de franja entre jornadas.
        'cupo':          _serializar_cupo(cupo, gestores),
        'orden_fijo':    bool(orden_fijo),
        'orden_motivo':  ('Heredado de los %d nidos ya enterrados: cambiar de '
                          'franja obligaria a desenterrarlos.' % len(nidos_ocupados)
                          if orden_fijo else
                          'Corral vacio: el AG eligio el reparto, y queda fijo '
                          'para el resto de la temporada.'),
        'sitio':         ({'latitud': sitio['latitud'],
                           'longitud': sitio['longitud'],
                           'orientacion': sitio['orientacion'],
                           'malla': sitio['malla'],
                           'riego': sitio['riego'],
                           'riego_activo': sitio['riego_activo'],
                           'altura_malla': sitio['altura_malla'],
                           'atenuacion_malla': sitio['atenuacion_malla'],
                           'supuestos': sitio['supuestos']}
                          if sitio is not None else None),
        'nidos_previos': nidos_previos_serial,
        'n_previos':     len(nidos_activos),
        'total_corral':  total_corral,
        'clustering':    clustering,
        'alerta_calor':  alerta_calor
    })


@app.route('/api/guardar-jornada', methods=['POST'])
def guardar_jornada():
    data  = request.get_json()
    fecha = data['fecha']
    n_g   = int(data.get('n_golfina', 0))
    n_p   = int(data.get('n_prieta', 0))
    n_l   = int(data.get('n_laud', 0))

    from cromosoma import Gen, Individuo
    genes = [Gen(g['id'], g['especie'], g['x'], g['y'], g['prof'])
             for g in data['mejor']['genes']]
    ind = Individuo(genes)
    ind.fitness = data['mejor']['fitness']
    ind.v1 = data['mejor']['v1']
    ind.v2 = data['mejor']['v2']
    ind.v3 = data['mejor']['v3']

    params = {'tam_pob': 50, 'n_gen': 100, 'prob_cruza': 0.85, 'prob_mut': 0.15}
    # El reparto de franjas que eligio el AG, no el historico. Si se guardara
    # siempre el historico, las jornadas futuras dibujarian los nidos previos
    # en zonas que nunca fueron las suyas.
    orden_guardar = data.get('orden_zonas') or None
    gestor_guardar = GestorZonas(CORRAL, n_g, n_p, n_l, BASE,
                                 orden=orden_guardar)
    jornada = REGISTRO.guardar_jornada(fecha, n_g, n_p, n_l, ind, params, gestor=gestor_guardar)
    return jsonify({'ok': True, 'jornada': jornada})


@app.route('/api/corral-temporada')
def corral_temporada():
    fecha_hoy = datetime.today().strftime('%Y-%m-%d')
    try:
        nidos = REGISTRO.obtener_nidos_activos(fecha_hoy, BASE)
    except Exception:
        nidos = []

    nidos_activos_incubando = [n for n in nidos if not n.get('eclosionado', False)]
    capacidad_max = int(CORRAL.get('capacidad_maxima_nidos_simultaneos', 400))
    pct = (len(nidos_activos_incubando) / capacidad_max) * 100.0 if capacidad_max > 0 else 0.0

    alerta_calor = {
        'activada': pct >= 75.0,
        # El hacinamiento NO feminiza. Honarvar et al. (2008) hallaron
        # diferencias de temperatura entre densidades solo en los dias 37, 40
        # y 47; el periodo termosensible de la golfina va del dia 17 al 33, de
        # modo que cuando el apinamiento calienta la arena el sexo ya quedo
        # determinado. Lo que cuesta el hacinamiento es ECLOSION, y en el
        # tercio final, riesgo de rebasar el limite letal.
        'mensaje': ("Atención: ocupación del corral por encima del 75 %. Al apretarse "
                    "los nidos sube la densidad local y cada nido pierde eclosión según "
                    "la curva de Honarvar et al. (2008); además el calor metabólico del "
                    "último tercio acerca la arena al límite letal de 36 °C. No altera la "
                    "proporción sexual: el sexo se determina entre los días 17 y 33, antes "
                    "de que el hacinamiento caliente la arena.") if pct >= 75.0 else None
    }

    clustering = obtener_clustering(nidos)

    return jsonify({
        'nidos':   nidos,
        'resumen': REGISTRO.resumen_temporada(),
        'corral':  CORRAL,
        'clustering': clustering,
        'alerta_calor': alerta_calor
    })


@app.route('/api/resumen-temporada')
def resumen_temporada():
    return jsonify(REGISTRO.resumen_temporada())


@app.route('/api/nueva-temporada', methods=['POST'])
def nueva_temporada():
    REGISTRO.limpiar_temporada()
    return jsonify({'ok': True})


import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    tb = traceback.format_exc()
    print("=== GLOBAL EXCEPTION CAUGHT ===")
    print(tb)
    return jsonify({
        "error": str(e),
        "traceback": tb
    }), 500




@app.route('/api/recomendacion')
def recomendacion():
    """Que conviene hacer con el corral, dado que el presupuesto es limitado.

    Responde las tres preguntas del riego -cuanto, donde y como- y la del
    reemplazo de arena, cuantificando en cada caso el costo en recursos y la
    ganancia esperada en eclosion y proporcion sexual.

    No decide por el Santuario: presenta el intercambio. La dosis que mas
    rinde por litro y la que mas acerca al equilibrio sexual NO son la misma,
    y esa eleccion es de quien administra el corral.

    Parametros de consulta: mes (1-12), largo_m, ancho_m, prof_cm.
    """
    import termico

    mes     = int(request.args.get('mes', datetime.today().month))
    largo_m = float(request.args.get('largo_m', float(CORRAL['largo_cm']) / 100.0))
    ancho_m = float(request.args.get('ancho_m', float(CORRAL['ancho_cm']) / 100.0))
    prof_cm = float(request.args.get('prof_cm', BASE['golfina']['prof_opt']))

    area_m2 = largo_m * ancho_m
    base_t  = termico.temperatura_base(mes)
    huevos  = termico.huevos_del_mes(mes)
    pe      = PARAMS_ESPECIE.get('golfina', {})

    # Sombra representativa: la del centro del corral, que es donde el toldo
    # protege mas. El perimetro se trata aparte, mas abajo.
    sombra_centro = 0.0
    sombra_borde  = 0.0
    if SITIO is not None:
        # Misma correccion que en /api/ag: la malla se reescala al corral que
        # se esta consultando, o la recomendacion de riego se calcularia con la
        # sombra de un corral que no es este.
        sitio_rec = solar.ajustar_a_corral(SITIO, largo_m * 100, ancho_m * 100)
        dia = datetime(datetime.today().year, mes, 15).timetuple().tm_yday
        sombra_centro = solar.fraccion_sombra_dia(largo_m * 50, ancho_m * 50, dia, sitio_rec)
        sombra_borde  = solar.fraccion_sombra_dia(50.0, 50.0, dia, sitio_rec)

    def escenario(mm, sombra, area_regada_m2):
        t_pts = termico.temperatura_pts(base_t, n_huevos=huevos, sombra=sombra,
                                        riego=mm, prof_cm=prof_cm)
        t_fin = termico.temperatura_ultimo_tercio(base_t, n_huevos=huevos,
                                                  densidad_m2=1.0, sombra=sombra,
                                                  riego=mm, prof_cm=prof_cm)
        sexo = termico.proporcion_sexual(t_pts, pe.get('pivote'), pe.get('s'))
        riesgo = termico.riesgo_letal(t_fin)
        # 1 mm sobre 1 m2 es 1 litro
        m3 = mm * area_regada_m2 / 1000.0
        return {
            'dosis_mm':        mm,
            'area_regada_m2':  round(area_regada_m2, 1),
            'agua_m3':         round(m3, 1),
            'temp_pts_c':      t_pts,
            'pct_hembras':     sexo['pct_hembras'],
            'pct_machos':      sexo['pct_machos'],
            'margen_letal_c':  riesgo['margen_c'],
            'en_riesgo':       riesgo['en_riesgo'],
        }

    # --- CUANTO regar -----------------------------------------------------
    sin_riego = escenario(0, sombra_centro, area_m2)
    dosis = []
    for mm in (0, 50, 100, 200, 323, 500, 721):
        e = escenario(mm, sombra_centro, area_m2)
        ganancia = sin_riego['temp_pts_c'] - e['temp_pts_c']
        e['ganancia_c'] = round(ganancia, 2)
        e['grados_por_m3'] = round(ganancia / e['agua_m3'], 4) if e['agua_m3'] else None
        dosis.append(e)

    utiles = [e for e in dosis if e['grados_por_m3']]
    mas_eficiente = max(utiles, key=lambda e: e['grados_por_m3']) if utiles else None
    # La que mas acerca al equilibrio sexual, sin reparar en el costo
    mas_equilibrada = min(dosis, key=lambda e: abs(e['pct_hembras'] - 50.0))

    # --- DONDE regar ------------------------------------------------------
    # El toldo cubre por arriba pero los lados estan abiertos, asi que el sol
    # bajo entra de costado y el perimetro se calienta mas que el centro. Regar
    # solo esa franja cuesta menos agua y ataca donde duele.
    franja_m = 1.5
    area_borde = max(0.0, area_m2 - max(0.0, largo_m - 2 * franja_m) * max(0.0, ancho_m - 2 * franja_m))
    mm_rec = mas_eficiente['dosis_mm'] if mas_eficiente else 100
    donde = {
        'franja_m':        franja_m,
        'area_perimetro_m2': round(area_borde, 1),
        'area_total_m2':   round(area_m2, 1),
        'sombra_centro':   round(sombra_centro, 3),
        'sombra_perimetro': round(sombra_borde, 3),
        'solo_perimetro':  escenario(mm_rec, sombra_borde, area_borde),
        'todo_el_corral':  escenario(mm_rec, sombra_centro, area_m2),
        'perimetro_sin_regar': escenario(0, sombra_borde, area_borde),
    }

    # --- COMO regar -------------------------------------------------------
    como = {
        'frecuencia':  'diaria, por la tarde',
        'dias':        termico.RIEGO_DIAS,
        'reparto':     'la misma cantidad cada dia',
        'ventana':     'durante el periodo termosensible, que es cuando se define el sexo',
        'fuente':      ('Hill, Paladino, Spotila y Santidrian Tomillo (2015), PLOS ONE '
                        '10(6):e0129528: regaron a diario por la tarde, en cantidades '
                        'iguales, durante 31 dias'),
        'advertencia': ('Los coeficientes corresponden a ESA pauta. Regar menos veces, '
                        'o de golpe, no produce el mismo enfriamiento. El estudio no '
                        'midio el efecto del riego sobre la eclosion ni sobre la '
                        'humedad del nido: sus propios autores lo dejan como pendiente.'),
    }

    # --- CAMBIO DE ARENA --------------------------------------------------
    prof_max_cm = max(BASE[e]['prof_max'] for e in BASE)
    arena = {
        'recomendacion':   'reemplazar la arena al terminar cada temporada',
        'volumen_m3':      round(area_m2 * prof_max_cm / 100.0, 1),
        'profundidad_cm':  prof_max_cm,
        'motivo':          ('En corrales que reutilizan arena se acumulan hongos y '
                            'bacterias patogenas. Se midio 5.2 % de Fusarium solani '
                            '(FSSC) en corrales contra 1.3 % en playa natural.'),
        'fuente':          ('Hoh, Lin, Liu, Mohamed Sidique y Tsai (2020). Nest '
                            'microbiota and pathogen abundance in sea turtle '
                            'hatcheries. Fungal Ecology, 47, 100964.'),
        'advertencia':     ('La fuente recomienda el reemplazo por temporada y mide la '
                            'carga de patogenos, pero NO da una ganancia de eclosion en '
                            'puntos porcentuales. No se inventa una cifra: el volumen '
                            'de arena es el costo, y la reduccion de patogenos el '
                            'beneficio documentado.'),
    }

    return jsonify({
        'mes': mes,
        'corral': {'largo_m': largo_m, 'ancho_m': ancho_m, 'area_m2': round(area_m2, 1)},
        'profundidad_cm': prof_cm,
        'cuanto': {
            'escenarios': dosis,
            'mas_eficiente': mas_eficiente,
            'mas_equilibrada': mas_equilibrada,
            'nota': ('La dosis que mas rinde por litro y la que mas acerca al '
                     'equilibrio sexual no son la misma. Elegir entre ellas es una '
                     'decision de presupuesto, no del modelo.'),
        },
        'donde': donde,
        'como': como,
        'arena': arena,
        'supuestos': (SITIO or {}).get('supuestos', []),
    })


# Arranque como script. Tiene que quedar al final del archivo: app.run()
# bloquea, asi que cualquier @app.route() escrito debajo nunca llegaria a
# registrarse y esa ruta responderia 404 al correr 'python api.py'.
# Con gunicorn o con test_client no se nota, porque ahi el modulo se
# importa y app.run() no se ejecuta.
if __name__ == '__main__':
    app.run(debug=True, port=5000)
