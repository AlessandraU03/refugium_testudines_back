import os, sys, math
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from cromosoma import (cargar_base_conocimiento, GestorZonas,
                       RegistroJornadas, calcular_capacidad_restante)
from ag import ejecutar_ag
from evaluacion import calcular_pts_window, calcular_semana_incubacion

# Servidor API Flask - Refugium Testudinis
app  = Flask(__name__)
CORS(app)

CSV_DIR       = os.path.join(os.path.dirname(__file__), 'csv')
JORNADAS_FILE = os.path.join(os.path.dirname(__file__), 'jornadas.json')

BASE, CORRAL = cargar_base_conocimiento(CSV_DIR)
REGISTRO     = RegistroJornadas(JORNADAS_FILE)


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


def calcular_modelo_girondot(f_siembra_dt, prof_cm):
    """
    Modelo de Sandoval et al. (2020) con ecuación de Girondot (1999) para Lepidochelys olivacea:
    Pivote P = 29.95 °C, S = -0.6301
    Formula: Pm = 1 / (1 + exp((P - T) / S))
    """
    mes = f_siembra_dt.month
    # Perfil térmico típico del Pacífico mexicano (Sandoval 2020, de la Torre 2017)
    temps_mes = {
        1: 25.5, 2: 26.0, 3: 27.0, 4: 28.5, 5: 30.0, 6: 31.8,
        7: 33.2, 8: 32.8, 9: 31.9, 10: 30.1, 11: 27.2, 12: 25.0
    }
    t_base = temps_mes.get(mes, 30.0)
    # A 45 cm la arena amortigua el calor solar; a menor profundidad se calienta más
    delta_prof = (45.0 - float(prof_cm)) * 0.08
    t_media_pts = round(t_base + delta_prof, 2)

    pivote = 29.95
    s = -0.6301
    try:
        exp_val = math.exp((pivote - t_media_pts) / s)
        pm = 1.0 / (1.0 + exp_val)
    except OverflowError:
        pm = 0.0 if t_media_pts > pivote else 1.0

    pm = max(0.0, min(1.0, pm))
    pct_macho = round(pm * 100.0, 1)
    pct_hembra = round((1.0 - pm) * 100.0, 1)
    return {
        'temp_estimada_pts': t_media_pts,
        'pct_machos': pct_macho,
        'pct_hembras': pct_hembra,
        'sesgo': 'Feminizado (Predominio Hembras)' if pct_hembra > 65.0 else ('Masculinizado (Predominio Machos)' if pct_macho > 65.0 else 'Equilibrado (~50/50)')
    }


def obtener_clustering(nidos):
    if not nidos:
        return {'centroids': [], 'labels': [], 'clusters': []}
    
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
        es_hotspot = conteo_activos >= 12
        
        clusters_list.append({
            'id': i,
            'cx': float(centroids[i][0]),
            'cy': float(centroids[i][1]),
            'nidos': nidos_cluster,
            'conteo_total': len(nidos_cluster),
            'conteo_activos': conteo_activos,
            'es_hotspot': es_hotspot
        })
        
    return {
        'centroids': centroids.tolist(),
        'labels': labels.tolist(),
        'clusters': clusters_list
    }


@app.route('/api/base-conocimiento')
def base_conocimiento():
    return jsonify({'base': BASE, 'corral': CORRAL})


@app.route('/api/ejecutar', methods=['POST'])
def ejecutar():
    data = request.get_json()

    n_g   = max(0, int(data.get('n_golfina', 0)))
    n_p   = 0
    n_l   = 0
    fecha = data.get('fecha', datetime.today().strftime('%Y-%m-%d'))

    if n_g == 0:
        return jsonify({'error': 'Ingresa al menos 1 nido de Golfina'}), 400

    gestor = GestorZonas(CORRAL, n_g, n_p, n_l, BASE)
    nidos_entrada = [(i, 'golfina') for i in range(n_g)]

    try:
        nidos_activos = REGISTRO.obtener_nidos_activos(fecha, BASE)
    except Exception:
        nidos_activos = []

    nidos_ocupados = [{'x': n['x'], 'y': n['y'], 'especie': n['especie']}
                      for n in nidos_activos]

    mejor, top3, historial = ejecutar_ag(
        nidos_entrada, gestor, BASE, CORRAL, nidos_ocupados)

    # Parsear fecha de siembra
    try:
        f_siembra = datetime.strptime(fecha, '%Y-%m-%d')
    except ValueError:
        f_siembra = datetime.today()

    # Calcular ventana de PTS
    pts_window = calcular_pts_window(f_siembra, 'golfina', BASE)

    def ser_ind(ind):
        return {
            'fitness': round(ind.fitness, 6),
            'v1': round(ind.v1, 4),
            'v2': round(ind.v2, 4),
            'v3': round(ind.v3, 4),
            'orden': round(ind.orden, 4) if ind.orden is not None else None,
            'genes': [{
                'id':               g.id_nido,
                'especie':          g.especie,
                'x':                g.x,
                'y':                g.y,
                'prof':             g.prof,
                'sector':           obtener_sector_fisico(g.x, g.y),
                'zona':             g.zona_correcta(),
                'num_huevas':       g.num_huevas if hasattr(g, 'num_huevas') else 0,
                'orden_incubacion': g.orden_incubacion if hasattr(g, 'orden_incubacion') else g.id_nido,
                'fecha_siembra':    fecha,
                'pts':              pts_window,
                'semana_info':      calcular_semana_incubacion(f_siembra, datetime.today(), BASE, g.especie),
                'proporcion_sexual': calcular_modelo_girondot(f_siembra, g.prof)
            } for g in ind.genes],
        }

    # Fechas de eclosión y Periodo Termosensible (PTS)
    fechas = []
    dp = BASE['golfina']['dias_prom']
    fecha_min = (f_siembra + timedelta(days=45)).strftime('%Y-%m-%d')
    fecha_max = (f_siembra + timedelta(days=55)).strftime('%Y-%m-%d')

    pts_ini_dias = round(dp / 3)
    pts_fin_dias = round(2 * dp / 3)
    pts_ini_str = (f_siembra + timedelta(days=pts_ini_dias)).strftime('%Y-%m-%d')
    pts_fin_str = (f_siembra + timedelta(days=pts_fin_dias)).strftime('%Y-%m-%d')

    prof_media = float(np.mean([g.prof for g in mejor.genes])) if mejor.genes else 45.0
    sex_ratio_global = calcular_modelo_girondot(f_siembra, prof_media)

    fechas.append({
        'especie':            'golfina',
        'fecha_siembra':      f_siembra.strftime('%Y-%m-%d'),
        'dias_incubacion':    dp,
        'fecha_eclosion':     (f_siembra + timedelta(days=dp)).strftime('%Y-%m-%d'),
        'fecha_eclosion_min': fecha_min,
        'fecha_eclosion_max': fecha_max,
        'pts_inicio':         pts_ini_str,
        'pts_fin':            pts_fin_str,
        'pts_dias':           f"Días {pts_ini_dias} al {pts_fin_dias} de incubación",
        'proporcion_sexual':  sex_ratio_global,
    })

    # Validación
    validacion = []
    gs_esp = [g for g in mejor.genes if g.especie == 'golfina']
    if gs_esp:
        e    = BASE['golfina']
        prfs = np.array([g.prof for g in gs_esp])
        fp   = np.exp(-((prfs - e['prof_opt'])**2) / (2 * e['sigma']**2))
        xs_n = np.array([g.x for g in gs_esp])
        ys_n = np.array([g.y for g in gs_esp])
        n_e  = len(gs_esp)

        xs_prev = np.array([n['x'] for n in nidos_ocupados if n['especie'] == 'golfina'])
        ys_prev = np.array([n['y'] for n in nidos_ocupados if n['especie'] == 'golfina'])
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
        validacion.append({
            'especie':  'golfina',
            'historico': round(float(e['tasa_promedio']), 4),
            'estimado':  round(float(tasas.mean()), 4),
            'maximo':    round(float(e['tasa_maxima']), 4),
            'benchmark_oaxaca_eclosion': 0.866,   # de la Torre-Robles et al. (2017)
            'benchmark_oaxaca_emergencia': 0.827, # de la Torre-Robles et al. (2017)
            'benchmark_oaxaca_mortalidad': 0.053, # de la Torre-Robles et al. (2017)
            'benchmark_sinaloa_pivote': 29.95,     # Sandoval et al. (2020)
        })

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
    capacidad_max = int(CORRAL.get('capacidad_maxima_nidos_simultaneos', 400))
    pct = (len(nidos_activos_incubando) / capacidad_max) * 100.0 if capacidad_max > 0 else 0.0
    alerta_calor = {
        'activada': pct >= 75.0,
        'mensaje': "Atención: Capacidad de incubación alta (>75%). El calor metabólico acumulado puede elevar la temperatura de la arena por encima del umbral crítico de 29.7°C, induciendo feminización de crías (sesgo de género) y afectando la viabilidad embrionaria." if pct >= 75.0 else None
    }

    return jsonify({
        'historial':     historial,
        'top3':          [ser_ind(i) for i in top3],
        'mejor':         ser_ind(mejor),
        'fechas':        fechas,
        'validacion':    validacion,
        'zonas':         zonas,
        'corral':        CORRAL,
        'n_golfina':     n_g,
        'n_prieta':      0,
        'n_laud':        0,
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
    n_p   = 0
    n_l   = 0

    from cromosoma import Gen, Individuo
    genes = [Gen(g['id'], g['especie'], g['x'], g['y'], g['prof'])
             for g in data['mejor']['genes']]
    ind = Individuo(genes)
    ind.fitness = data['mejor']['fitness']
    ind.v1 = data['mejor']['v1']
    ind.v2 = data['mejor']['v2']
    ind.v3 = data['mejor']['v3']

    params = {'tam_pob': 50, 'n_gen': 100, 'prob_cruza': 0.85, 'prob_mut': 0.15}
    gestor_guardar = GestorZonas(CORRAL, n_g, n_p, n_l, BASE)
    jornada = REGISTRO.guardar_jornada(fecha, n_g, n_p, n_l, ind, params, gestor=gestor_guardar)
    return jsonify({'ok': True, 'jornada': jornada})


@app.route('/api/capacidad')
def capacidad():
    """
    Diagnóstico de capacidad: contrasta la superficie instalada contra la
    curva estacional de llegada de nidos, a densidad de 1 nido/m2.
    """
    from capacidad import (capacidad_segura, simular_temporada,
                           resumen_temporada, separacion_implicita,
                           DENSIDAD_MAX_NIDOS_M2, DIAS_HASTA_EXCAVACION)
    from cromosoma import cargar_csv
    from datetime import date, timedelta
    import calendar

    corrales_rows = cargar_csv(os.path.join(CSV_DIR, 'corrales.csv'))
    corrales = [{
        'id':     int(r['corral_id']),
        'nombre': r['nombre'],
        'largo_m': float(r['largo_m']),
        'ancho_m': float(r['ancho_m']),
        'area_m2': float(r['largo_m']) * float(r['ancho_m']),
        'capacidad': capacidad_segura(r['largo_m'], r['ancho_m']),
        'fuente': r.get('fuente', ''),
    } for r in corrales_rows]

    cap_total = sum(c['capacidad'] for c in corrales)
    area_total = sum(c['area_m2'] for c in corrales)

    filas = cargar_csv(os.path.join(CSV_DIR, 'anidacion_mensual.csv'))
    anio = int(filas[0]['anio']) if filas else 2022
    por_mes = {int(f['mes']): float(f['nidos']) for f in filas}
    fuente_datos = filas[0].get('fuente', '') if filas else ''

    arribos = {}
    for m, n in por_mes.items():
        dim = calendar.monthrange(anio, m)[1]
        for d in range(dim):
            arribos[date(anio, m, 1) + timedelta(days=d)] = n / dim

    dias_inc = BASE['golfina']['dias_prom']
    serie = simular_temporada(arribos, cap_total, dias_inc)
    r = resumen_temporada(serie, cap_total)

    con_deficit = [d for d, v in serie.items() if v['deficit'] > 0]

    return jsonify({
        'densidad_max_nidos_m2': DENSIDAD_MAX_NIDOS_M2,
        'dias_hasta_excavacion': DIAS_HASTA_EXCAVACION,
        'dias_incubacion':       dias_inc,
        'anio_datos':            anio,
        'fuente_datos':          fuente_datos,
        'corrales':              corrales,
        'capacidad_total':       cap_total,
        'area_total_m2':         area_total,
        'nidos_temporada':       int(sum(por_mes.values())),
        'pico_ocupacion':        round(r['pico_ocupacion']),
        'fecha_pico':            r['fecha_pico'].strftime('%Y-%m-%d'),
        'cobertura_del_pico':    round(r['cobertura_del_pico'], 4),
        'deficit_maximo':        round(r['deficit_maximo']),
        'dias_con_deficit':      r['dias_con_deficit'],
        'area_faltante_m2':      round(r['area_faltante_m2']),
        'separacion_en_pico_m':  round(separacion_implicita(area_total, 1.0, r['pico_ocupacion']), 2),
        'ventana_inicio':        min(con_deficit).strftime('%Y-%m-%d') if con_deficit else None,
        'ventana_fin':           max(con_deficit).strftime('%Y-%m-%d') if con_deficit else None,
        'serie': [{'fecha': d.strftime('%Y-%m-%d'),
                   'ocupados': round(v['ocupados']),
                   'deficit':  round(v['deficit'])}
                  for d, v in serie.items() if d.year == anio],
    })


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
        'mensaje': "Atención: Capacidad de incubación alta (>75%). El calor metabólico acumulado puede elevar la temperatura de la arena por encima del umbral crítico de 29.7°C, induciendo feminización de crías (sesgo de género) y afectando la viabilidad embrionaria." if pct >= 75.0 else None
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
