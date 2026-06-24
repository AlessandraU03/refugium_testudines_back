import os, sys, math
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))

from cromosoma import (cargar_base_conocimiento, GestorZonas,
                       RegistroJornadas, calcular_capacidad_restante)
from ag import ejecutar_ag

app  = Flask(__name__)
CORS(app)

CSV_DIR       = os.path.join(os.path.dirname(__file__), 'csv')
JORNADAS_FILE = os.path.join(os.path.dirname(__file__), 'jornadas.json')

BASE, CORRAL = cargar_base_conocimiento(CSV_DIR)
REGISTRO     = RegistroJornadas(JORNADAS_FILE)


@app.route('/api/base-conocimiento')
def base_conocimiento():
    return jsonify({'base': BASE, 'corral': CORRAL})


@app.route('/api/ejecutar', methods=['POST'])
def ejecutar():
    data = request.get_json()

    n_g   = max(0, int(data.get('n_golfina', 0)))
    n_p   = max(0, int(data.get('n_prieta',  0)))
    n_l   = max(0, int(data.get('n_laud',    0)))
    fecha = data.get('fecha', datetime.today().strftime('%Y-%m-%d'))

    if n_g + n_p + n_l == 0:
        return jsonify({'error': 'Ingresa al menos 1 nido'}), 400

    gestor = GestorZonas(CORRAL, n_g, n_p, n_l, BASE)

    nidos_entrada = (
        [(i,             'golfina') for i in range(n_g)] +
        [(n_g + i,       'prieta')  for i in range(n_p)] +
        [(n_g + n_p + i, 'laud')    for i in range(n_l)]
    )

    try:
        nidos_activos = REGISTRO.obtener_nidos_activos(fecha, BASE)
    except Exception:
        nidos_activos = []

    nidos_ocupados = [{'x': n['x'], 'y': n['y'], 'especie': n['especie']}
                      for n in nidos_activos]

    mejor, top3, historial = ejecutar_ag(
        nidos_entrada, gestor, BASE, CORRAL, nidos_ocupados)

    def ser_ind(ind):
        return {
            'fitness': round(ind.fitness, 6),
            'v1': round(ind.v1, 4),
            'v2': round(ind.v2, 4),
            'v3': round(ind.v3, 4),
            'v4': round(ind.v4, 4),
            'genes': [{
                'id':      g.id_nido,
                'especie': g.especie,
                'x':       g.x,
                'y':       g.y,
                'prof':    g.prof,
                'zona':    g.zona_correcta(),
            } for g in ind.genes],
        }

    # Fechas de eclosión
    try:
        f_siembra = datetime.strptime(fecha, '%Y-%m-%d')
    except ValueError:
        f_siembra = datetime.today()

    fechas = []
    for esp in ['golfina', 'prieta', 'laud']:
        if not any(g.especie == esp for g in mejor.genes):
            continue
        dp = BASE[esp]['dias_prom']
        fechas.append({
            'especie':         esp,
            'fecha_siembra':   f_siembra.strftime('%Y-%m-%d'),
            'dias_incubacion': dp,
            'fecha_eclosion':  (f_siembra + timedelta(days=dp)).strftime('%Y-%m-%d'),
        })

    # Validación por especie
    import numpy as np
    validacion = []
    for esp in ['golfina', 'prieta', 'laud']:
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
            dist = np.sqrt(dx**2 + dy**2)
            for li in range(n_e):
                dist[li, li] = np.inf
            ok = (dist >= e['sep_min']).sum(axis=1)
            fs = ok / (n_all - 1)
        else:
            fs = np.ones(n_e)

        tasas = e['tasa_promedio'] + (e['tasa_maxima'] - e['tasa_promedio']) * fp * fs
        validacion.append({
            'especie':  esp,
            'historico': round(float(e['tasa_promedio']), 4),
            'estimado':  round(float(tasas.mean()), 4),
            'maximo':    round(float(e['tasa_maxima']), 4),
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
        'zonas_jornada':  n.get('zonas_jornada', {}),
        'jornada_previa': True,
    } for n in nidos_activos]

    total_corral = len(mejor.genes) + len(nidos_activos)

    return jsonify({
        'historial':     historial,
        'top3':          [ser_ind(i) for i in top3],
        'mejor':         ser_ind(mejor),
        'fechas':        fechas,
        'validacion':    validacion,
        'zonas':         zonas,
        'corral':        CORRAL,
        'n_golfina':     n_g,
        'n_prieta':      n_p,
        'n_laud':        n_l,
        'nidos_previos': nidos_previos_serial,
        'n_previos':     len(nidos_activos),
        'total_corral':  total_corral,
    })


@app.route('/api/guardar-jornada', methods=['POST'])
def guardar_jornada():
    data  = request.get_json()
    fecha = data['fecha']
    n_g, n_p, n_l = data['n_golfina'], data['n_prieta'], data['n_laud']

    from cromosoma import Gen, Individuo
    genes = [Gen(g['id'], g['especie'], g['x'], g['y'], g['prof'])
             for g in data['mejor']['genes']]
    ind = Individuo(genes)
    ind.fitness = data['mejor']['fitness']
    ind.v1 = data['mejor']['v1']
    ind.v2 = data['mejor']['v2']
    ind.v3 = data['mejor']['v3']
    ind.v4 = data['mejor']['v4']

    # params guardados en jornada solo para registro histórico
    params = {'tam_pob': 50, 'n_gen': 100, 'prob_cruza': 0.85, 'prob_mut': 0.15}
    # Reconstruir gestor con los mismos conteos para guardar los límites de zona
    gestor_guardar = GestorZonas(CORRAL, n_g, n_p, n_l, BASE)
    jornada = REGISTRO.guardar_jornada(fecha, n_g, n_p, n_l, ind, params, gestor=gestor_guardar)
    return jsonify({'ok': True, 'jornada': jornada})


@app.route('/api/corral-temporada')
def corral_temporada():
    fecha_hoy = datetime.today().strftime('%Y-%m-%d')
    try:
        nidos = REGISTRO.obtener_nidos_activos(fecha_hoy, BASE)
    except Exception:
        nidos = []
    return jsonify({
        'nidos':   nidos,
        'resumen': REGISTRO.resumen_temporada(),
        'corral':  CORRAL,
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
