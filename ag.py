from evaluacion  import evaluar_poblacion
from operadores  import (seleccion_torneo, cruza_por_especie,
                         mutacion_combinada, poda_elitismo)
from inicializacion import inicializar_poblacion

# ── Parámetros internos del AG ─────────────────────────────────────────────────
_TAM_POB    = 50
_N_GEN      = 100
_PROB_CRUZA = 0.85
_PROB_MUT   = 0.15


def ejecutar_ag(nidos_entrada, gestor, base, corral,
                nidos_ocupados=None, callback=None):
   
    previos_raw = nidos_ocupados or []
    
    # Preprocesar nidos_ocupados a un caché NumPy para optimizar el rendimiento del AG
    import numpy as np
    previos_cache = {"__cache__": True}
    if nidos_ocupados:
        por_esp = {}
        for n in nidos_ocupados:
            e = n['especie']
            por_esp.setdefault(e, ([], [], []))
            por_esp[e][0].append(float(n['x']))
            por_esp[e][1].append(float(n['y']))
            por_esp[e][2].append(float(n.get('prof', 0)))
        
        for esp, (xs, ys, profs) in por_esp.items():
            e = base.get(esp, {})
            prof_opt = e.get('prof_opt', 45.0)
            xs_p = np.array(xs)
            ys_p = np.array(ys)
            prfs_p = np.array([float(pf) if pf != 0 else prof_opt for pf in profs])
            previos_cache[esp] = (xs_p, ys_p, prfs_p)

    h = {k: [] for k in [
        'mejor', 'promedio',
        'v1_mejor', 'v1_promedio',
        'v2_mejor', 'v3_mejor',
        'v3_promedio', 'v4_mejor'
    ]}

    # ── 1. Inicialización ──────────────────────────────────────────────────────
    poblacion = inicializar_poblacion(
        _TAM_POB, nidos_entrada, gestor, base, previos_raw)

    # ── 2. Evaluación inicial ──────────────────────────────────────────────────
    evaluar_poblacion(poblacion, gestor, base, corral, previos_cache)
    mejor_global = max(poblacion, key=lambda i: i.fitness).copia()
    _registrar(h, poblacion, mejor_global)

    fits_ini = [i.fitness for i in poblacion]
    prom_ini = sum(fits_ini) / len(fits_ini)
    print(f"\n{'='*60}")
    print(f"  AG: {len(nidos_entrada)} nidos nuevos | "
          f"{len(previos_raw)} previos | pop={_TAM_POB} | gen={_N_GEN}")
    print(f"  Fitness inicial -> mejor:{mejor_global.fitness:.4f} prom:{prom_ini:.4f}")
    print(f"  V1={mejor_global.v1:.3f} V2={mejor_global.v2:.3f} "
          f"V3={mejor_global.v3:.4f} V4={mejor_global.v4:.4f}")
    print(f"{'='*60}")

    # ── 3-6. Ciclo evolutivo ───────────────────────────────────────────────────
    for gen_num in range(_N_GEN):
        parejas      = seleccion_torneo(poblacion, k=3)
        descendencia = []

        for p1, p2 in parejas:
            h1, h2 = cruza_por_especie(p1, p2, _PROB_CRUZA)
            h1 = mutacion_combinada(h1, _PROB_MUT, base, gestor, previos_cache)
            h2 = mutacion_combinada(h2, _PROB_MUT, base, gestor, previos_cache)
            descendencia.extend([h1, h2])

        evaluar_poblacion(descendencia, gestor, base, corral, previos_cache)
        poblacion = poda_elitismo(poblacion, descendencia, _TAM_POB, n_elite=2)

        mejor_actual = max(poblacion, key=lambda i: i.fitness)
        if mejor_actual.fitness > mejor_global.fitness:
            mejor_global = mejor_actual.copia()

        _registrar(h, poblacion, mejor_global)

        if (gen_num + 1) % 25 == 0:
            fits = [i.fitness for i in poblacion]
            prom = sum(fits) / len(fits)
            print(f"  Gen {gen_num+1:3d} | Mejor={mejor_global.fitness:.4f} "
                  f"Prom={prom:.4f} | V1={mejor_global.v1:.3f} "
                  f"V3={mejor_global.v3:.4f}")

        if callback:
            fits = [i.fitness for i in poblacion]
            callback(gen_num + 1, mejor_global.fitness, sum(fits)/len(fits))

    fits_fin = [i.fitness for i in poblacion]
    prom_fin = sum(fits_fin) / len(fits_fin)
    print(f"\n  [OK] Finalizado | Mejor={mejor_global.fitness:.4f} Prom={prom_fin:.4f}")
    print(f"  V1={mejor_global.v1:.3f} V2={mejor_global.v2:.3f} "
          f"V3={mejor_global.v3:.4f} V4={mejor_global.v4:.4f}")

    top3 = sorted(poblacion, key=lambda i: i.fitness, reverse=True)[:3]
    return mejor_global, top3, h


def _registrar(h, pob, mejor):
    fits = [i.fitness for i in pob]
    v1s  = [i.v1 for i in pob if i.v1 is not None]
    v3s  = [i.v3 for i in pob if i.v3 is not None]

    h['mejor'].append(mejor.fitness)
    h['promedio'].append(sum(fits) / len(fits) if fits else 0)
    h['v1_mejor'].append(mejor.v1 or 0)
    h['v1_promedio'].append(sum(v1s) / len(v1s) if v1s else 0)
    h['v2_mejor'].append(mejor.v2 or 0)
    h['v3_mejor'].append(mejor.v3 or 0)
    h['v3_promedio'].append(sum(v3s) / len(v3s) if v3s else 0)
    h['v4_mejor'].append(mejor.v4 or 0)
