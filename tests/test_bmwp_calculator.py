"""Tests del índice BMWP.

Cada test corresponde a uno de los cuatro defectos que la auditoría encontró
(B1-B4). Si alguno vuelve a romperse, el sistema produciría un veredicto de
calidad de agua incorrecto sin avisar, que es el peor modo de falla posible
para un sistema de bioindicación.
"""

from utils.bmwp_calculator import BMWPCalculator


def _det(familia, cantidad=1, conf=0.9):
    return {"familia": familia, "cantidad": cantidad, "confidence_promedio": conf}


# --- B1: presencia/ausencia, no abundancia -------------------------------

def test_la_abundancia_no_cambia_el_puntaje():
    """Cuarenta quironómidos indican lo mismo que uno: presencia de un taxón
    tolerante. Multiplicar por abundancia convertía un sitio degradado en uno
    'muy limpio'."""
    calc = BMWPCalculator()

    uno = calc.calculate_site([_det("Chironomidae", cantidad=1)])
    muchos = calc.calculate_site([_det("Chironomidae", cantidad=40)])

    assert uno.total_score == muchos.total_score == 2
    assert muchos.water_quality_class == "V"  # muy crítica, no "muy limpia"


def test_cada_familia_aporta_una_sola_vez_aunque_venga_repetida():
    calc = BMWPCalculator()

    resultado = calc.calculate_site([
        _det("Dytiscidae", cantidad=3),
        _det("Dytiscidae", cantidad=5),
    ])

    assert resultado.total_score == 9
    assert resultado.n_families_scored == 1


# --- B2: cobertura completa, sin omisiones silenciosas --------------------

def test_las_19_familias_del_modelo_tienen_puntaje():
    """El modelo reconoce 19 familias; si alguna no tiene puntaje queda fuera
    del índice, y antes eso pasaba en silencio para 10 de ellas."""
    calc = BMWPCalculator()
    familias_modelo = {
        "Ampullariidae", "Ancylidae", "Belostomatidae", "Ceratopogonidae",
        "Chironomidae", "Coenagrionidae", "Dytiscidae", "Gerridae",
        "Glossiphoniidae", "Hirudinidae", "Hydrophilidae", "Hyriidae",
        "Libellulidae", "Miridae", "Noteridae", "Notonectidae", "Physidae",
        "Planorbidae", "Psychodidae",
    }

    assert familias_modelo <= set(calc.get_available_families())


def test_una_familia_sin_puntaje_se_reporta_en_vez_de_descartarse():
    calc = BMWPCalculator()

    resultado = calc.calculate_site([_det("Dytiscidae"), _det("Familia Inventada")])

    assert resultado.unscored_families == ["Familia Inventada"]
    assert any("sin puntaje" in a for a in resultado.warnings)


def test_los_puntajes_provisionales_quedan_marcados():
    """Hyriidae y Miridae no están en la tabla BMWP/Col; su puntaje es por
    proximidad y debe declararse como tal."""
    calc = BMWPCalculator()

    resultado = calc.calculate_site([_det("Hyriidae"), _det("Miridae")])

    assert set(resultado.provisional_families) == {"Hyriidae", "Miridae"}
    assert any("provisional" in a.lower() for a in resultado.warnings)


# --- B3: los puntajes respetan el sentido ecológico -----------------------

def test_los_taxones_tolerantes_puntuan_menos_que_los_sensibles():
    """Chironomidae e Hirudinidae son indicadores de deterioro. Si puntúan
    alto, el índice se invierte y un arroyo contaminado da 'agua limpia'."""
    calc = BMWPCalculator()

    tolerantes = ["Chironomidae", "Hirudinidae", "Physidae", "Psychodidae"]
    sensibles = ["Dytiscidae", "Gerridae", "Coenagrionidae", "Notonectidae"]

    peor_sensible = min(calc.get_family_score(f) for f in sensibles)
    mejor_tolerante = max(calc.get_family_score(f) for f in tolerantes)

    assert mejor_tolerante < peor_sensible


def test_un_sitio_tolerante_puntua_peor_que_uno_sensible():
    calc = BMWPCalculator()

    degradado = calc.calculate_site([
        _det("Chironomidae"), _det("Physidae"), _det("Hirudinidae")
    ])
    saludable = calc.calculate_site([
        _det("Dytiscidae"), _det("Gerridae"), _det("Coenagrionidae")
    ])

    assert degradado.total_score < saludable.total_score


# --- B4: el índice es por sitio, no por fotografía ------------------------

def test_aggregate_images_reune_las_fotos_en_un_sitio():
    calc = BMWPCalculator()

    sitio = calc.aggregate_images([
        [_det("Dytiscidae", cantidad=2)],
        [_det("Gerridae", cantidad=1)],
        [_det("Dytiscidae", cantidad=3)],
    ])

    por_familia = {d["familia"]: d["cantidad"] for d in sitio}
    assert por_familia == {"Dytiscidae": 5, "Gerridae": 1}

    resultado = calc.calculate_site(sitio)
    assert resultado.total_score == 9 + 8
    assert resultado.n_families_scored == 2


def test_sin_detecciones_el_indice_no_es_determinable():
    """Ausencia de evidencia no es evidencia de mala calidad. Declarar
    'muy crítica' ante una foto vacía inventa un veredicto ambiental."""
    calc = BMWPCalculator()

    resultado = calc.calculate_site([])

    assert resultado.water_quality_class == "N/D"
    assert "no equivale a mala calidad" in " ".join(resultado.warnings)


def test_pocas_familias_avisan_que_el_indice_es_poco_informativo():
    calc = BMWPCalculator()

    resultado = calc.calculate_site([_det("Dytiscidae")])

    assert any("poco informativo" in a for a in resultado.warnings)


# --- ASPT -----------------------------------------------------------------

def test_aspt_es_el_promedio_por_familia():
    calc = BMWPCalculator()

    resultado = calc.calculate_site([_det("Dytiscidae"), _det("Chironomidae")])

    assert resultado.total_score == 11
    assert resultado.aspt == 5.5


def test_aspt_no_sube_por_mas_esfuerzo_de_muestreo_en_sitio_homogeneo():
    """El BMWP crece al sumar familias; el ASPT no, y por eso es comparable
    entre sitios muestreados con distinto esfuerzo."""
    calc = BMWPCalculator()

    poco = calc.calculate_site([_det("Physidae"), _det("Psychodidae")])
    mucho = calc.calculate_site([
        _det("Physidae"), _det("Psychodidae"),
        _det("Hydrophilidae"), _det("Glossiphoniidae"),
    ])

    assert mucho.total_score > poco.total_score
    assert mucho.aspt == poco.aspt == 3.0
