% ============================================================
% BASE DE CONOCIMIENTO (EJEMPLO DIDÁCTICO, NO CLÍNICO REAL)
% ============================================================

% Caso: enfermedades pulmonares con crisis respiratoria
% Evitar sedantes fuertes y opioides (por depresión respiratoria)

medicamento_contraindicado(alta, asma,      cronica_si, pecho_si, resp_si, sedantes_fuertes).
medicamento_contraindicado(alta, asma,      cronica_si, pecho_no, resp_si, opioides_fuertes).
medicamento_contraindicado(alta, bronquitis,cronica_si, pecho_si, resp_si, sedantes_fuertes).
medicamento_contraindicado(alta, enfisema,  cronica_si, pecho_no, resp_si, opioides_fuertes).
medicamento_contraindicado(alta, apnea,     cronica_si, pecho_no, resp_si, benzodiacepinas).

% Caso: enfermedad pulmonar crónica, aunque urgencia ML no sea alta
% Evitar medicamentos que puedan empeorar la respiración

medicamento_contraindicado(mediana, asma,      cronica_si, pecho_no, resp_si, betabloqueantes_no_selectivos).
medicamento_contraindicado(mediana, bronquitis,cronica_si, pecho_no, resp_si, betabloqueantes_no_selectivos).
medicamento_contraindicado(baja,    enfisema,  cronica_si, pecho_no, resp_si, antitusivos_opioides).

% Caso: dolor de pecho + sospecha pulmonar
% Evitar AINEs en contexto de dolor torácico hasta descartar otras causas

medicamento_contraindicado(mediana, asma,      cronica_no, pecho_si, resp_no, aines_altas_dosis).
medicamento_contraindicado(mediana, bronquitis,cronica_no, pecho_si, resp_no, aines_altas_dosis).

% Caso: apnea del sueño
% Evitar medicamentos que deprimen SNC

medicamento_contraindicado(baja, apnea, cronica_si, pecho_no, resp_no, hipnoticos_fuertes).
medicamento_contraindicado(mediana, apnea, cronica_si, pecho_no, resp_si, opioides_fuertes).

% Caso: fibromialgia o migrañas con baja urgencia
% Ejemplo de medicamentos que se decide NO usar en este flujo

medicamento_contraindicado(baja, fibromialgia, cronica_si, pecho_no, resp_no, corticoides_sistemicos_prolongados).
medicamento_contraindicado(baja, migrañas,     cronica_no, pecho_no, resp_no, triptanes_en_antecedente_pulmonar).

% Caso: reflujo con dolor de pecho
% No dar AINEs que puedan agravar síntomas digestivos

medicamento_contraindicado(mediana, reflujo, cronica_si, pecho_si, resp_no, aines_gastroerosivos).
medicamento_contraindicado(baja,    reflujo, cronica_no, pecho_si, resp_no, aines_gastroerosivos).

% Caso genérico: ALTA urgencia pulmonar → no iniciar nada sedante en el flujo automático

medicamento_contraindicado(alta, asma,      cronica_no, pecho_si, resp_si, cualquier_sedante).
medicamento_contraindicado(alta, bronquitis,cronica_no, pecho_no, resp_si, cualquier_sedante).
medicamento_contraindicado(alta, enfisema,  cronica_no, pecho_no, resp_si, cualquier_sedante).
