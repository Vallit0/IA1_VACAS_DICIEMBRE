% Inferencia
recomendar_medicamento(Urg, Enf, Cron, Pecho, Resp, Med) :-
    medicamento_contraindicado(Urg, Enf, Cron, Pecho, Resp, Med).

% Instrucción automática
:- initialization(main).

main :-
    recomendar_medicamento(alta, asma, cronica_si, pecho_si, resp_si, M),
    format("Medicamento contraindicado: ~w~n", [M]),
    halt.
