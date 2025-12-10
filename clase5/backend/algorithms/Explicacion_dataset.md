# Las variables en el Dataset 
## Son las siguientes 

# a_al azar
a_* son probabilidades predichas (0–1) para cada enfermedad en enfermedades_vocab (vector a).


BERT les va a devolver un JSON. Recorren el json o de alguna forma sacan los valores. 
Bronquitis: 0.0032
a_bronquitis = 0.0032 
asma: 0.0043 

a_asma = 0.00043

# Vector de probabilidades
vector_a = 
[
a_asma 
a_bronquitis 
a_enfisema 
]

# numero_sintomas 
n_sintomas = conteo de sintomas_keywords.

n_cronicas = conteo de cronicas_keywords.

redflag_pecho y redflag_respiracion son booleanos (0/1).

tiene_cronicas = 1 si n_cronicas > 0.

urgencia es la target (0 = baja, 1 = media, 2 = alta), coherente con red flags, #síntomas y #crónicas.


# Frontend 
## POST 
### POST A BERT 
#### BERT les devuelve scores 
#### ustedes meten los scores a un vector 
##### VECTOR A el vector de scores 

##### Analia