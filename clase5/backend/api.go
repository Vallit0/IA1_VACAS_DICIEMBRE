package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/mndrix/golog"
	"gonum.org/v1/gonum/mat"

	"unmatch/backend/algorithms"
)

// estructura que abstrae el MedicamentoRecomendado
type MedicamentoRecomendado struct {
	Urgencia    string
	Enfermedad  string
	Cronica     string
	Pecho       string
	Respiracion string
	Medicamento string
	Match       float64
}

func recomendarMedicamentos(
	m golog.Machine,
	urgencia, enfermedad, cronica, pecho, respiracion string,
) []MedicamentoRecomendado {

	// Usamos la regla de inferencia, no directamente la base de hechos.
	query := "recomendar_medicamento(Urg, Enf, Cron, Pecho, Resp, Med)."
	solutions := m.ProveAll(query)

	results := []MedicamentoRecomendado{}

	for _, sol := range solutions {
		urg := sol.ByName_("Urg").String()
		enf := sol.ByName_("Enf").String()
		cron := sol.ByName_("Cron").String()
		pech := sol.ByName_("Pecho").String()
		resp := sol.ByName_("Resp").String()
		med := sol.ByName_("Med").String()

		matchCount := 0

		// 1: urgencia
		if urg == urgencia {
			matchCount++
		}
		// 2: enfermedad
		if enf == enfermedad {
			matchCount++
		}
		// 3: cronicidad
		if cron == cronica {
			matchCount++
		}
		// 4: pecho
		if pech == pecho {
			matchCount++
		}
		// 5: respiración
		if resp == respiracion {
			matchCount++
		}

		// 6: “peso extra” por combinación peligrosa (ejemplos)
		// alta urgencia + problema respiratorio
		if urgencia == "alta" && respiracion == "resp_si" {
			matchCount++
		}
		// enfermedad pulmonar crónica + respiración comprometida
		if (enfermedad == "asma" || enfermedad == "bronquitis" || enfermedad == "enfisema") &&
			cronica == "cronica_si" && respiracion == "resp_si" {
			matchCount++
		}

		// Tenemos 5 condiciones base + 2 extras posibles = divisor 7.0
		matchPercent := float64(matchCount) / 7.0 * 100.0

		results = append(results, MedicamentoRecomendado{
			Urgencia:    urg,
			Enfermedad:  enf,
			Cronica:     cron,
			Pecho:       pech,
			Respiracion: resp,
			Medicamento: med,
			Match:       matchPercent,
		})
	}

	return results
}

// Agregar numeros para recomendacion
func cargarProlog(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	return string(data)
}

type PerfilEstudiante struct {
	Aptitud    string `json:"aptitud"`
	Habilidad  string `json:"habilidad"`
	Interes    string `json:"interes"`
	Habilidad2 string `json:"habilidad2"`
	Interes2   string `json:"interes2"`
}

type CarreraRecomendada struct {
	Facultad string  `json:"facultad"`
	Carrera  string  `json:"carrera"`
	Match    float64 `json:"match"`
}

type DiagnosticoRequest struct {
	Texto string `json:"texto"`
}

// ===== Tipos para el modelo Softmax =====

type SoftmaxTrainRequest struct {
	X         [][]float64 `json:"x"`          // matriz nSamples x nFeatures
	Y         []int       `json:"y"`          // etiquetas enteras 0..K-1
	Lr        float64     `json:"lr"`         // opcional, default 0.1
	NIter     int         `json:"n_iter"`     // opcional, default 2000
	RegLambda float64     `json:"reg_lambda"` // opcional, default 1e-3
}

type SoftmaxPredictRequest struct {
	X [][]float64 `json:"x"` // matriz nSamples x nFeatures
}

var softmaxModel *algorithms.SoftmaxRegression

const softmaxModelPath = algorithms.DefaultSoftmaxModelPath

// helpers para convertir entre [][]float64 y *mat.Dense
func slice2DToDense(x [][]float64) (*mat.Dense, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("la matriz X no puede estar vacía")
	}
	nSamples := len(x)
	nFeatures := len(x[0])
	data := make([]float64, 0, nSamples*nFeatures)
	for i := 0; i < nSamples; i++ {
		if len(x[i]) != nFeatures {
			return nil, fmt.Errorf("todas las filas de X deben tener el mismo número de columnas")
		}
		data = append(data, x[i]...)
	}
	return mat.NewDense(nSamples, nFeatures, data), nil
}

func denseTo2D(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := m.RawRowView(i)
		dst := make([]float64, c)
		copy(dst, row)
		out[i] = dst
	}
	return out
}

// Esta funcion llama a HuggingFace
func llamarHuggingFace(texto string) (interface{}, error) {
	fmt.Print("HuggingFaceCall")
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		return nil, fmt.Errorf("la variable de entorno HF_TOKEN no está configurada")
	}

	url := "https://router.huggingface.co/hf-inference/models/PlanTL-GOB-ES/bsc-bio-ehr-es"

	payload, err := json.Marshal(map[string]string{
		"inputs": texto,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("error de HuggingFace: %s - %s", resp.Status, string(body))
	}

	var resultado interface{}
	if err := json.Unmarshal(body, &resultado); err != nil {
		return nil, err
	}

	return resultado, nil
}

// limpiar datos hasta llegar aca
type vectorEntrada struct {
	a_asma              float32
	a_bronquitis        float32
	a_enfisema          float32
	a_apnea             float32
	a_fibromialgia      float32
	a_migranas          float32
	a_reflujo           float32
	n_sintomas          int
	n_cronicas          int
	redflag_pecho       bool
	redflag_respiracion bool
	tiene_cronicas      bool
}

// estructura para el analisis de texto
type featuresText struct {
	n_sintomas          int
	n_cronicas          int
	redflag_pecho       bool
	redflag_respiracion bool
	tiene_cronicas      bool
}

// keywords en un slice
var sintomasKeywords = []string{
	"pecho",
	"tos",
	"flema",
	"silbido",
	"falta de aire",
	"ahogo",
	"dificultad para respirar",
	"opresion",
	"dolor al respirar",
}

var cronicasKeywords = []string{
	"asma",
	"epoc",
	"bronquitis cronica",
	"fibrosis pulmonar",
	"enfisema",
}

// Funcion que sea util para analizar el texto
// Diccionario con los sintomas y enfermedades cronicas
func analizarTexto(texto string) featuresText {
	// Hacer el analisis del texto
	// Agregar un for para recorrer el texto y contar sintomas
	// en esta funcion vamos a analizar el texto y devolver un struct featuresText
	// Agregamos valores dummy al vector
	var vector featuresText
	vector.n_sintomas = 3
	vector.n_cronicas = 1
	vector.redflag_pecho = true
	vector.redflag_respiracion = false
	vector.tiene_cronicas = true
	// Aqui va el analisis del texto

	// Logica de analisis
	return vector
}

// cambiar funcion luego
func main() {
	app := fiber.New()
	// Golog - Levanta un objeto que se llama Maquina de Inferencia
	// instanciar Vector de Entrada
	var entrada vectorEntrada
	// resolver problema de CORS
	app.Use(func(c *fiber.Ctx) error {
		c.Set("Access-Control-Allow-Origin", "*")
		c.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Method() == fiber.MethodOptions {
			return c.SendStatus(fiber.StatusOK)
		}

		return c.Next()
	})
	// vecotres todo el proceso el proceso
	// variables globales
	programa := cargarProlog("./prolog/conocimiento.pl")
	// Cargar las inferencias
	m := golog.NewMachine()
	m = m.Consult(programa)

	// Intentar cargar el modelo Softmax desde disco (si existe)
	if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
		softmaxModel = model
		fmt.Println("Modelo Softmax cargado desde", softmaxModelPath)
	} else {
		fmt.Println("Modelo Softmax no cargado (aún). Entrénelo vía /softmax/train")
	}

	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendString("Servidor UniMatch funcionando 🧠")
	})
	// Entrenar modelo Softmax con datos enviados por el cliente
	app.Post("/softmax/train", func(c *fiber.Ctx) error {
		var req SoftmaxTrainRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Error de entrada."})
		}
		if len(req.X) == 0 || len(req.Y) == 0 {
			return c.Status(400).JSON(fiber.Map{"error": "X e y son requeridos."})
		}
		if len(req.X) != len(req.Y) {
			return c.Status(400).JSON(fiber.Map{"error": "X e y deben tener el mismo número de filas."})
		}

		lr := req.Lr
		if lr == 0 {
			lr = 0.1
		}
		nIter := req.NIter
		if nIter == 0 {
			nIter = 2000
		}
		reg := req.RegLambda
		if reg == 0 {
			reg = 1e-3
		}

		Xmat, err := slice2DToDense(req.X)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
		}

		model := algorithms.NewSoftmaxRegression(lr, nIter, reg)
		model.Fit(Xmat, req.Y)
		acc := model.Accuracy(Xmat, req.Y)

		softmaxModel = model
		if err := model.SaveToFile(softmaxModelPath); err != nil {
			fmt.Println("Error al guardar el modelo Softmax:", err)
		}

		return c.JSON(fiber.Map{
			"mensaje":  "Modelo Softmax entrenado",
			"accuracy": acc,
		})
	})

	// Usar el modelo Softmax entrenado para predecir
	app.Post("/softmax/predict", func(c *fiber.Ctx) error {
		var req SoftmaxPredictRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Error de entrada."})
		}
		if len(req.X) == 0 {
			return c.Status(400).JSON(fiber.Map{"error": "X es requerido."})
		}

		if softmaxModel == nil {
			// intentar cargar desde disco por si se entrenó antes
			if model, err := algorithms.LoadSoftmaxRegression(softmaxModelPath); err == nil {
				softmaxModel = model
			} else {
				return c.Status(400).JSON(fiber.Map{"error": "Modelo no entrenado. Primero llame a /softmax/train."})
			}
		}

		Xmat, err := slice2DToDense(req.X)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": err.Error()})
		}

		yPred := softmaxModel.Predict(Xmat)
		probsMat := softmaxModel.PredictProba(Xmat)
		probs := denseTo2D(probsMat)

		return c.JSON(fiber.Map{
			"y_pred": yPred,
			"probs":  probs,
		})
	})

	// Diagnostico de Texto Medico
	app.Post("/diagnostico", func(c *fiber.Ctx) error {
		var req DiagnosticoRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "Error de entrada."})
		}
		if req.Texto == "" {
			return c.Status(400).JSON(fiber.Map{"error": "El campo 'texto' es requerido."})
		}
		var textoEntrada = req.Texto
		// le agregamos al texto la <mask> del modelo
		req.Texto = req.Texto + " padezco de <mask>."

		respuesta, err := llamarHuggingFace(req.Texto)
		if err != nil {
			fmt.Println("Error al llamar a HuggingFace:", err)
			return c.Status(500).JSON(fiber.Map{"error": "Error al consultar HuggingFace", "detalle": err.Error()})
		}
		// tomar respuesta y asignarlo al vector de entrada
		entrada.a_asma = respuesta.(map[string]interface{})["asma"].(float32)
		entrada.a_bronquitis = respuesta.(map[string]interface{})["bronquitis"].(float32)
		entrada.a_enfisema = respuesta.(map[string]interface{})["enfisema"].(float32)
		entrada.a_apnea = respuesta.(map[string]interface{})["apnea"].(float32)
		entrada.a_fibromialgia = respuesta.(map[string]interface{})["fibromialgia"].(float32)
		entrada.a_migranas = respuesta.(map[string]interface{})["migrañas"].(float32)
		entrada.a_reflujo = respuesta.(map[string]interface{})["reflujo"].(float32)
		// Recorra el texto y vaya generando los insights
		// llamamos a la funcion analizarTexto
		vectorsito_con_texto := analizarTexto(textoEntrada)
		// luego de tener el vector proyecto
		// trasladamos de proyecto a entrada
		entrada.n_sintomas = vectorsito_con_texto.n_sintomas
		entrada.n_cronicas = vectorsito_con_texto.n_cronicas
		entrada.redflag_pecho = vectorsito_con_texto.redflag_pecho
		entrada.redflag_respiracion = vectorsito_con_texto.redflag_respiracion
		entrada.tiene_cronicas = vectorsito_con_texto.tiene_cronicas
		// Convertimos el VectorEntrada a algo que pueda usar el modelo Softmax
		// 1. Crear matriz Gonum con los datos del vector de entrada
		Xdata := []float64{
			float64(entrada.a_asma),
			float64(entrada.a_bronquitis),
			float64(entrada.a_enfisema),
			float64(entrada.a_apnea),
			float64(entrada.a_fibromialgia),
			float64(entrada.a_migranas),
			float64(entrada.a_reflujo),
			float64(entrada.n_sintomas),
			float64(entrada.n_cronicas),
		}
		// aunque mapeamos a
		Xmat := mat.NewDense(1, len(Xdata), Xdata)

		// ingresamos al modelo Softmax
		// 1. Crear la matriz Gonum con los datos del vector de entrada
		inferencias_softmax := softmaxModel.Predict(Xmat)
		// luego adaptamos los resultados al esquema de entrada de Prolog
		var urgencia, enfermedad, cronica, pecho, respiracion string
		// usando la funcion recomendarMedicacion
		switch inferencias_softmax[0] {
		case 0:
			urgencia = "baja"
			enfermedad = "ninguna"
			cronica = "cronica_no"
		case 1:
			urgencia = "media"
			enfermedad = "asma"
			cronica = "cronica_si"
		case 2:
			urgencia = "alta"
			enfermedad = "bronquitis"
			cronica = "cronica_si"
		default:
			urgencia = "baja"
			enfermedad = "ninguna"
			cronica = "cronica_no"
		}
		if entrada.redflag_pecho {
			pecho = "pecho_si"
		} else {
			pecho = "pecho_no"
		}
		if entrada.redflag_respiracion {
			respiracion = "resp_si"
		} else {
			respiracion = "resp_no"
		}
		// 2. Llamar a recomendarMedicacion
		resultados := recomendarMedicamentos(
			m,
			urgencia,
			enfermedad,
			cronica,
			pecho,
			respiracion,
		)
		fmt.Println("Resultados de la recomendación:", resultados)
		// preparar la respuesta para devolverla al cliente

		// ahora le pasamos a prolog todos los datos

		// 2. Asegurarse que el modelo Softmax esté cargado
		// 3. Pasar el vector al modelo softmaxModel.Predict

		// 4. Recuperamos todos los datos del informe
		// 5. Pasamos a Prolog
		// 6. Devolvemos la respuesta al cliente
		// 7. Ustedes aca proponen una accion con el RPA
		return c.JSON(fiber.Map{
			"resultado": respuesta,
		})
	})

	//
	fmt.Println("Servidor UniMatch iniciado en el puerto 8080")
	app.Listen(":8080")
}
