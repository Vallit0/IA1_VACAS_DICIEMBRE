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

func recomendarCarreras(m golog.Machine, aptitud, habilidad1, interes1, habilidad2, interes2 string) []CarreraRecomendada {
	query := "carrera(Fac, Carr, Apt, Hab, Int)."
	solutions := m.ProveAll(query)

	results := []CarreraRecomendada{}

	for _, sol := range solutions {
		apt := sol.ByName_("Apt").String()
		hab := sol.ByName_("Hab").String()
		ints := sol.ByName_("Int").String()

		matchCount := 0
		if apt == aptitud {
			matchCount++
		}
		if hab == habilidad1 || hab == habilidad2 {
			matchCount++
		}
		if ints == interes1 || ints == interes2 {
			matchCount++
		}
		if habilidad1 == ints || habilidad2 == apt || interes1 == hab || interes2 == hab {
			matchCount++ // peso extra si hay cruce interesante
		}

		matchPercent := float64(matchCount) / 5.0 * 100.0

		results = append(results, CarreraRecomendada{
			Facultad: sol.ByName_("Fac").String(),
			Carrera:  sol.ByName_("Carr").String(),
			Match:    matchPercent,
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

// cambiar funcion luego
func xd() {
	app := fiber.New()
	// Golog - Levanta un objeto que se llama Maquina de Inferencia
	//
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

	app.Post("/recomendar", func(c *fiber.Ctx) error {
		var perfil PerfilEstudiante
		if err := c.BodyParser(&perfil); err != nil {
			return c.Status(400).SendString("Error de entrada.")
		}
		fmt.Println("Perfil recibido:", perfil)
		resultados := recomendarCarreras(m, perfil.Aptitud, perfil.Habilidad, perfil.Interes, perfil.Interes2, perfil.Habilidad2)
		if len(resultados) == 0 {
			return c.JSON(fiber.Map{"mensaje": "No se encontraron coincidencias."})
		}

		return c.JSON(fiber.Map{
			"recomendaciones": resultados,
		})
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
		// le agregamos al texto la <mask> del modelo
		req.Texto = req.Texto + " padezco de <mask>."

		respuesta, err := llamarHuggingFace(req.Texto)
		if err != nil {
			fmt.Println("Error al llamar a HuggingFace:", err)
			return c.Status(500).JSON(fiber.Map{"error": "Error al consultar HuggingFace", "detalle": err.Error()})
		}
		// 1. Jalar los datos y meterlos al vector de entrada
		// 2. Analizar el texto y hacer feature Engineering
		// vector_a := analizarTexto(req.Texto)
		//
		// 3. Pasar el vector al modelo softmaxModel.Predict
		// y_pred := softmaxModel.Predict(vector_a)
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
