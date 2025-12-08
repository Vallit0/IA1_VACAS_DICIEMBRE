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

func main() {
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

		return c.JSON(fiber.Map{
			"resultado": respuesta,
		})
	})

	fmt.Println("Servidor UniMatch iniciado en el puerto 8080")
	app.Listen(":8080")
}
