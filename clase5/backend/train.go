package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"

	"unmatch/backend/algorithms"
)

// SoftmaxToyTest entrena el modelo Softmax con un dataset
// muy simple de 3 clases en 2D y genera archivos CSV para
// que puedas graficar los puntos y sus probabilidades.
func TrainSoftmax() error {
	///// Archivo de entrenamiento /////
	// Tomar el CSV con el dataset que les comparti
	// Pasarlo a Vectores de Gonum
	// Entrenar el modelo con .fit
	// guardar el modelo entrenado saveToFile

	/// API /////
	// crear el ednpoint con la importacion del modelo entrenado
	// Pasarle el vector de entrada
	// predecir con el modelo entrenado
	// devolver la prediccion al cliente

	// Dataset: 3 clases en 2D
	// Clase 0: alrededor de (-1, -1)
	// Clase 1: alrededor de (0, 1)
	// Clase 2: alrededor de (2, 2)
	Xdata := []float64{
		-1.0, -1.2,
		-0.8, -0.9,
		-1.2, -1.1,

		0.0, 1.0,
		0.2, 0.8,
		-0.1, 1.1,

		2.0, 2.1,
		1.8, 1.9,
		2.2, 2.0,
	}
	y := []int{
		0, 0, 0,
		1, 1, 1,
		2, 2, 2,
	}

	X := mat.NewDense(9, 2, Xdata)

	model := algorithms.NewSoftmaxRegression(0.1, 2000, 1e-3)
	model.Fit(X, y)

	acc := model.Accuracy(X, y)
	fmt.Printf("Accuracy entrenamiento (toy): %.4f\n", acc)

	// Probabilidades en los mismos puntos de entrenamiento
	trainProbs := model.PredictProba(X)

	// Algunos puntos nuevos para ver cómo generaliza
	XtestData := []float64{
		-1.0, -0.8,
		0.1, 1.2,
		2.1, 2.0,
	}
	Xtest := mat.NewDense(3, 2, XtestData)
	testProbs := model.PredictProba(Xtest)

	// Aseguramos que exista la carpeta weights
	_ = os.MkdirAll("./weights", 0o755)

	if err := exportPointsCSV("./weights/softmax_train_points.csv", X, y, trainProbs); err != nil {
		return err
	}
	// Para los puntos de prueba no tenemos etiqueta verdadera, usamos -1
	yTestDummy := []int{-1, -1, -1}
	if err := exportPointsCSV("./weights/softmax_test_points.csv", Xtest, yTestDummy, testProbs); err != nil {
		return err
	}

	fmt.Println("Se generaron:")
	fmt.Println("  - weights/softmax_train_points.csv")
	fmt.Println("  - weights/softmax_test_points.csv")
	fmt.Println("Puedes cargar esos CSV en Python, R, Excel, etc. para graficar.")

	return nil
}

// exportPointsCSV escribe X, y, y probabilidades a un CSV.
// Formato columnas: x1, x2, y_true, y_pred, p0, p1, ..., pK
func exportPointsCSV(path string, X *mat.Dense, y []int, probs *mat.Dense) error {

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	rows, _ := X.Dims()
	_, nClasses := probs.Dims()

	// Encabezado
	header := []string{"x1", "x2", "y_true", "y_pred"}
	for k := 0; k < nClasses; k++ {
		header = append(header, fmt.Sprintf("p%d", k))
	}
	if err := w.Write(header); err != nil {
		return err
	}

	// Filas
	for i := 0; i < rows; i++ {
		xRow := X.RawRowView(i)
		pRow := probs.RawRowView(i)

		yTrue := -1
		if i < len(y) {
			yTrue = y[i]
		}

		// argmax para y_pred
		yPred := 0
		maxVal := pRow[0]
		for k := 1; k < nClasses; k++ {
			if pRow[k] > maxVal {
				maxVal = pRow[k]
				yPred = k
			}
		}

		record := []string{
			fmt.Sprintf("%f", xRow[0]),
			fmt.Sprintf("%f", xRow[1]),
			fmt.Sprintf("%d", yTrue),
			fmt.Sprintf("%d", yPred),
		}
		for k := 0; k < nClasses; k++ {
			record = append(record, fmt.Sprintf("%f", pRow[k]))
		}

		if err := w.Write(record); err != nil {
			return err
		}
	}

	return nil
}

// exportLossCSV escribe el historial de pérdida a un CSV.
// Formato columnas: iter, loss
func exportLossCSV(path string, loss []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if err := w.Write([]string{"iter", "loss"}); err != nil {
		return err
	}

	for i, v := range loss {
		record := []string{
			strconv.Itoa(i),
			fmt.Sprintf("%f", v),
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}

	return nil
}

// TrainSoftmaxBronco entrena el modelo Softmax con el dataset
// bronco_dataset.csv y guarda el modelo y la curva de pérdida.
func TrainSoftmaxBronco() error {
	f, err := os.Open("./algorithms/bronco_dataset.csv")
	if err != nil {
		return fmt.Errorf("no se pudo abrir bronco_dataset.csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return fmt.Errorf("error leyendo bronco_dataset.csv: %w", err)
	}
	if len(records) < 2 {
		return fmt.Errorf("bronco_dataset.csv no tiene suficientes filas")
	}

	header := records[0]
	if len(header) < 2 {
		return fmt.Errorf("bronco_dataset.csv debe tener al menos una feature y la columna de etiqueta")
	}

	// asumimos que la etiqueta es la columna 'urgencia'
	labelIdx := -1
	for i, h := range header {
		if h == "urgencia" {
			labelIdx = i
			break
		}
	}
	if labelIdx == -1 {
		return fmt.Errorf("no se encontró la columna 'urgencia' en bronco_dataset.csv")
	}

	nFeatures := len(header) - 1
	nSamples := len(records) - 1

	Xdata := make([]float64, 0, nSamples*nFeatures)
	y := make([]int, 0, nSamples)

	for _, row := range records[1:] {
		if len(row) != len(header) {
			return fmt.Errorf("todas las filas deben tener %d columnas", len(header))
		}

		for j, val := range row {
			if j == labelIdx {
				lbl, err := strconv.Atoi(val)
				if err != nil {
					return fmt.Errorf("no se pudo convertir etiqueta '%s' a int: %w", val, err)
				}
				y = append(y, lbl)
			} else {
				v, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return fmt.Errorf("no se pudo convertir valor '%s' a float64: %w", val, err)
				}
				Xdata = append(Xdata, v)
			}
		}
	}

	if len(y) != nSamples {
		return fmt.Errorf("se esperaban %d etiquetas y se obtuvieron %d", nSamples, len(y))
	}
	if len(Xdata) != nSamples*nFeatures {
		return fmt.Errorf("dimension de X inconsistente: esperados %d valores, obtenidos %d", nSamples*nFeatures, len(Xdata))
	}

	X := mat.NewDense(nSamples, nFeatures, Xdata)

	model := algorithms.NewSoftmaxRegression(0.1, 3000, 1e-3)
	model.Fit(X, y)

	acc := model.Accuracy(X, y)
	fmt.Printf("Accuracy entrenamiento (bronco): %.4f\n", acc)

	// aseguramos carpeta weights y guardamos modelo compatible con la API
	_ = os.MkdirAll("./weights", 0o755)
	if err := model.SaveToFile(algorithms.DefaultSoftmaxModelPath); err != nil {
		return fmt.Errorf("error al guardar el modelo Softmax: %w", err)
	}

	// exportamos curva de pérdida para graficar
	if len(model.LossHistory) > 0 {
		if err := exportLossCSV("./weights/softmax_bronco_loss.csv", model.LossHistory); err != nil {
			return fmt.Errorf("error al exportar curva de pérdida: %w", err)
		}
		fmt.Println("Se generó: weights/softmax_bronco_loss.csv (iter, loss)")
	}

	return nil
}
