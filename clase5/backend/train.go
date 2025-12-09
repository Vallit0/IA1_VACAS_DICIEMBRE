package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"

	"unmatch/backend/algorithms"
)

// SoftmaxToyTest entrena el modelo Softmax con un dataset
// muy simple de 3 clases en 2D y genera archivos CSV para
// que puedas graficar los puntos y sus probabilidades.
func TrainSoftmax() error {
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
