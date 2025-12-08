package algorithms

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// SoftmaxRegression es una regresión logística multinomial (Softmax)
// implementada con gonum/mat
type SoftmaxRegression struct {
	W         *mat.Dense    // (nFeatures x nClasses)
	B         *mat.VecDense // (nClasses)
	Lr        float64
	NIter     int
	RegLambda float64
}

// NewSoftmaxRegression crea el modelo con hiperparámetros
func NewSoftmaxRegression(lr float64, nIter int, regLambda float64) *SoftmaxRegression {
	return &SoftmaxRegression{
		Lr:        lr,
		NIter:     nIter,
		RegLambda: regLambda,
	}
}

// oneHotDense construye Y en one-hot: (nSamples x nClasses)
func oneHotDense(y []int, nSamples, nClasses int) *mat.Dense {
	Y := mat.NewDense(nSamples, nClasses, nil)
	for i := 0; i < nSamples; i++ {
		Y.Set(i, y[i], 1.0)
	}
	return Y
}

// softmaxRows aplica softmax fila a fila con estabilidad numérica.
// scores: (n x K) -> probs: (n x K)
func softmaxRows(scores *mat.Dense) *mat.Dense {
	// Crea una matriz densa
	r, c := scores.Dims()
	out := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		// nuestra matriz de entrada
		row := scores.RawRowView(i)
		// nuestra matriz de salida
		outRow := out.RawRowView(i)

		// 1) max por fila
		maxVal := row[0]
		for k := 1; k < c; k++ {
			if row[k] > maxVal {
				maxVal = row[k]
			}
		}

		// 2) exp(s - max) y suma
		sumExp := 0.0
		for k := 0; k < c; k++ {
			e := math.Exp(row[k] - maxVal)
			outRow[k] = e
			sumExp += e
		}

		// 3) normalizar
		for k := 0; k < c; k++ {
			outRow[k] /= sumExp
		}
	}
	return out
}

// forward: scores = XW + b, probs = softmax(scores)
// X: (nSamples x nFeatures)
func (m *SoftmaxRegression) forward(X *mat.Dense) (*mat.Dense, *mat.Dense) {
	nSamples, _ := X.Dims()
	_, nClasses := m.W.Dims()

	// scores = X * W
	scores := mat.NewDense(nSamples, nClasses, nil)
	scores.Mul(X, m.W)

	// sumar bias b a cada fila
	for i := 0; i < nSamples; i++ {
		row := scores.RawRowView(i)
		for k := 0; k < nClasses; k++ {
			row[k] += m.B.AtVec(k)
		}
	}

	// probs = softmax(scores)
	probs := softmaxRows(scores)
	return scores, probs
}

// Fit entrena el modelo sobre X (n x d) y y (n,)
func (m *SoftmaxRegression) Fit(X *mat.Dense, y []int) {
	nSamples, nFeatures := X.Dims()
	if nSamples == 0 {
		log.Fatal("Fit: X vacío")
	}
	if len(y) != nSamples {
		log.Fatal("Fit: X y y con distinto número de muestras")
	}

	// número de clases = max(y) + 1
	nClasses := 0
	for _, yi := range y {
		if yi+1 > nClasses {
			nClasses = yi + 1
		}
	}

	// inicializar W y B
	rand.Seed(time.Now().UnixNano())

	if m.W == nil {
		dataW := make([]float64, nFeatures*nClasses)
		for i := range dataW {
			dataW[i] = 0.01 * rand.NormFloat64()
		}
		m.W = mat.NewDense(nFeatures, nClasses, dataW)
	}
	if m.B == nil {
		m.B = mat.NewVecDense(nClasses, nil)
	}

	// one-hot de y
	Y := oneHotDense(y, nSamples, nClasses)

	for iter := 0; iter < m.NIter; iter++ {
		// ==== FORWARD ====
		_, probs := m.forward(X) // probs: (n x K)

		// ==== dScores = (probs - Y)/n ====
		dScores := mat.NewDense(nSamples, nClasses, nil)
		dScores.Sub(probs, Y)
		dScores.Scale(1.0/float64(nSamples), dScores)

		// ==== dW = Xᵀ * dScores + λW ====
		var XT mat.Dense
		XT.CloneFrom(X.T()) // (d x n)

		dW := mat.NewDense(nFeatures, nClasses, nil)
		dW.Mul(&XT, dScores) // (d x n)*(n x K) = (d x K)

		if m.RegLambda > 0 {
			var regW mat.Dense
			regW.CloneFrom(m.W)
			regW.Scale(m.RegLambda, &regW)
			dW.Add(dW, &regW)
		}

		// ==== db = suma filas de dScores ====
		dbData := make([]float64, nClasses)
		for i := 0; i < nSamples; i++ {
			row := dScores.RawRowView(i)
			for k := 0; k < nClasses; k++ {
				dbData[k] += row[k]
			}
		}
		db := mat.NewVecDense(nClasses, dbData)

		// ==== UPDATE ====
		// W = W - lr * dW
		var scaledDW mat.Dense
		scaledDW.Scale(m.Lr, dW)
		m.W.Sub(m.W, &scaledDW)

		// B = B - lr * db
		var scaledDB mat.VecDense
		scaledDB.ScaleVec(m.Lr, db)
		m.B.SubVec(m.B, &scaledDB)
	}
}

// PredictProba devuelve matriz (n x K) con probabilidades
func (m *SoftmaxRegression) PredictProba(X *mat.Dense) *mat.Dense {
	if m.W == nil || m.B == nil {
		log.Fatal("PredictProba: modelo no entrenado")
	}
	_, probs := m.forward(X)
	return probs
}

// Predict devuelve el argmax de cada fila de probs
func (m *SoftmaxRegression) Predict(X *mat.Dense) []int {
	probs := m.PredictProba(X)
	nSamples, nClasses := probs.Dims()
	yPred := make([]int, nSamples)

	for i := 0; i < nSamples; i++ {
		row := probs.RawRowView(i)
		maxIdx := 0
		maxVal := row[0]
		for k := 1; k < nClasses; k++ {
			if row[k] > maxVal {
				maxVal = row[k]
				maxIdx = k
			}
		}
		yPred[i] = maxIdx
	}
	return yPred
}

// Accuracy calcula porcentaje de aciertos
func (m *SoftmaxRegression) Accuracy(X *mat.Dense, y []int) float64 {
	yPred := m.Predict(X)
	if len(yPred) != len(y) {
		log.Fatal("Accuracy: tamaños distintos")
	}
	correct := 0
	for i := range y {
		if yPred[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}

// ===== Ejemplo mínimo =====

func main() {
	// Dataset ultra simple: 3 clases en 2D
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

	model := NewSoftmaxRegression(
		0.1,  // lr
		2000, // iteraciones
		1e-3, // lambda
	)

	model.Fit(X, y)

	acc := model.Accuracy(X, y)
	fmt.Printf("Accuracy training: %.4f\n", acc)

	// Probamos con puntos nuevos
	XtestData := []float64{
		-1.0, -0.8,
		0.1, 1.2,
		2.1, 2.0,
	}
	Xtest := mat.NewDense(3, 2, XtestData)

	probs := model.PredictProba(Xtest)
	yPred := model.Predict(Xtest)

	fmt.Println("Predicciones:")
	for i := 0; i < 3; i++ {
		row := probs.RawRowView(i)
		fmt.Printf("x=%v -> probs=%v, y_pred=%d\n",
			Xtest.RawRowView(i),
			row,
			yPred[i],
		)
	}
}
