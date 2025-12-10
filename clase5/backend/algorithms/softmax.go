package algorithms

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

const DefaultSoftmaxModelPath = "./weights/softmax_model.json"

// SoftmaxRegression implements multinomial logistic regression (softmax).
type SoftmaxRegression struct {
	W         *mat.Dense    // (nFeatures x nClasses)
	B         *mat.VecDense // (nClasses)
	Lr        float64       // Learning Rate
	NIter     int           // Number of iterations
	RegLambda float64       // Regularization strength
	LossHistory []float64   // Training loss per iteration
}

// NewSoftmaxRegression creates a new model with hyperparameters.
func NewSoftmaxRegression(lr float64, nIter int, regLambda float64) *SoftmaxRegression {
	return &SoftmaxRegression{
		Lr:        lr,
		NIter:     nIter,
		RegLambda: regLambda,
	}
}

// oneHotDense builds Y in one-hot format: (nSamples x nClasses).
func oneHotDense(y []int, nSamples, nClasses int) *mat.Dense {
	Y := mat.NewDense(nSamples, nClasses, nil)
	for i := 0; i < nSamples; i++ {
		Y.Set(i, y[i], 1.0)
	}
	return Y
}

// softmaxRows applies softmax row-wise with numerical stability.
func softmaxRows(scores *mat.Dense) *mat.Dense {
	r, c := scores.Dims()
	out := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		row := scores.RawRowView(i)
		outRow := out.RawRowView(i)

		// max per row // argmax
		maxVal := row[0]
		for k := 1; k < c; k++ {
			if row[k] > maxVal {
				maxVal = row[k]
			}
		}

		// exp(s - max) and sum
		sumExp := 0.0
		for k := 0; k < c; k++ {
			e := math.Exp(row[k] - maxVal)
			outRow[k] = e
			sumExp += e
		}

		// normalize
		for k := 0; k < c; k++ {
			outRow[k] /= sumExp
		}
	}
	return out
}

// forward: scores = XW + b, probs = softmax(scores).
// X: (nSamples x nFeatures)
func (m *SoftmaxRegression) forward(X *mat.Dense) (*mat.Dense, *mat.Dense) {
	nSamples, _ := X.Dims()
	_, nClasses := m.W.Dims()

	scores := mat.NewDense(nSamples, nClasses, nil)
	scores.Mul(X, m.W)

	// add bias b to each row
	for i := 0; i < nSamples; i++ {
		row := scores.RawRowView(i)
		for k := 0; k < nClasses; k++ {
			row[k] += m.B.AtVec(k)
		}
	}

	probs := softmaxRows(scores)
	return scores, probs
}

// Fit trains the model on X (n x d) and y (n,).
// Entrenar el modelo
func (m *SoftmaxRegression) Fit(X *mat.Dense, y []int) {
	// X es el vector de entrada que nosotros tenemos
	nSamples, nFeatures := X.Dims()
	// n muestras y n features
	if nSamples == 0 {
		log.Fatal("Fit: X is empty")
	}
	if len(y) != nSamples {
		log.Fatal("Fit: X and y have different number of samples")
	}

	// number of classes = max(y) + 1
	//
	nClasses := 0
	for _, yi := range y {
		if yi+1 > nClasses {
			nClasses = yi + 1
		}
	}

	// inicializamos con un Random Seed los valores
	// Weights -> b1, b2, ..., bd
	// Bias -> b0
	// y = WX + B
	// W = Matriz de pesos
	// X = vector de entrada
	// B = Vector de Bias

	// initialize W and B
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
	// El objeto recibe K labels (puede recibir cualquier cantida )
	// El objeto tambien recibe n features (cualquier cantidad)
	// y = {bajo, medio, alto} -> {0,1,2}
	// x = {feature1, feature2, feature3, ..., featureN}
	// one-hot labels
	// one-hot encoding de las etiquetas
	// este modelo recibe las labels normales
	// este modelo SI recibe las etiquetas normales
	// y = {amarillo, rojo, azul} -> {0,1,2}
	Y := oneHotDense(y, nSamples, nClasses)

	// Reset loss history for this training run
	m.LossHistory = nil

	// Gradient Descent
	// Algunos otros gradientes que nos permiten saber
	// hacia donde mover los pesos
	// dW, db
	// iter = epochs
	for iter := 0; iter < m.NIter; iter++ {
		_, probs := m.forward(X) // probs: (n x K)

		// Compute cross-entropy loss with optional L2 regularization
		loss := 0.0
		for i := 0; i < nSamples; i++ {
			pRow := probs.RawRowView(i)
			yRow := Y.RawRowView(i)
			for k := 0; k < nClasses; k++ {
				if yRow[k] == 1.0 {
					p := pRow[k]
					if p < 1e-15 {
						p = 1e-15
					}
					loss -= math.Log(p)
				}
			}
		}
		loss /= float64(nSamples)

		if m.RegLambda > 0 {
			rowsW, colsW := m.W.Dims()
			regSum := 0.0
			for i := 0; i < rowsW; i++ {
				row := m.W.RawRowView(i)
				for k := 0; k < colsW; k++ {
					regSum += row[k] * row[k]
				}
			}
			loss += 0.5 * m.RegLambda * regSum
		}
		m.LossHistory = append(m.LossHistory, loss)

		// dScores = (probs - Y)/n
		dScores := mat.NewDense(nSamples, nClasses, nil)
		dScores.Sub(probs, Y)
		dScores.Scale(1.0/float64(nSamples), dScores)

		// dW = X^T * dScores + lambda * W
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

		// db = row-wise sum of dScores
		dbData := make([]float64, nClasses)
		for i := 0; i < nSamples; i++ {
			row := dScores.RawRowView(i)
			for k := 0; k < nClasses; k++ {
				dbData[k] += row[k]
			}
		}
		db := mat.NewVecDense(nClasses, dbData)

		// update W and B
		// W = W - lr * dW
		var scaledDW mat.Dense
		scaledDW.Scale(m.Lr, dW)
		m.W.Sub(m.W, &scaledDW)

		var scaledDB mat.VecDense
		scaledDB.ScaleVec(m.Lr, db)
		m.B.SubVec(m.B, &scaledDB)
	}
}

// PredictProba returns an (n x K) matrix with probabilities.
func (m *SoftmaxRegression) PredictProba(X *mat.Dense) *mat.Dense {
	if m.W == nil || m.B == nil {
		log.Fatal("PredictProba: model not trained")
	}
	_, probs := m.forward(X)
	return probs
}

// Predict returns argmax class index for each row.
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

// Accuracy computes the fraction of correct predictions.
func (m *SoftmaxRegression) Accuracy(X *mat.Dense, y []int) float64 {
	yPred := m.Predict(X)
	if len(yPred) != len(y) {
		log.Fatal("Accuracy: different lengths for predictions and labels")
	}
	correct := 0
	for i := range y {
		if yPred[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}

// ===== Model persistence to disk =====
// ARTEFACTO.
type softmaxModelFile struct {
	NFeatures int       `json:"n_features"`
	NClasses  int       `json:"n_classes"`
	W         []float64 `json:"w"`
	B         []float64 `json:"b"`
	Lr        float64   `json:"lr"`
	NIter     int       `json:"n_iter"`
	RegLambda float64   `json:"reg_lambda"`
}

// SaveToFile saves weights and biases to a JSON file.
func (m *SoftmaxRegression) SaveToFile(path string) error {
	if m.W == nil || m.B == nil {
		return fmt.Errorf("SaveToFile: model not trained")
	}

	nFeatures, nClasses := m.W.Dims()
	dataW := make([]float64, nFeatures*nClasses)
	for i := 0; i < nFeatures; i++ {
		row := m.W.RawRowView(i)
		copy(dataW[i*nClasses:(i+1)*nClasses], row)
	}

	dataB := make([]float64, nClasses)
	for i := 0; i < nClasses; i++ {
		dataB[i] = m.B.AtVec(i)
	}

	fileStruct := softmaxModelFile{
		NFeatures: nFeatures,
		NClasses:  nClasses,
		W:         dataW,
		B:         dataB,
		Lr:        m.Lr,
		NIter:     m.NIter,
		RegLambda: m.RegLambda,
	}

	bytes, err := json.MarshalIndent(fileStruct, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, bytes, 0o644)
}

// LoadSoftmaxRegression loads a model from a JSON file.
func LoadSoftmaxRegression(path string) (*SoftmaxRegression, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var fileStruct softmaxModelFile
	if err := json.Unmarshal(bytes, &fileStruct); err != nil {
		return nil, err
	}

	if len(fileStruct.W) != fileStruct.NFeatures*fileStruct.NClasses {
		return nil, fmt.Errorf("LoadSoftmaxRegression: W dimensions mismatch")
	}
	if len(fileStruct.B) != fileStruct.NClasses {
		return nil, fmt.Errorf("LoadSoftmaxRegression: B dimensions mismatch")
	}

	W := mat.NewDense(fileStruct.NFeatures, fileStruct.NClasses, fileStruct.W)
	B := mat.NewVecDense(fileStruct.NClasses, fileStruct.B)

	model := &SoftmaxRegression{
		W:         W,
		B:         B,
		Lr:        fileStruct.Lr,
		NIter:     fileStruct.NIter,
		RegLambda: fileStruct.RegLambda,
	}
	return model, nil
}

// ===== Minimal example (not used by API, just for reference) =====

func Example() {
	// CSV sintetico
	// pasarlo a un vector de entrada
	// simple 3-class, 2D dataset (same idea as original main)
	Xdata := []float64{
		// Caracteristica 1
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

	// implementar estos metodos en un objetos
	// implementar
	X := mat.NewDense(9, 2, Xdata)
	model := NewSoftmaxRegression(0.1, 2000, 1e-3)
	model.Fit(X, y)
	acc := model.Accuracy(X, y)
	fmt.Printf("training accuracy: %.4f\n", acc)
}
