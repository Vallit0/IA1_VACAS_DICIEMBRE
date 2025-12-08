package algorithms

import (
	"gonum.org/v1/gonum/mat"
)

type SoftmaxRegression struct {
	W            *mat.Dense // [nFeatures x nClasses]
	LearningRate float64
	RegLambda    float64
	nFeatures    int // N Features de Entrada
	nClasses     int // N Clases de Salida
}

// Python linear.LinearRegression()
// model.fit
