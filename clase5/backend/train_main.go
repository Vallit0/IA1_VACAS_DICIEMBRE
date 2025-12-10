package main

import "fmt"

func train() {
	if err := TrainSoftmaxBronco(); err != nil {
		fmt.Println("TrainSoftmaxBronco error:", err)
	}
}
