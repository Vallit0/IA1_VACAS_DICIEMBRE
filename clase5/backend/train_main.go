package main

import "fmt"

func main() {
	if err := TrainSoftmaxBronco(); err != nil {
		fmt.Println("TrainSoftmaxBronco error:", err)
	}
}
