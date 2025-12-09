package main

import "fmt"

func main() {
	if err := TrainSoftmax(); err != nil {
		fmt.Println("SoftmaxToyTest error:", err)
	}
}
