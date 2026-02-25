package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/Ryvion/node-agent/internal/inference"
)

func main() {
	infMgr := inference.New("/Users/caspian/.ryvion")
	go infMgr.Start(context.Background())
	time.Sleep(3 * time.Second)

	fmt.Println("Switching to phi-4...")
	err := infMgr.EnsureModel(context.Background(), "phi-4")
	if err != nil {
		fmt.Println("Failed to switch model:", err)
		os.Exit(1)
	}
	fmt.Println("Model switched.")

	payload := []byte(`{
		"model": "phi-4",
		"messages": [
			{"role": "user", "content": "7*9"},
			{"role": "assistant", "content": "7 * 9 = 63"},
			{"role": "user", "content": "How to learn Japanese?"},
			{"role": "assistant", "content": "Learning Japanese can be a challenging but rewarding experience..."},
			{"role": "user", "content": "What is the best websites?"}
		],
		"stream": true,
		"max_tokens": 1024
	}`)

	req, _ := http.NewRequest("POST", "http://localhost:8081/v1/chat/completions", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")

	fmt.Println("Sending request...")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println("HTTP Error:", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	fmt.Println("Status:", resp.Status)
	io.Copy(os.Stdout, resp.Body)
}
