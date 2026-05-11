package main

import (
	"net/http"
	"os"
	"strconv"
	"testing"
)

const llamaCppHelperEnv = "RYVION_TEST_LLAMA_SERVER_HELPER"

func TestMain(m *testing.M) {
	if os.Getenv(llamaCppHelperEnv) == "1" {
		runTestLlamaServer()
		return
	}
	os.Exit(m.Run())
}

func runTestLlamaServer() {
	host := "127.0.0.1"
	port := ""
	for i := 1; i < len(os.Args)-1; i++ {
		switch os.Args[i] {
		case "--host":
			host = os.Args[i+1]
		case "--port":
			port = os.Args[i+1]
		}
	}
	if _, err := strconv.Atoi(port); err != nil {
		os.Exit(2)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []any{}})
	})
	if err := http.ListenAndServe(host+":"+port, mux); err != nil {
		os.Exit(1)
	}
}
